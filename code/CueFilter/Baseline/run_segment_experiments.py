import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader

BASELINE_ROOT = Path(__file__).resolve().parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from audio_views import DATASET_CONFIGS, normalize_variant_name
from classical_features import extract_handcrafted_features_batch
from experiment_utils import (
    RecordingSplitDataManager,
    compute_metrics,
    finalize_predictions,
    format_mean_std,
    set_seed,
)
from models import (
    DALFConfig,
    DALFNet,
    DMPFConfig,
    DMPFRegressor,
    DepAudioNetBackbone,
    DisNetConfig,
    DisNetRegressor,
    STFNModel,
)


MODEL_DEFAULTS: Dict[str, Dict[str, float]] = {
    "SVM": {"batch_size": 1, "max_epochs": 1, "learning_rate": 0.0, "weight_decay": 0.0, "patience": 1},
    "RF": {"batch_size": 1, "max_epochs": 1, "learning_rate": 0.0, "weight_decay": 0.0, "patience": 1},
    "DepAudioNet": {"batch_size": 8, "max_epochs": 80, "learning_rate": 5e-4, "weight_decay": 1e-4, "patience": 12},
    "DisNet": {"batch_size": 2, "max_epochs": 100, "learning_rate": 1e-4, "weight_decay": 1e-4, "patience": 15},
    "DMPF": {"batch_size": 4, "max_epochs": 60, "learning_rate": 1e-3, "weight_decay": 1e-4, "patience": 12},
    "DALF": {"batch_size": 8, "max_epochs": 100, "learning_rate": 5e-4, "weight_decay": 1e-4, "patience": 15},
    "STFN": {"batch_size": 4, "max_epochs": 120, "learning_rate": 5e-4, "weight_decay": 1e-4, "patience": 15},
}


CLASSICAL_FEATURE_CACHE: Dict[Tuple[object, ...], np.ndarray] = {}
CLASSICAL_CACHE_DIR = BASELINE_ROOT / ".cache" / "classical_features"
DISPLAY_VARIANTS = {
    "pre": "pre",
    "cue-only": "cue-only",
    "cue-excluded": "cue-removed",
    "random": "random-removed",
}


def evaluate_regressor(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    predictor: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    scaler: StandardScaler,
    eval_level: str = "recording",
) -> Tuple[float, Dict[str, float]]:
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    preds_scaled: List[float] = []
    labels_scaled: List[float] = []
    sample_ids_all: List[str] = []

    with torch.no_grad():
        for audios, labels, sample_ids in loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            outputs = predictor(model, audios)
            if outputs.ndim > 1 and outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds_scaled.extend(outputs.cpu().numpy().reshape(-1).tolist())
            labels_scaled.extend(labels.cpu().numpy().reshape(-1).tolist())
            sample_ids_all.extend(list(sample_ids))

    preds_raw = scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).flatten()
    labels_raw = scaler.inverse_transform(np.asarray(labels_scaled).reshape(-1, 1)).flatten()
    preds_eval, labels_eval = finalize_predictions(preds_raw, labels_raw, sample_ids_all, eval_level=eval_level)
    metrics = compute_metrics(preds_eval, labels_eval)
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, metrics


def fit_simple_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    predictor: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    eval_level: str,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=max(2, patience // 3))

    best_score = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(max_epochs):
        model.train()
        for audios, labels, _sample_ids in train_loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            optimizer.zero_grad()
            if hasattr(model, "compute_training_loss"):
                loss, outputs = model.compute_training_loss(audios, labels)
            else:
                outputs = predictor(model, audios)
                if outputs.ndim > 1 and outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_loss, val_metrics = evaluate_regressor(
            model,
            val_loader,
            device,
            predictor,
            scaler,
            eval_level=eval_level,
        )
        scheduler.step(val_loss)
        score = val_metrics["MAE"]

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_stfn_regressor(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: StandardScaler,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    hcpc_weight: float = 0.1,
    eval_level: str = "recording",
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)

    best_score = float("inf")
    best_state = None
    patience_counter = 0

    for _ in range(max_epochs):
        model.train()
        for audios, labels, _sample_ids in train_loader:
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).squeeze(-1)
            optimizer.zero_grad()
            outputs, hcpc_loss = model(audios)
            if outputs.ndim > 1 and outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            mse = criterion(outputs, labels)
            loss = mse + hcpc_weight * hcpc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        _val_loss, val_metrics = evaluate_regressor(
            model,
            val_loader,
            device,
            predictor=lambda m, x: m(x)[0],
            scaler=scaler,
            eval_level=eval_level,
        )
        score = val_metrics["MAE"]

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def create_model(model_name: str) -> Tuple[nn.Module, Callable[[nn.Module, torch.Tensor], torch.Tensor], bool]:
    if model_name == "DepAudioNet":
        model = DepAudioNetBackbone()
        return model, lambda m, x: m(x), False
    if model_name == "DisNet":
        model = DisNetRegressor(
            DisNetConfig(
                sample_rate=16000,
                n_fft=1024,
                win_length=400,
                hop_length=160,
                num_filters=128,
                embed_dim=256,
                depth=4,
                num_heads=8,
                temporal_hidden_dim=128,
            )
        )
        return model, lambda m, x: m(x), False
    if model_name == "DMPF":
        model = DMPFRegressor(
            DMPFConfig(
                sample_rate=16000,
                common_time_steps=64,
                perspective_seq_dim=128,
                perspective_out_dim=256,
                common_dim=128,
                fusion_hidden_dim=64,
                fusion_out_dim=16,
            )
        )
        return model, lambda m, x: m(x)["y_hat"], False
    if model_name == "DALF":
        model = DALFNet(DALFConfig(
            sample_rate=16000,
            num_filters=64,
            gabor_kernel=401,
            pool_kernel=401,
            pool_stride=160,
            meb_blocks=4,
            mssa_hidden=128,
            head_channels=128,
            dropout=0.2,
        ))
        return model, lambda m, x: m(x), False
    if model_name == "STFN":
        model = STFNModel(input_dim=1, dropout=0.3, prediction_steps=1)
        return model, lambda m, x: m(x)[0], True
    raise KeyError(model_name)


@dataclass
class RunResult:
    metrics: Dict[str, float]
    info: Dict[str, int]


def _classical_cache_key(
    dataset_key: str,
    variant: str,
    seed: int,
    dm: RecordingSplitDataManager,
    sample_rate: int,
) -> Tuple[object, ...]:
    preview_ids = tuple(dm.sample_ids[:10])
    preview_lengths = tuple(int(len(seg)) for seg in dm.segments[:10])
    return (
        dataset_key,
        variant,
        seed,
        sample_rate,
        len(dm.segments),
        preview_ids,
        preview_lengths,
    )


def _classical_cache_path(cache_key: Tuple[object, ...]) -> Path:
    digest = hashlib.md5(repr(cache_key).encode("utf-8")).hexdigest()
    CLASSICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CLASSICAL_CACHE_DIR / f"{digest}.npy"


def _get_classical_features(
    dataset_key: str,
    variant: str,
    seed: int,
    dm: RecordingSplitDataManager,
    sample_rate: int,
) -> np.ndarray:
    cache_key = _classical_cache_key(dataset_key, variant, seed, dm, sample_rate)
    cached = CLASSICAL_FEATURE_CACHE.get(cache_key)
    if cached is None:
        cache_path = _classical_cache_path(cache_key)
        if cache_path.exists():
            cached = np.load(cache_path)
        else:
            cached = extract_handcrafted_features_batch(dm.segments, sample_rate=sample_rate)
            np.save(cache_path, cached)
        CLASSICAL_FEATURE_CACHE[cache_key] = cached
    return cached


def fit_classical_regressor(
    model_name: str,
    dataset_key: str,
    train_variant: str,
    eval_variant: str,
    seed: int,
    train_dm: RecordingSplitDataManager,
    eval_dm: RecordingSplitDataManager,
    sample_rate: int,
    eval_level: str,
) -> Dict[str, float]:
    features = _get_classical_features(
        dataset_key=dataset_key,
        variant=train_variant,
        seed=seed,
        dm=train_dm,
        sample_rate=sample_rate,
    )
    eval_features = features if train_variant == eval_variant and train_dm is eval_dm else _get_classical_features(
        dataset_key=dataset_key,
        variant=eval_variant,
        seed=seed,
        dm=eval_dm,
        sample_rate=sample_rate,
    )
    train_idx = np.asarray(train_dm.train_indices, dtype=np.int64)
    test_idx = np.asarray(eval_dm.test_indices, dtype=np.int64)

    x_train = features[train_idx]
    x_test = eval_features[test_idx]
    y_train = train_dm.labels_raw[train_idx]
    y_test = eval_dm.labels_raw[test_idx]
    sample_ids_test = [eval_dm.sample_ids[i] for i in test_idx.tolist()]

    feature_scaler = StandardScaler()
    x_train = feature_scaler.fit_transform(x_train)
    x_test = feature_scaler.transform(x_test)

    if model_name == "SVM":
        model = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    elif model_name == "RF":
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        )
    else:
        raise KeyError(model_name)

    model.fit(x_train, y_train)
    preds = np.asarray(model.predict(x_test), dtype=np.float32)
    preds_eval, labels_eval = finalize_predictions(preds, y_test, sample_ids_test, eval_level=eval_level)
    return compute_metrics(preds_eval, labels_eval)


def run_single_experiment(
    dataset_key: str,
    model_name: str,
    variant: str,
    seed: int,
    args: argparse.Namespace,
    shared_split_map: Optional[Dict[str, str]] = None,
) -> RunResult:
    set_seed(seed)
    defaults = MODEL_DEFAULTS[model_name]
    batch_size = args.batch_size or int(defaults["batch_size"])
    max_epochs = args.max_epochs or int(defaults["max_epochs"])
    learning_rate = args.learning_rate or float(defaults["learning_rate"])
    weight_decay = args.weight_decay or float(defaults["weight_decay"])
    patience = args.patience or int(defaults["patience"])
    eval_variant = normalize_variant_name(variant)
    train_variant = normalize_variant_name(args.train_variant or variant)

    train_dm = RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant=train_variant,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        max_groups_per_split=args.max_groups_per_split,
        split_map=shared_split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        segment_first=args.segment_first,
    )
    eval_dm = train_dm if train_variant == eval_variant else RecordingSplitDataManager(
        dataset_key=dataset_key,
        variant=eval_variant,
        segment_length=args.segment_length,
        sample_rate=args.sample_rate,
        batch_size=batch_size,
        num_workers=args.num_workers,
        random_state=seed,
        max_groups_per_split=args.max_groups_per_split,
        split_map=train_dm.split_map,
        cue_role=args.cue_role,
        speech_scope=args.speech_scope,
        segment_first=args.segment_first,
    )
    info = eval_dm.get_info()

    if model_name in {"SVM", "RF"}:
        metrics = fit_classical_regressor(
            model_name=model_name,
            dataset_key=dataset_key,
            train_variant=train_variant,
            eval_variant=eval_variant,
            seed=seed,
            train_dm=train_dm,
            eval_dm=eval_dm,
            sample_rate=args.sample_rate,
            eval_level=args.eval_level,
        )
        return RunResult(metrics=metrics, info=info)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, predictor, is_stfn = create_model(model_name)
    model = model.to(device)

    train_loader = train_dm.get_loader("train", shuffle=True)
    val_loader = train_dm.get_loader("val", shuffle=False)
    test_loader = eval_dm.get_loader("test", shuffle=False)

    if is_stfn:
        model = fit_stfn_regressor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=train_dm.scaler,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            hcpc_weight=args.hcpc_weight,
            eval_level=args.eval_level,
        )
    else:
        model = fit_simple_regressor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            scaler=train_dm.scaler,
            predictor=predictor,
            device=device,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            eval_level=args.eval_level,
        )

    _test_loss, metrics = evaluate_regressor(
        model,
        test_loader,
        device,
        predictor,
        train_dm.scaler,
        eval_level=args.eval_level,
    )
    return RunResult(metrics=metrics, info=info)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cue-aware baseline experiments for patient-centered auditing and role-effect analysis."
    )
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS.keys()), default=["edaic"])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["pre", "cue-only", "cue-excluded", "cue-removed", "random", "random-removed"],
        default=["pre", "cue-only", "cue-excluded", "random"],
    )
    parser.add_argument(
        "--train-variant",
        choices=["pre", "cue-only", "cue-excluded", "cue-removed", "random", "random-removed"],
        default=None,
        help="Optional shared training view. When set, the model is trained once on this view and evaluated on --variants.",
    )
    parser.add_argument(
        "--experiment",
        choices=["manual", "patient-baseline", "role-effect"],
        default="manual",
        help="Preconfigured experiment layout. 'patient-baseline' uses participant speech with patient cues; "
        "'role-effect' switches to dialogue speech and compares patient/doctor/all cue partitions.",
    )
    parser.add_argument("--cue-role", choices=["patient", "doctor", "all"], default=None)
    parser.add_argument("--speech-scope", choices=["participant", "interviewer", "dialogue"], default=None)
    parser.add_argument("--role-effect-roles", nargs="+", choices=["patient", "doctor", "all"], default=["patient", "doctor", "all"])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_DEFAULTS.keys()),
        default=["SVM", "RF", "DepAudioNet", "DisNet", "DMPF", "DALF", "STFN"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--segment-length", type=float, default=30.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--hcpc-weight", type=float, default=0.1)
    parser.add_argument("--max-groups-per-split", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-level", choices=["recording", "sample"], default="recording")
    parser.add_argument("--segment-first", action="store_true", help="Segment pre audio first, then reconstruct variants within each 30-s sample.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--info-only", action="store_true")
    return parser


def resolve_experiment_settings(args: argparse.Namespace) -> Tuple[List[str], List[str], str]:
    variants = list(args.variants)
    if args.experiment == "patient-baseline":
        return variants, ["patient"], "participant"
    if args.experiment == "role-effect":
        return variants, list(args.role_effect_roles), "dialogue"

    cue_role = args.cue_role or "patient"
    speech_scope = args.speech_scope or "participant"
    return variants, [cue_role], speech_scope


def main() -> None:
    args = build_parser().parse_args()
    variants, cue_roles, speech_scope = resolve_experiment_settings(args)
    summary_rows: List[Dict[str, object]] = []

    for dataset_key in args.datasets:
        dataset_name = DATASET_CONFIGS[dataset_key].display_name
        reference_splits: Dict[int, Dict[str, str]] = {}
        for seed in args.seeds:
            try:
                reference_dm = RecordingSplitDataManager(
                    dataset_key=dataset_key,
                    variant="pre",
                    segment_length=args.segment_length,
                    sample_rate=args.sample_rate,
                    batch_size=args.batch_size or 4,
                    num_workers=args.num_workers,
                    random_state=seed,
                    max_groups_per_split=args.max_groups_per_split,
                    cue_role=cue_roles[0],
                    speech_scope=speech_scope,
                    segment_first=args.segment_first,
                )
                reference_splits[seed] = reference_dm.split_map
            except Exception as exc:
                print(f"[{dataset_name} | seed={seed}] split setup failed: {exc}")

        for cue_role in cue_roles:
            args.cue_role = cue_role
            args.speech_scope = speech_scope
            for variant in variants:
                try:
                    dm = RecordingSplitDataManager(
                        dataset_key=dataset_key,
                        variant=variant,
                        segment_length=args.segment_length,
                        sample_rate=args.sample_rate,
                        batch_size=args.batch_size or 4,
                        num_workers=args.num_workers,
                        random_state=args.seeds[0],
                        max_groups_per_split=args.max_groups_per_split,
                        split_map=reference_splits.get(args.seeds[0]),
                        cue_role=cue_role,
                        speech_scope=speech_scope,
                        segment_first=args.segment_first,
                    )
                    info = dm.get_info()
                    print(
                        f"[{dataset_name} | role={cue_role} | scope={speech_scope} | {variant}] "
                        f"recordings={info['total_recordings']} "
                        f"(train/val/test={info['train_recordings']}/{info['val_recordings']}/{info['test_recordings']}) | "
                        f"segments={info['total_segments']}"
                    )
                except Exception as exc:
                    print(f"[{dataset_name} | role={cue_role} | {variant}] skipped: {exc}")
                    continue

                if args.info_only:
                    continue

                for model_name in args.models:
                    run_metrics: List[Dict[str, float]] = []
                    last_info = info
                    for seed in args.seeds:
                        if seed not in reference_splits:
                            continue
                        print(
                            f"Running {dataset_name} | {model_name} | role={cue_role} | "
                            f"scope={speech_scope} | {variant} | seed={seed} ...",
                            flush=True,
                        )
                        try:
                            result = run_single_experiment(
                                dataset_key,
                                model_name,
                                variant,
                                seed,
                                args,
                                shared_split_map=reference_splits[seed],
                            )
                        except Exception as exc:
                            print(f"  failed: {exc}")
                            continue
                        run_metrics.append(result.metrics)
                        last_info = result.info

                    if not run_metrics:
                        continue

                    summary_rows.append({
                        "Experiment": args.experiment,
                        "Dataset": dataset_name,
                        "CueRole": cue_role,
                        "SpeechScope": speech_scope,
                        "Model": model_name,
                        "TrainVariant": DISPLAY_VARIANTS.get(normalize_variant_name(args.train_variant or variant), args.train_variant or variant),
                        "Variant": DISPLAY_VARIANTS.get(normalize_variant_name(variant), variant),
                        "EvalLevel": args.eval_level,
                        "SegmentFirst": args.segment_first,
                        "MAE": format_mean_std([m["MAE"] for m in run_metrics]),
                        "RMSE": format_mean_std([m["RMSE"] for m in run_metrics]),
                        "R2": format_mean_std([m["R2"] for m in run_metrics]),
                        "Seeds": len(run_metrics),
                        "TestRecs": last_info["test_recordings"],
                        "TestSegs": last_info["test_segments"],
                    })

    if not summary_rows:
        print("No experiment results were produced.")
        return

    results_df = pd.DataFrame(summary_rows)
    print("\n" + results_df.to_markdown(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            results_df.to_csv(output_path, index=False)
        else:
            output_path.write_text(results_df.to_markdown(index=False), encoding="utf-8")
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
