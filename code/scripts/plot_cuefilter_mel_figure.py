"""Train CueFilter on ManDIC data, generate 3-panel mel spectrogram comparison figure."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from CueFilter.Baseline.models.depaudionet import DepAudioNetBackbone
from CueFilter.models.cuefilter import CueFilter
from CueFilter.losses import cuefilter_pretrain_loss
from CueFilter.Baseline.audio_views import (
    iter_output_samples,
    audio_path_for_sample,
    load_json,
    load_metadata_mapping,
    speech_intervals_from_transcript,
    cue_intervals_from_outputs,
    extract_intervals,
    map_intervals_to_concatenated_timeline,
    intersect_intervals,
    label_key_for_sample,
)
from CueFilter.Baseline.experiment_utils import set_seed
from sklearn.metrics import f1_score as sk_f1

SR = 16000
N_FFT = 320
WIN_LENGTH = 320
HOP_LENGTH = 160
N_MELS = 64
SEGMENT_SEC = 30.0
SEGMENT_SAMPLES = int(SEGMENT_SEC * SR)

INPUT_DIM = 32
N_BLOCKS = 2
KERNEL_SIZE = 5
GROUPS = 8
ALPHA = 0.8
GAMMA = 0.2
SEED = 42

OUTDIR = PROJECT_ROOT / "agent" / "outputs" / "ManDIC"
ANNDIR = PROJECT_ROOT / "CueAnnotatorSys" / "annotations"


def compute_log_mel(audio):
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS)
    spec = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window="hann")) ** 2
    mel = np.dot(mel_basis, spec)
    return np.log(np.maximum(mel, 1e-6))


def load_recordings(dataset_key="mandic", n_recordings=None, use_annotations_only=False):
    """Load audio recordings and process into 30s segments with cue labels.

    Args:
        use_annotations_only: if True, only use recordings with manual annotations.
                              if False, use all recordings with LLM-detected cues.
    """
    samples = iter_output_samples(dataset_key)
    metadata = load_metadata_mapping(dataset_key)

    all_annotated = set()
    if use_annotations_only:
        for s in samples:
            sid = str(s["sample_id"])
            ann_path = ANNDIR / f"annotation_ManDIC_{sid}.json"
            if ann_path.exists():
                all_annotated.add(sid)

    available = []
    for s in samples:
        sid = str(s["sample_id"])
        if use_annotations_only and sid not in all_annotated:
            continue
        ap = audio_path_for_sample(dataset_key, sid)
        if ap.exists():
            lk = label_key_for_sample(dataset_key, sid)
            if lk in metadata:
                available.append((sid, s, float(metadata[lk]["label"])))

    if n_recordings and len(available) > n_recordings:
        rng = random.Random(SEED)
        available = rng.sample(available, n_recordings)

    print(f"Loading {len(available)} recordings...")
    items = []
    t0 = time.time()
    for i, (sid, s, label) in enumerate(available):
        transcript_path = s["transcript_path"]
        cues_path = s["cues_path"]
        transcript = load_json(transcript_path)
        transcript["sample_id"] = sid
        cues = load_json(cues_path)

        ap = audio_path_for_sample(dataset_key, sid)
        try:
            audio, _sr = librosa.load(ap, sr=SR, mono=True)
        except Exception:
            continue

        # Extract participant speech and map cues to preprocessed timeline
        retained = speech_intervals_from_transcript(transcript, speech_scope="participant")
        cue_iv = cue_intervals_from_outputs(dataset_key, sid, transcript, cues, cue_role="patient")
        cue_iv = intersect_intervals(retained, cue_iv)
        pre_audio = extract_intervals(audio, SR, retained)
        mapped_cues = map_intervals_to_concatenated_timeline(retained, cue_iv)

        if len(pre_audio) == 0:
            continue

        # Segment into 30s chunks
        for seg_idx in range(0, len(pre_audio), SEGMENT_SAMPLES):
            chunk = pre_audio[seg_idx:seg_idx + SEGMENT_SAMPLES]
            if len(chunk) < SEGMENT_SAMPLES:
                chunk = np.pad(chunk, (0, SEGMENT_SAMPLES - len(chunk)))

            seg_start = seg_idx / SR
            seg_end = seg_start + SEGMENT_SEC
            cue_spans = []
            for cs, ce in mapped_cues:
                rs = max(cs - seg_start, 0.0)
                re_ = min(ce - seg_start, SEGMENT_SEC)
                if re_ > rs:
                    cue_spans.append((rs, re_))

            items.append({
                "audio": chunk.astype(np.float32),
                "label_raw": label,
                "sample_id": sid,
                "cue_spans_sec": cue_spans,
                "duration_sec": SEGMENT_SEC,
            })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(available)}, {len(items)} segments, {time.time()-t0:.0f}s")

    print(f"Total: {len(items)} segments from {len(available)} recordings, {time.time()-t0:.0f}s")
    return items


def build_cue_labels(cue_spans_sec, num_frames, duration_sec, device="cpu"):
    labels = torch.zeros(num_frames, device=device)
    if not cue_spans_sec:
        return labels
    frame_dur = duration_sec / num_frames
    for s, e in cue_spans_sec:
        fs = max(0, int(s / frame_dur))
        fe = min(num_frames, int(np.ceil(e / frame_dur)))
        labels[fs:fe] = 1.0
    return labels


def main():
    set_seed(SEED)

    # --- Load ALL ManDIC recordings for training ---
    items = load_recordings("mandic", use_annotations_only=False)

    # Split by sample_id (80/10/10)
    sample_ids = sorted(set(it["sample_id"] for it in items))
    rng = random.Random(SEED)
    rng.shuffle(sample_ids)
    n_train = int(0.8 * len(sample_ids))
    n_val = int(0.1 * len(sample_ids))
    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train:n_train + n_val])
    test_ids = set(sample_ids[n_train + n_val:])

    train_items = [it for it in items if it["sample_id"] in train_ids]
    val_items = [it for it in items if it["sample_id"] in val_ids]
    test_items = [it for it in items if it["sample_id"] in test_ids]
    print(f"Split: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")

    # Normalize labels
    train_labels = np.array([it["label_raw"] for it in train_items], dtype=np.float32)
    label_mean = float(train_labels.mean())
    label_std = float(max(train_labels.std(ddof=0), 1e-6))
    for it in items:
        it["label_scaled"] = (it["label_raw"] - label_mean) / label_std

    # --- Models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    backbone = DepAudioNetBackbone().to(device)
    backbone.eval()

    cuefilter = CueFilter(
        input_dim=INPUT_DIM, n_blocks=N_BLOCKS, kernel_size=KERNEL_SIZE,
        groups=GROUPS, alpha=ALPHA, gamma=GAMMA,
    ).to(device)

    # Estimate feature stats
    print("Estimating feature stats...")
    all_feats = []
    for it in train_items[:300]:
        audio_t = torch.from_numpy(it["audio"]).unsqueeze(0).to(device)
        with torch.no_grad():
            sf = backbone.encode(audio_t)
        all_feats.append(sf.squeeze(0))
    stacked = torch.cat(all_feats, dim=0)
    cuefilter.set_feature_stats(
        stacked.mean(dim=0),
        stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
    )

    # --- Train ---
    optimizer = torch.optim.AdamW(cuefilter.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val_f1 = -1.0
    best_state = None
    patience = 15
    patience_counter = 0
    batch_size = 16

    print(f"Training CueFilter (max 50 epochs, patience={patience})...")
    for epoch in range(50):
        cuefilter.train()
        total_loss = 0.0
        n_batches = 0
        rng.shuffle(train_items)

        for i in range(0, len(train_items), batch_size):
            batch = train_items[i:i + batch_size]
            audios = torch.stack([torch.from_numpy(it["audio"]) for it in batch]).unsqueeze(1).to(device)

            with torch.no_grad():
                sf_batch = backbone.encode(audios)

            cue_labels_list = []
            cue_coverages = []
            for it in batch:
                cl = build_cue_labels(it["cue_spans_sec"], sf_batch.shape[1], it["duration_sec"], device)
                cue_labels_list.append(cl)
                cue_coverages.append(cl.mean())
            cue_labels = torch.stack(cue_labels_list)
            cue_coverage = torch.tensor(cue_coverages, device=device)

            outputs = cuefilter(sf_batch, mode="soft")
            loss, _ = cuefilter_pretrain_loss(
                outputs["p_cue"], cue_labels, cue_coverage,
                lambda_d=0.5, lambda_b=0.1,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cuefilter.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Validation
        cuefilter.eval()
        val_preds, val_labels_list = [], []
        for i in range(0, len(val_items), batch_size):
            batch = val_items[i:i + batch_size]
            audios = torch.stack([torch.from_numpy(it["audio"]) for it in batch]).unsqueeze(1).to(device)
            with torch.no_grad():
                sf_batch = backbone.encode(audios)
                out = cuefilter(sf_batch, mode="soft")
                p = out["p_cue"]
            for j, it in enumerate(batch):
                cl = build_cue_labels(it["cue_spans_sec"], sf_batch.shape[1], it["duration_sec"], device)
                val_preds.append(p[j].cpu())
                val_labels_list.append(cl.cpu())

        all_preds = torch.cat(val_preds).numpy()
        all_labels_t = torch.cat(val_labels_list).numpy()
        bin_preds = (all_preds >= 0.5).astype(int)
        val_f1 = sk_f1(all_labels_t, bin_preds, zero_division=0)
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: loss={avg_loss:.4f}, val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in cuefilter.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        cuefilter.load_state_dict(best_state)
    print(f"Best val F1: {best_val_f1:.4f}")

    # --- Select figure sample: must have annotation + be in test set ---
    annotated_samples = []
    for s in iter_output_samples("mandic"):
        sid = str(s["sample_id"])
        if ANNDIR / f"annotation_ManDIC_{sid}.json" in ANNDIR.iterdir():
            pass
    # Find test samples with annotations
    annotated_test = []
    for sid in sorted(test_ids):
        ann_path = ANNDIR / f"annotation_ManDIC_{sid}.json"
        if ann_path.exists():
            annotated_test.append(sid)

    if not annotated_test:
        # Fall back: any annotated sample
        for s in iter_output_samples("mandic"):
            sid = str(s["sample_id"])
            if (ANNDIR / f"annotation_ManDIC_{sid}.json").exists():
                annotated_test.append(sid)
        rng.shuffle(annotated_test)

    fig_sid = annotated_test[0]
    print(f"Figure sample: {fig_sid}")

    # Load this specific sample with annotations for the GT mask
    ann_items = load_recordings("mandic", use_annotations_only=True)
    fig_candidates = [it for it in ann_items if it["sample_id"] == fig_sid and len(it["cue_spans_sec"]) > 0]
    if not fig_candidates:
        fig_candidates = [it for it in ann_items if it["sample_id"] == fig_sid]
    fig_item = fig_candidates[0]
    print(f"  Segment cues: {len(fig_item['cue_spans_sec'])}")

    # --- Compute mel spectrogram ---
    audio = fig_item["audio"]
    log_mel = compute_log_mel(audio)
    n_mel_frames = log_mel.shape[1]
    print(f"  Mel: {log_mel.shape}")

    # --- CueFilter predicted gate ---
    audio_t = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        sf = backbone.encode(audio_t)
        out = cuefilter(sf, mode="soft", renorm=False)
        gate = out["gate"].squeeze(0).cpu().numpy()
        p_cue = out["p_cue"].squeeze(0).cpu().numpy()

    print(f"  CueFilter: p_cue=[{p_cue.min():.4f}, {p_cue.max():.4f}], gate=[{gate.min():.4f}, {gate.max():.4f}]")

    # --- GT mask ---
    dt = HOP_LENGTH / SR
    gt_mask = np.ones(n_mel_frames, dtype=np.float32)
    for cs, ce in fig_item["cue_spans_sec"]:
        fs = max(0, int(cs / dt))
        fe = min(n_mel_frames, int(np.ceil(ce / dt)))
        gt_mask[fs:fe] = GAMMA
    n_supp = (gt_mask < 1.0).sum()
    print(f"  GT mask: {n_supp}/{n_mel_frames} frames suppressed ({100*n_supp/n_mel_frames:.1f}%)")

    # --- Resample gate to mel resolution ---
    gate_t = torch.from_numpy(gate).float().unsqueeze(0).unsqueeze(0).to(device)
    gate_rs = F.interpolate(gate_t, size=n_mel_frames, mode="linear", align_corners=True).squeeze(0).squeeze(0).cpu().numpy()

    mel_cf = log_mel * gate_rs[np.newaxis, :]
    mel_mask = log_mel * gt_mask[np.newaxis, :]

    # --- Plot ---
    vmin, vmax = log_mel.min(), log_mel.max()
    fig, axes = plt.subplots(3, 1, figsize=(8, 4))
    for ax, data in zip(axes, [log_mel, mel_cf, mel_mask]):
        ax.imshow(data, aspect="auto", origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.03)

    out_path = PROJECT_ROOT / "cuefilter_mel_comparison.jpg"
    fig.savefig(out_path, dpi=300, format="jpg", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
