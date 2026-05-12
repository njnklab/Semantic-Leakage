#!/usr/bin/env python3
"""
Unified Agent Pipeline Entry Point
统一的数据处理入口
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running as script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.core import UnifiedASRPipeline, UnifiedPipelineConfig
from agent.cue import CueDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_asr(
    data_dir: str,
    output_dir: str,
    dataset: str,
    language: str = "auto",
    device: str = "cuda",
    enable_diarization: bool = True,
    ollama_config: dict | None = None,
    overwrite: bool = False,
    workers: int = 1,
):
    """ASR处理阶段"""
    logger.info(f"=== ASR Processing: {dataset} ===")

    config = UnifiedPipelineConfig(
        language=language,
        enable_diarization=enable_diarization,
        device=device,
        ollama_config=ollama_config,
    )

    pipeline = UnifiedASRPipeline(config)
    try:
        pipeline.process_dataset(
            data_dir=data_dir,
            output_dir=output_dir,
            dataset=dataset,
            file_pattern="*.wav",
            skip_existing=not overwrite,
            max_workers=workers,
        )
    finally:
        pipeline.release_resources()

    logger.info(f"ASR complete. Output: {output_dir}")


def process_cues(
    output_dir: str,
    dataset: str,
    ollama_config: dict,
    use_llm_extraction: bool = True,
    overwrite: bool = False,
):
    """Cue检测阶段"""
    logger.info(f"=== Cue Detection: {dataset} ===")
    logger.info(f"LLM extraction: {'enabled' if use_llm_extraction else 'disabled'}")

    detector = CueDetector(
        ollama_config=ollama_config,
        use_llm_extraction=use_llm_extraction
    )

    dataset_output_dir = Path(output_dir) / dataset
    if not dataset_output_dir.exists():
        logger.error(f"Dataset output directory not found: {dataset_output_dir}")
        sys.exit(1)

    results = detector.process_directory(
        input_dir=str(dataset_output_dir),
        dataset=dataset,
        skip_existing=not overwrite,
    )

    logger.info(f"Cue detection complete. Processed {len(results)} samples")


def main():
    parser = argparse.ArgumentParser(description="Unified Agent Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["asr", "cue", "all"],
        default="all",
        help="Processing stage",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Input data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_SCRIPT_DIR / "outputs"),
        help="Base output directory (dataset subdirectory will be created automatically)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["E-DAIC", "ManDIC", "PDCH", "CMDC"],
        help="Dataset name",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=["auto", "en", "zh"],
        help="Language code",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    parser.add_argument(
        "--no-llm-cue",
        action="store_true",
        help="Disable LLM-based cue extraction (use rule-based only)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs instead of resuming from unfinished files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Requested ASR workers. Values greater than 1 will fall back to sequential mode for safety.",
    )

    args = parser.parse_args()

    default_data_dirs = {
        "E-DAIC": _PROJECT_ROOT / "data" / "E-DAIC",
        "ManDIC": _PROJECT_ROOT / "data" / "ManDIC",
        "PDCH": _PROJECT_ROOT / "data" / "PDCH",
        "CMDC": _PROJECT_ROOT / "data" / "CMDC_EULA",
    }

    from agent.config import settings

    api_keys = settings.load_api_keys()
    ollama_config = {
        "model": api_keys.get("OLLAMA_MODEL", "qwen2.5:72b"),
        "base_url": api_keys.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    }

    # 执行阶段
    if args.stage in ["asr", "all"]:
        if not args.data_dir:
            args.data_dir = str(default_data_dirs.get(args.dataset, ""))

        if not args.data_dir or not Path(args.data_dir).exists():
            logger.error(f"Data directory not found: {args.data_dir}")
            sys.exit(1)

        process_asr(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dataset=args.dataset,
            language=args.language,
            device=args.device,
            enable_diarization=not args.no_diarization,
            ollama_config=ollama_config,
            overwrite=args.overwrite,
            workers=max(1, args.workers),
        )

    if args.stage in ["cue", "all"]:
        process_cues(
            output_dir=args.output_dir,
            dataset=args.dataset,
            ollama_config=ollama_config,
            use_llm_extraction=not args.no_llm_cue,
            overwrite=args.overwrite,
        )

    logger.info("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()
