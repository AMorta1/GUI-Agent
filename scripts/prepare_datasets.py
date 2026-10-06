"""Prepare supported GUI datasets in the Week 3 canonical JSONL format."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui_agent.datasets import (
    ConversionManifest,
    convert_mind2web,
    convert_screenagent,
    convert_webarena,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    screenagent = subparsers.add_parser("screenagent")
    _add_common_arguments(screenagent)
    screenagent.add_argument("--source", type=Path, required=True)
    screenagent.add_argument("--split", choices=("train", "test"), required=True)

    webarena = subparsers.add_parser("webarena")
    _add_common_arguments(webarena)
    webarena.add_argument("--source", type=Path, required=True)

    mind2web = subparsers.add_parser("mind2web")
    _add_common_arguments(mind2web)
    mind2web.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="External Hugging Face dataset cache root.",
    )
    mind2web.add_argument(
        "--dataset-id",
        default="osunlp/Multimodal-Mind2Web",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "screenagent":
        manifest = convert_screenagent(
            args.source,
            args.output,
            split=args.split,
            revision=args.revision,
            source_acquired_at=args.source_acquired_at,
            limit=args.limit,
        )
    elif args.dataset == "webarena":
        manifest = convert_webarena(
            args.source,
            args.output,
            revision=args.revision,
            source_acquired_at=args.source_acquired_at,
            limit=args.limit,
        )
    else:
        manifest = _convert_mind2web(args)
    print(json.dumps(manifest.model_dump(), ensure_ascii=False, indent=2))
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--source-acquired-at",
        required=True,
        help="Stable acquisition date or timestamp recorded in the manifest.",
    )
    parser.add_argument("--limit", type=int)


def _convert_mind2web(args: argparse.Namespace) -> ConversionManifest:
    cache_dir = args.cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    cache_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    rows = load_dataset(
        args.dataset_id,
        split="train",
        revision=args.revision,
        streaming=True,
        cache_dir=str(cache_dir / "datasets"),
    )
    return convert_mind2web(
        rows,
        args.output,
        revision=args.revision,
        source_acquired_at=args.source_acquired_at,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
