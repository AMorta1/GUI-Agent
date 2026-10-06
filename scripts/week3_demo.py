"""Generate a validated Week 3 GUI task plan from an optional static image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui_agent.agent import GuiPlanningAgent
from gui_agent.models import OpenAICompatibleVisionClient, TransformersVisionClient


DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("local", "api"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--model")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--device-map", choices=("cuda", "auto", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = _local_client(args) if args.provider == "local" else _api_client(args)
    plan = GuiPlanningAgent(client).plan(
        args.task,
        image_path=args.image,
        context=args.context,
    )
    print(plan.model_dump_json(indent=2))
    return 0


def _local_client(args: argparse.Namespace) -> TransformersVisionClient:
    if args.cache_dir is None:
        raise SystemExit("--cache-dir is required for the local provider")
    cache_dir = args.cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache_dir.parent)
    os.environ["HF_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_XET_CACHE"] = str(cache_dir.parent / "xet")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return TransformersVisionClient(
        args.model or DEFAULT_LOCAL_MODEL,
        cache_dir,
        device_map=args.device_map,
    )


def _api_client(args: argparse.Namespace) -> OpenAICompatibleVisionClient:
    if not args.model:
        raise SystemExit("--model is required for the API provider")
    if not args.base_url:
        raise SystemExit("--base-url is required for the API provider")
    return OpenAICompatibleVisionClient(
        args.model,
        args.base_url,
        api_key_env=args.api_key_env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
