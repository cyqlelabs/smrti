from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from smrti.decisions.student import registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.decisions.distill", description=__doc__)
    parser.add_argument("stage", choices=["corpus", "label", "train", "export", "evaluate", "all"])
    parser.add_argument("--tasks", default="", help="comma-separated heads; default every registry head")
    parser.add_argument("--scale", type=float, default=1.0, help="corpus size multiplier")
    parser.add_argument("--limit", type=int, default=None, help="label at most this many states per task")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--teacher", default=None, help="the /v1/systemone server that labels (SMRTI_DECISIONS_URL)")
    parser.add_argument("--student", default=None, help="an exported student directory to evaluate")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or list(registry.tasks())
    stages = ["corpus", "label", "train", "export", "evaluate"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "corpus":
            from .corpus import build

            build(tasks, args.scale)
        elif stage == "label":
            from .label import TEACHER_URL, label

            label(tasks, args.teacher or TEACHER_URL, args.limit)
        elif stage == "train":
            from .train import train

            train(tasks, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
        elif stage == "export":
            from .export import export

            export()
        elif stage == "evaluate":
            from .evaluate import OUT, evaluate

            evaluate(Path(args.student) if args.student else OUT / "student", tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
