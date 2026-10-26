"""CLI entry point for the recommendation prototype."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import generate_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI Recommendation Prototype")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate catalog, users, interactions")
    gen.add_argument("--seed", type=int, default=42, help="Random seed")
    gen.add_argument("--num-courses", type=int, default=500, help="Number of courses")
    gen.add_argument("--num-users", type=int, default=200, help="Number of users")

    args = parser.parse_args(argv)

    if args.command == "generate":
        generate_all(seed=args.seed, num_courses=args.num_courses, num_users=args.num_users)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
