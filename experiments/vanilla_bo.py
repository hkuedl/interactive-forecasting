"""Joint family/feature/HP TPE search through the production SearchEngine."""

from __future__ import annotations

from common import Context, run_search


def run(context: Context) -> dict:
    return run_search(context, "vanilla_bo")


if __name__ == "__main__":
    import sys

    from run_all import main

    main(["--experiment", "vanilla_bo", *sys.argv[1:]])
