"""One full-budget production TPE run per pre-fixed architecture."""

from __future__ import annotations

from common import Context, run_search

from interactive_forecasting.domain.types import ModelFamily


def run(context: Context, family: ModelFamily) -> dict:
    if tuple(item.family for item in context.space.families) != (family,):
        raise ValueError("family must be fixed before architecture-specific optimization")
    return run_search(context, "architecture_specific")


if __name__ == "__main__":
    import sys

    from run_all import main

    main(["--experiment", "architecture_specific", *sys.argv[1:]])
