"""Seven independent, non-optimized fixed/default references."""

from __future__ import annotations

from common import Context, run_search

from interactive_forecasting.domain.types import ModelFamily


def run(context: Context, family: ModelFamily) -> dict:
    return run_search(context, "fixed_default", fixed=family)


if __name__ == "__main__":
    import sys

    from run_all import main

    main(["--experiment", "fixed_default", *sys.argv[1:]])
