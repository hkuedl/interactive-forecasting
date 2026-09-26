"""Quantile-capable production search, pinball fit and CRPS selection."""

from __future__ import annotations

from common import Context, run_search


def run(context: Context) -> dict:
    if context.metric.objective != "crps":
        raise ValueError("probabilistic experiment requires frozen CRPS protocol")
    return run_search(context, "probabilistic")


if __name__ == "__main__":
    import sys

    from run_all import main

    main(["--experiment", "probabilistic", *sys.argv[1:]])
