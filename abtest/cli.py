"""Command line interface.

    abtest power --baseline 0.05 --mde 0.005
    abtest ztest --control 480 1000 --treatment 530 1000
    abtest bayes --control 480 1000 --treatment 530 1000
    abtest ttest --csv data.csv --group-col variant --metric-col revenue
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import (
    bayes_means,
    bayes_proportions,
    sample_size_means,
    sample_size_proportions,
    ttest_means,
    ztest_proportions,
)


def _add_counts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--control", nargs=2, type=int, metavar=("CONVERSIONS", "N"), required=True)
    p.add_argument("--treatment", nargs=2, type=int, metavar=("CONVERSIONS", "N"), required=True)


def _add_csv(p: argparse.ArgumentParser) -> None:
    p.add_argument("--csv", required=True)
    p.add_argument("--group-col", default="group")
    p.add_argument("--metric-col", default="metric")
    p.add_argument("--control-label", default="control")
    p.add_argument("--treatment-label", default="treatment")


def _load_arms(args: argparse.Namespace):
    df = pd.read_csv(args.csv)
    a = df.loc[df[args.group_col] == args.control_label, args.metric_col].to_numpy()
    b = df.loc[df[args.group_col] == args.treatment_label, args.metric_col].to_numpy()
    if a.size == 0 or b.size == 0:
        sys.exit("one of the arms is empty; check --group-col and labels")
    return a, b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="abtest", description="Design and analyse A/B tests")
    sub = p.add_subparsers(dest="cmd", required=True)

    pw = sub.add_parser("power", help="per-arm sample size for a conversion-rate test")
    pw.add_argument("--baseline", type=float, required=True)
    pw.add_argument("--mde", type=float, required=True, help="absolute lift, or relative with --relative")
    pw.add_argument("--relative", action="store_true")
    pw.add_argument("--alpha", type=float, default=0.05)
    pw.add_argument("--power", type=float, default=0.8)

    pm = sub.add_parser("power-means", help="per-arm sample size for a difference in means")
    pm.add_argument("--std", type=float, required=True)
    pm.add_argument("--mde", type=float, required=True)
    pm.add_argument("--alpha", type=float, default=0.05)
    pm.add_argument("--power", type=float, default=0.8)

    z = sub.add_parser("ztest", help="two-proportion z-test")
    _add_counts(z)
    z.add_argument("--alpha", type=float, default=0.05)

    b = sub.add_parser("bayes", help="Beta-Binomial comparison of conversion rates")
    _add_counts(b)

    t = sub.add_parser("ttest", help="Welch t-test on a CSV of per-user metrics")
    _add_csv(t)
    t.add_argument("--alpha", type=float, default=0.05)

    bm = sub.add_parser("bayes-means", help="Bayesian comparison of means from a CSV")
    _add_csv(bm)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "power":
        out = {
            "n_per_arm": sample_size_proportions(
                args.baseline, args.mde, args.alpha, args.power, relative=args.relative
            )
        }
    elif args.cmd == "power-means":
        out = {"n_per_arm": sample_size_means(args.std, args.mde, args.alpha, args.power)}
    elif args.cmd == "ztest":
        out = ztest_proportions(*args.control, *args.treatment, alpha=args.alpha).to_dict()
    elif args.cmd == "bayes":
        out = bayes_proportions(*args.control, *args.treatment).to_dict()
    elif args.cmd == "ttest":
        out = ttest_means(*_load_arms(args), alpha=args.alpha).to_dict()
    elif args.cmd == "bayes-means":
        out = bayes_means(*_load_arms(args)).to_dict()
    else:  # pragma: no cover
        raise SystemExit(2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
