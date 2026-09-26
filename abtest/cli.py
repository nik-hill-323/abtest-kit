"""Command line interface.

    abtest power --baseline 0.05 --mde 0.005
    abtest mde-means --std 12 --n-per-arm 8000
    abtest ztest --control 480 1000 --treatment 530 1000
    abtest bayes --control 480 1000 --treatment 530 1000
    abtest ttest --csv data.csv --group-col variant --metric-col revenue
    abtest ratio --csv data.csv --numerator-col revenue --denominator-col sessions
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import (
    bayes_means,
    bayes_proportions,
    mde_means,
    ratio_metric,
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


def _load_ratio_arms(args: argparse.Namespace):
    df = pd.read_csv(args.csv)
    arms = []
    for label in (args.control_label, args.treatment_label):
        rows = df.loc[df[args.group_col] == label]
        if rows.empty:
            sys.exit(f"no rows with {args.group_col} == {label!r}; check --group-col and labels")
        arms.append(rows[args.numerator_col].to_numpy())
        arms.append(rows[args.denominator_col].to_numpy())
    return arms


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

    mm = sub.add_parser("mde-means", help="smallest detectable difference in means for a fixed n")
    mm.add_argument("--std", type=float, required=True)
    mm.add_argument("--n-per-arm", type=int, required=True)
    mm.add_argument("--alpha", type=float, default=0.05)
    mm.add_argument("--power", type=float, default=0.8)

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

    rt = sub.add_parser("ratio", help="delta-method test for a per-user ratio metric")
    rt.add_argument("--csv", required=True, help="one row per user")
    rt.add_argument("--group-col", default="group")
    rt.add_argument("--numerator-col", required=True, help="e.g. the user's total revenue")
    rt.add_argument("--denominator-col", required=True, help="e.g. the user's session count")
    rt.add_argument("--control-label", default="control")
    rt.add_argument("--treatment-label", default="treatment")
    rt.add_argument("--alpha", type=float, default=0.05)
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
    elif args.cmd == "mde-means":
        out = {"mde": mde_means(args.std, args.n_per_arm, args.alpha, args.power)}
    elif args.cmd == "ztest":
        out = ztest_proportions(*args.control, *args.treatment, alpha=args.alpha).to_dict()
    elif args.cmd == "bayes":
        out = bayes_proportions(*args.control, *args.treatment).to_dict()
    elif args.cmd == "ttest":
        out = ttest_means(*_load_arms(args), alpha=args.alpha).to_dict()
    elif args.cmd == "bayes-means":
        out = bayes_means(*_load_arms(args)).to_dict()
    elif args.cmd == "ratio":
        out = ratio_metric(*_load_ratio_arms(args), alpha=args.alpha).to_dict()
    else:  # pragma: no cover
        raise SystemExit(2)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
