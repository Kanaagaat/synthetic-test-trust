"""Real-vs-real baseline for rank agreement: is the synthetic test's ranking worse than the ranking
of another real test of the same size?

    .venv/bin/python scripts/tau_split_half.py [--splits 1000]   # cached per-image stats of eval_many.py only

The paper reports Kendall tau_b between the real test and the S-test on the full, image-paired sets
(0.23 for AP50 over 72 models). A low tau can come from the synthetic test, or simply from a small test:
two independent real tests of ~25 gliomas may also reorder close models. This script separates the two.

Design (same stratified group split as the split-half design of regret.py): the 138 leakage groups are
halved at random, groups with a glioma and groups without one separately. On every split we score each
model on half A and half B and compute
    real_vs_real   tau(real A, real B)        two independent real tests of the same size   <- baseline
    syn_vs_real    tau(S-test A, real B)      synthetic half vs an independent real half
    gem_vs_real    tau(Gemini S-test A, real B)
    syn_vs_real_paired  tau(S-test A, real A) same images, as in the paper but at half size (reference)
    gem_vs_real_paired  tau(Gemini A, real A)
and the per-split difference syn_vs_real - real_vs_real (negative = the synthetic half predicts the real
ranking worse than a real half of the same size does). Ranges are 2.5-97.5% over splits: the spread of
the statistic over random halvings of this one test, not a confidence interval for the population.

Pools as in regret.py: all72 (the paper's RQ1 set), best18, and both without D.
Output: viz/tau_split_half.json, printed summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_many as M  # noqa: E402
import eval_tests as E  # noqa: E402
import regret as R  # noqa: E402

METRICS = ("AP50", "recall@0.25", "AP50-95")
TESTS = ("real", "s_test", "s_test_gemini")
STATS = ("real_vs_real", "syn_vs_real", "gem_vs_real", "syn_vs_real_paired", "gem_vs_real_paired")


def scores(names, data, test, idx):
    """models x metrics on images idx."""
    out = np.empty((len(names), len(METRICS)))
    for i, n in enumerate(names):
        m = E.metrics([data[n, test]["rec"][j] for j in idx])
        out[i] = [m[k] for k in METRICS]
    return out


def rng_range(v):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    models, real, data, groups = R.load()
    pools = {"all72": [m["name"] for m in models if m["epochs"] is None or m["epochs"] >= 25],
             "best18": [m["name"] for m in models if m["ckpt"] == "best"]}
    pools["all60_noD"] = [n for n in pools["all72"] if R.variant_of(n) != "D"]
    pools["best15_noD"] = [n for n in pools["best18"] if R.variant_of(n) != "D"]
    names = sorted(set().union(*pools.values()))
    pos = {n: i for i, n in enumerate(names)}
    pix = {p: np.array([pos[n] for n in ns]) for p, ns in pools.items()}

    keys = sorted(groups)
    n_gt = data[names[0], "real"]["n_gt"]
    strata = [[k for k in keys if n_gt[groups[k]].sum() > 0], [k for k in keys if n_gt[groups[k]].sum() == 0]]
    all_idx = np.arange(len(real))
    full = {t: scores(names, data, t, all_idx) for t in TESTS}

    acc = {p: {k: {s: [] for s in STATS} for k in METRICS} for p in pools}
    n_gl_a = []
    for it in range(args.splits):
        half_a, half_b = [], []
        for st in strata:
            perm = rng.permutation(len(st))
            cut = len(st) // 2 + (rng.random() < 0.5) * (len(st) % 2)
            half_a += [st[j] for j in perm[:cut]]
            half_b += [st[j] for j in perm[cut:]]
        a = np.array([i for g in half_a for i in groups[g]])
        b = np.array([i for g in half_b for i in groups[g]])
        n_gl_a.append(int(n_gt[a].sum()))
        sa = {t: scores(names, data, t, a) for t in TESTS}
        rb = scores(names, data, "real", b)
        for p, ix in pix.items():
            for j, k in enumerate(METRICS):
                ra_, rb_ = sa["real"][ix, j], rb[ix, j]
                s_, g_ = sa["s_test"][ix, j], sa["s_test_gemini"][ix, j]
                d = acc[p][k]
                d["real_vs_real"].append(M.kendall_tau_b(ra_, rb_))
                d["syn_vs_real"].append(M.kendall_tau_b(s_, rb_))
                d["gem_vs_real"].append(M.kendall_tau_b(g_, rb_))
                d["syn_vs_real_paired"].append(M.kendall_tau_b(s_, ra_))
                d["gem_vs_real_paired"].append(M.kendall_tau_b(g_, ra_))
        if (it + 1) % 100 == 0:
            print(f"  {it + 1}/{args.splits} splits", flush=True)

    out = {"set": "dataset scenario (real test with dataset boxes; S-tests with blind twin boxes), image-paired",
           "n_images": len(real), "n_groups": len(groups), "n_glioma_groups": len(strata[0]),
           "splits": args.splits, "glioma_boxes_in_half_A_range": [min(n_gl_a), max(n_gl_a)],
           "range_note": "2.5-97.5% over random stratified halvings, not a CI for the population",
           "pools": {}}
    for p, ix in pix.items():
        po = {"n_models": len(ix)}
        for j, k in enumerate(METRICS):
            d = {s: np.asarray(v, float) for s, v in acc[p][k].items()}
            e = {"full_test_tau": {"syn_vs_real": M.kendall_tau_b(full["s_test"][ix, j], full["real"][ix, j]),
                                   "gem_vs_real": M.kendall_tau_b(full["s_test_gemini"][ix, j], full["real"][ix, j])}}
            for s, v in d.items():
                e[s] = {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "range95": rng_range(v),
                        "n_nan": int(np.isnan(v).sum())}
            for s in ("syn_vs_real", "gem_vs_real"):
                diff = d[s] - d["real_vs_real"]
                ok = ~np.isnan(diff)
                e[f"{s}_minus_real_vs_real"] = {"mean": float(np.nanmean(diff)), "range95": rng_range(diff),
                                                "share_of_splits_below_0": float(np.mean(diff[ok] < 0))}
            po[k] = e
        out["pools"][p] = po
    path = E.ROOT / "viz" / "tau_split_half.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)

    print(f"glioma boxes in half A: {out['glioma_boxes_in_half_A_range']}")
    for p, po in out["pools"].items():
        print(f"\n== {p} ({po['n_models']} models)")
        for k in METRICS:
            e = po[k]
            f = lambda s: f"{e[s]['mean']:+.2f} {np.round(e[s]['range95'], 2).tolist()}"  # noqa: E731
            print(f"  {k:12s} full-test tau syn {e['full_test_tau']['syn_vs_real']:+.2f}, gem {e['full_test_tau']['gem_vs_real']:+.2f}")
            print(f"  {'':12s} halves: real A~real B {f('real_vs_real')} | S A~real B {f('syn_vs_real')}"
                  f" | Gem A~real B {f('gem_vs_real')} | S A~real A {f('syn_vs_real_paired')}")
            for s in ("syn_vs_real", "gem_vs_real"):
                g = e[f"{s}_minus_real_vs_real"]
                print(f"  {'':12s} {s} - real_vs_real: {g['mean']:+.2f} {np.round(g['range95'], 2).tolist()},"
                      f" below 0 in {g['share_of_splits_below_0']:.0%} of splits")


if __name__ == "__main__":
    main()
