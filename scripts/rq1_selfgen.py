"""RQ1, variant D: does the S-test overrate models trained on the same generator beyond scale compression?

    .venv/bin/python scripts/rq1_selfgen.py          (after eval_many.py; uses its val cache)

The S-test overrates weak models more than strong ones (gap = S-test - real falls with the real
metric). So a large gap for D could be just "D is weak". We separate the two:
  reference models  never saw synthetic images: A_full, A_low, B_low (checkpoints >= 25 epochs + best)
  same-generator    C_low, C_full, D (trained with synthetic gliomas of the S-test's generator)
For each scenario and metric: fit gap = a + b * real on the reference models (OLS), then for each
same-generator variant take the mean residual of its models (all >= 25-epoch checkpoints + best, and
best.pt only). Residual > 0: the S-test rates the variant higher than a never-synthetic model of the
same real level would be rated. Uncertainty: one joint bootstrap over leakage groups (as eval_many),
the reference line is refitted in every resample. Also: the real-level range of the reference models,
to flag extrapolation.

Output: viz/rq1_many/selfgen.json

    .venv/bin/python scripts/rq1_selfgen.py --syn s_test_gemini   -> viz/rq1_many/selfgen_s_test_gemini.json
The same residuals on the main (OpenAI) S-test and on the S-test of a second generator (Gemini twins
of the same real gliomas), in one shared bootstrap, plus their paired difference (other - main).
If D's residual is large on its own generator's S-test but about 0 on the other one, the S-test
favours its own generator; if it is large on both, it favours synthetic images in general.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_many as M  # noqa: E402
import eval_tests as E  # noqa: E402

REF = ("A_full", "A_low", "B_low")
SAME = ("C_low", "C_full", "D")
RUNS = ("colab/runs_A", "colab/runs_BC", "colab/runs_FD")


def fit_resid(r, s, ref_ix, ix):
    g = s - r
    b, a = np.polyfit(r[ref_ix], g[ref_ix], 1)
    return float(np.mean(g[ix] - (a + b * r[ix]))), (float(a), float(b))


def main(n_boot=2000, min_epoch=25, syns=("s_test",)):
    models = M.find_models([M.ROOT / r for r in RUNS])
    models = [m for m in models if m["variant"] != "bonus" and (m["epochs"] is None or m["epochs"] >= min_epoch)]
    sets, recs, _ = M.evaluate(models, tuple(x for x in syns if x != "s_test"))
    names = [m["name"] for m in models]
    var = np.array([m["variant"] for m in models])
    best = np.array([m["ckpt"] == "best" for m in models])
    ref_ix = np.isin(var, REF)
    out = {"reference_variants": REF, "n_models": len(names), "n_reference_models": int(ref_ix.sum()),
           "bootstrap": n_boot, "scenarios": {}}
    rng = np.random.default_rng(0)
    for scen in M.SCENARIOS:
        real = sets[scen]["real"]
        groups = defaultdict(list)
        for i, x in enumerate(real):
            groups[x["group"]].append(i)
        keys = sorted(groups)
        tags = ("real", *syns)
        pt = {t: {n: E.metrics(recs[n, scen, t]) for n in names} for t in tags}
        bt = {t: {k: np.zeros((n_boot, len(names))) for k in M.METRICS} for t in tags}
        for b in range(n_boot):
            idx = [i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]]
            for j, n in enumerate(names):
                for t in tags:
                    mt = E.metrics([recs[n, scen, t][i] for i in idx])
                    for k in M.METRICS:
                        bt[t][k][b, j] = mt[k]
        by_syn, boots_all = {}, {}
        for syn in syns:
            res = {}
            for k in M.METRICS:
                r = np.array([pt["real"][n][k] for n in names])
                s = np.array([pt[syn][n][k] for n in names])
                d = {"reference_real_range": [float(r[ref_ix].min()), float(r[ref_ix].max())],
                     "reference_fit": dict(zip(("intercept", "slope"), fit_resid(r, s, ref_ix, ref_ix)[1])),
                     "corr_gap_vs_real_reference": float(np.corrcoef(r[ref_ix], (s - r)[ref_ix])[0, 1])}
                for v in SAME + REF:
                    for sel, tag in ((var == v, "all_ckpts"), ((var == v) & best, "best")):
                        if not sel.any():
                            continue
                        est, _ = fit_resid(r, s, ref_ix, sel)
                        boots = [fit_resid(bt["real"][k][b], bt[syn][k][b], ref_ix, sel)[0] for b in range(n_boot)]
                        boots_all[syn, k, f"{v}/{tag}"] = (est, np.array(boots))
                        d[f"{v}/{tag}"] = {"n": int(sel.sum()), "real_mean": float(r[sel].mean()),
                                           "s_test_mean": float(s[sel].mean()), "gap_mean": float((s - r)[sel].mean()),
                                           "residual": est, "residual_ci95": M.ci(boots)}
                res[k] = d
            by_syn[syn] = res
        if len(syns) == 1:
            out["scenarios"][scen] = by_syn[syns[0]]
            continue
        a, o = syns[0], syns[1]
        diff = {k: {key: {"residual_diff": boots_all[o, k, key][0] - boots_all[a, k, key][0],
                          "residual_diff_ci95": M.ci(boots_all[o, k, key][1] - boots_all[a, k, key][1])}
                    for (sy, kk, key) in boots_all if sy == a and kk == k}
                for k in M.METRICS}
        out["scenarios"][scen] = {**by_syn, f"{o} - {a}": diff}
    if len(syns) > 1:  # the default selfgen.json keeps its old keys
        out["syn_sets"] = list(syns)
    p = M.OUT / ("selfgen.json" if tuple(syns) == ("s_test",) else f"selfgen_{syns[-1]}.json")
    p.write_text(json.dumps(out, indent=1))
    for scen, sres in out["scenarios"].items():
        for syn, res in (sres.items() if len(syns) > 1 else [(syns[0], sres)]):
            print(f"\n[{scen} | {syn}]")
            if " - " in syn:
                for k, dd in res.items():
                    for key, v in dd.items():
                        print(f"  {k:12s} {key:16s} residual diff {v['residual_diff']:+.3f} "
                              f"{np.round(v['residual_diff_ci95'], 3).tolist()}")
                continue
            for k, d in res.items():
                print(f"  {k}: reference real range {np.round(d['reference_real_range'], 3).tolist()}, "
                      f"fit {d['reference_fit']}, corr {d['corr_gap_vs_real_reference']:.2f}")
                for key, v in d.items():
                    if "/" in key:
                        print(f"    {key:16s} n={v['n']:2d} real {v['real_mean']:.3f} s_test {v['s_test_mean']:.3f} "
                              f"gap {v['gap_mean']:+.3f} residual {v['residual']:+.3f} "
                              f"{np.round(v['residual_ci95'], 3).tolist()}")
    print("->", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syn", choices=list(M.OTHER_SYN), help="also the S-test of this second generator")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    main(n_boot=a.boot, syns=("s_test", a.syn) if a.syn else ("s_test",))
