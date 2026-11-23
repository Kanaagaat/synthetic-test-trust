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
"""
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


def main(n_boot=2000, min_epoch=25):
    models = M.find_models([M.ROOT / r for r in RUNS])
    models = [m for m in models if m["variant"] != "bonus" and (m["epochs"] is None or m["epochs"] >= min_epoch)]
    sets, recs, _ = M.evaluate(models)
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
        pt = {t: {n: E.metrics(recs[n, scen, t]) for n in names} for t in ("real", "s_test")}
        bt = {t: {k: np.zeros((n_boot, len(names))) for k in M.METRICS} for t in ("real", "s_test")}
        for b in range(n_boot):
            idx = [i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]]
            for j, n in enumerate(names):
                for t in ("real", "s_test"):
                    mt = E.metrics([recs[n, scen, t][i] for i in idx])
                    for k in M.METRICS:
                        bt[t][k][b, j] = mt[k]
        res = {}
        for k in M.METRICS:
            r = np.array([pt["real"][n][k] for n in names])
            s = np.array([pt["s_test"][n][k] for n in names])
            d = {"reference_real_range": [float(r[ref_ix].min()), float(r[ref_ix].max())],
                 "reference_fit": dict(zip(("intercept", "slope"), fit_resid(r, s, ref_ix, ref_ix)[1])),
                 "corr_gap_vs_real_reference": float(np.corrcoef(r[ref_ix], (s - r)[ref_ix])[0, 1])}
            for v in SAME + REF:
                for sel, tag in ((var == v, "all_ckpts"), ((var == v) & best, "best")):
                    if not sel.any():
                        continue
                    est, _ = fit_resid(r, s, ref_ix, sel)
                    boots = [fit_resid(bt["real"][k][b], bt["s_test"][k][b], ref_ix, sel)[0] for b in range(n_boot)]
                    d[f"{v}/{tag}"] = {"n": int(sel.sum()), "real_mean": float(r[sel].mean()),
                                       "s_test_mean": float(s[sel].mean()), "gap_mean": float((s - r)[sel].mean()),
                                       "residual": est, "residual_ci95": M.ci(boots)}
            res[k] = d
        out["scenarios"][scen] = res
    p = M.OUT / "selfgen.json"
    p.write_text(json.dumps(out, indent=1))
    for scen, res in out["scenarios"].items():
        print(f"\n[{scen}]")
        for k, d in res.items():
            print(f"  {k}: reference real range {np.round(d['reference_real_range'], 3).tolist()}, "
                  f"fit {d['reference_fit']}, corr {d['corr_gap_vs_real_reference']:.2f}")
            for key, v in d.items():
                if "/" in key:
                    print(f"    {key:16s} n={v['n']:2d} real {v['real_mean']:.3f} s_test {v['s_test_mean']:.3f} "
                          f"gap {v['gap_mean']:+.3f} residual {v['residual']:+.3f} {np.round(v['residual_ci95'], 3).tolist()}")
    print("->", p)


if __name__ == "__main__":
    main()
