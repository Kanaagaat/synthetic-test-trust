"""RQ1 control: is D's residual explained by hard real gliomas that synthetic twins lack?

    .venv/bin/python scripts/rq1_visible_control.py      (after eval_many.py --syn s_test_gemini)

The annotator could not see a tumor on 6 of the 52 real test gliomas (generated/real_test_blind:
no blind label), but saw one on 51 of 52 OpenAI twins and on all Gemini twins. If the S-test
overrates D only because the real test contains such hard cases (D may fail on them especially),
D's residual beyond scale compression should shrink when they are removed.

Set "visible": the real test with dataset boxes, but only the glioma images the annotator could see
and that have twins (45 pairs; eval_tests real_labels="dataset_subset"); the same 151 non-glioma
images; the OpenAI S-test and the Gemini S-test on the same images. Evaluated with `ultralytics val`
on these exact sets (cached; per-image stats of non-square images depend on the batch composition,
so they are not taken from the 51-pair sets). Compared with the 51-pair "dataset" scenario of
eval_many (its cache).

Residuals as in rq1_selfgen.py (reference line gap ~ real fitted on A_full / A_low / B_low models,
>= 25-epoch checkpoints + best). Uncertainty: one bootstrap over leakage groups of the 51-pair set;
the visible set uses the same drawn groups (its images within them), so visible - full is paired.

Output: viz/rq1_many/visible_control.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_many as M  # noqa: E402
import eval_tests as E  # noqa: E402
import rq1_selfgen as S  # noqa: E402

TAGS = ("real", "s_test", "s_test_gemini")
VISIBLE = dict(twin_labels="labels_blind", require_both=False, real_labels="dataset_subset")


def visible_records(models):
    real, syn, _ = E.build_sets(**VISIBLE)
    sets = {"real": real, "s_test": syn, "s_test_gemini": E.swap_twins(syn, "s_test_gemini")}
    yamls = {t: E.write_dataset(items, f"visible_{t}") for t, items in sets.items()}
    recs = {}
    for k, m in enumerate(models):
        for t, y in yamls.items():
            st = E.val_stats(m["weights"], y, M.OUT / "stats" / f"{m['name'].replace('@', '__')}__visible_{t}.json")
            recs[m["name"], t] = [E.glioma_record(st["images"][x["img"].name]) for x in sets[t]]
        print(f"[{k + 1}/{len(models)}] {m['name']}", flush=True)
    return sets, recs


def main(n_boot=2000, min_epoch=25):
    models = M.find_models([M.ROOT / r for r in S.RUNS])
    models = [m for m in models if m["variant"] != "bonus" and (m["epochs"] is None or m["epochs"] >= min_epoch)]
    names = [m["name"] for m in models]
    var = np.array([m["variant"] for m in models])
    best = np.array([m["ckpt"] == "best" for m in models])
    ref_ix = np.isin(var, S.REF)

    fsets, frecs, _ = M.evaluate(models, ("s_test_gemini",))
    full = {"items": fsets["dataset"]["real"], "recs": {(n, t): frecs[n, "dataset", t] for n in names for t in TAGS}}
    vsets, vrecs = visible_records(models)
    vis = {"items": vsets["real"], "recs": vrecs}
    assert [x["id"] for x in vsets["s_test_gemini"]] == [x["id"] for x in vis["items"]]
    for s in (full, vis):
        s["n_glioma"] = sum(r["n_gt"] > 0 for r in s["recs"][names[0], "real"])
        s["groups"] = defaultdict(list)
        for i, x in enumerate(s["items"]):
            s["groups"][x["group"]].append(i)
    keys = sorted(full["groups"])

    pt = {sn: {t: {n: E.metrics(s["recs"][n, t]) for n in names} for t in TAGS} for sn, s in (("full", full), ("visible", vis))}
    bt = {sn: {t: {k: np.zeros((n_boot, len(names))) for k in M.METRICS} for t in TAGS} for sn in ("full", "visible")}
    rng = np.random.default_rng(0)
    for b in range(n_boot):
        draw = [keys[g] for g in rng.choice(len(keys), len(keys))]
        for sn, s in (("full", full), ("visible", vis)):
            idx = [i for g in draw for i in s["groups"].get(g, [])]
            for j, n in enumerate(names):
                for t in TAGS:
                    mt = E.metrics([s["recs"][n, t][i] for i in idx])
                    for k in M.METRICS:
                        bt[sn][t][k][b, j] = mt[k]

    out = {"n_models": len(names), "bootstrap": n_boot,
           "sets": {"full": {"images": len(full["items"]), "glioma_pairs": full["n_glioma"]},
                    "visible": {"images": len(vis["items"]), "glioma_pairs": vis["n_glioma"]}},
           "results": {}}
    for syn in ("s_test", "s_test_gemini"):
        res = {}
        for k in M.METRICS:
            d = {}
            for v in S.SAME + S.REF:
                for sel, tag in ((var == v, "all_ckpts"), ((var == v) & best, "best")):
                    row = {}
                    boots = {}
                    for sn in ("full", "visible"):
                        r = np.array([pt[sn]["real"][n][k] for n in names])
                        s = np.array([pt[sn][syn][n][k] for n in names])
                        est, _ = S.fit_resid(r, s, ref_ix, sel)
                        boots[sn] = np.array([S.fit_resid(bt[sn]["real"][k][b], bt[sn][syn][k][b], ref_ix, sel)[0]
                                              for b in range(n_boot)])
                        row[sn] = {"real_mean": float(r[sel].mean()), "s_test_mean": float(s[sel].mean()),
                                   "gap_mean": float((s - r)[sel].mean()), "residual": est,
                                   "residual_ci95": M.ci(boots[sn])}
                    row["visible - full"] = {"residual_diff": row["visible"]["residual"] - row["full"]["residual"],
                                             "residual_diff_ci95": M.ci(boots["visible"] - boots["full"])}
                    d[f"{v}/{tag}"] = row
            res[k] = d
        out["results"][syn] = res
    p = M.OUT / "visible_control.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["sets"]))
    for syn, res in out["results"].items():
        print(f"\n[{syn}] residual: full ({full['n_glioma']} pairs) -> visible ({vis['n_glioma']} pairs), diff [95% CI]")
        for k, d in res.items():
            for key in ("D/best", "C_low/best", "C_full/best", "A_low/best", "B_low/best", "A_full/best"):
                f, v, df = d[key]["full"], d[key]["visible"], d[key]["visible - full"]
                print(f"  {k:12s} {key:12s} real {f['real_mean']:.3f}->{v['real_mean']:.3f}  "
                      f"residual {f['residual']:+.3f} -> {v['residual']:+.3f} {np.round(v['residual_ci95'], 3).tolist()}  "
                      f"diff {df['residual_diff']:+.3f} {np.round(df['residual_diff_ci95'], 3).tolist()}")
    print("->", p)


if __name__ == "__main__":
    main()
