"""Business view of RQ1: what does it cost to pick a model with a synthetic test instead of a real one?

    .venv/bin/python scripts/regret.py            # uses the cached per-image stats of eval_many.py only

Scenario: a hospital has several trained detectors and one test set to choose which one to deploy.
It picks the model with the highest glioma AP50 (or recall@0.25) on that test. We then look at how the
chosen model behaves on REAL images, in units a clinic understands:
    missed gliomas per 100 gliomas      = 100 * (1 - recall@0.25), IoU 0.5, real test (dataset boxes).
                                          Unit: annotated glioma boxes (58 boxes on 51 images), NOT patients
                                          (no patient IDs) — a decision-level proxy, not patient harm.
    false alarms per 100 tumor-free     = tumor-free images with any prediction (any class) at conf >= 0.25
    glioma false alarms per 100 non-glioma images = images without a glioma that get a glioma box >= 0.25
Regret = outcome of the model chosen by the synthetic test - outcome of the model chosen by the real test
(positive missed_per_100 = the synthetic choice misses more gliomas on real images).

Two designs:
    in_sample   choose and evaluate on the same real test (the real-test choice is an oracle, slightly
                optimistic). CI: bootstrap of whole leakage groups, selection redone inside every resample.
    split_half  groups split at random in two halves, stratified (groups with a glioma and groups without
                one are halved separately, so each half has 18-19 of the 37 glioma groups): choose on half A
                (real A vs its synthetic twins), evaluate on the real images of half B. Repeated; the
                real-test choice is no longer in-sample. The reported 2.5-97.5% range is the spread over
                splits, not a confidence interval for the population regret.
Ties (e.g. many models at the same S-test recall) count as a random pick: the outcome is the mean over
the tied models.

Pools: best18 = best.pt of the 18 runs (6 variants x 3 seeds), all72 = RQ1 model set (epochs >= 25 + best);
best15_noD / all60_noD = the same without D (a clinic would rarely hold a model with no real gliomas in train).
Sets: the "dataset" scenario of eval_many.py (real test with dataset boxes; S-test = the same images
with every real glioma swapped for its blind-boxed twin), so real and synthetic sets are paired image by image.

Output: viz/regret.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_many as M  # noqa: E402
import eval_tests as E  # noqa: E402

ROOT = E.ROOT
RUNS = ["colab/runs_A", "colab/runs_BC", "colab/runs_FD"]
TESTS = ("real", "s_test", "s_test_gemini")
SELECT_BY = ("AP50", "recall@0.25")
N_BOOT, N_SPLIT = 1000, 1000


def load():
    models = [m for m in M.find_models([ROOT / r for r in RUNS]) if m["variant"] != "bonus"]
    if not models:
        raise SystemExit("no run weights found: unzip runs_weights.zip (release) in the repository root, "
                         "then run eval_many.py (also with --syn s_test_gemini)")
    real, syn, _ = E.build_sets(**M.SCENARIOS["dataset"])
    sets = {"real": real, "s_test": syn, "s_test_gemini": E.swap_twins(syn, "s_test_gemini", "labels_blind")}
    for t in TESTS[1:]:
        assert [x["id"] for x in sets[t]] == [x["id"] for x in real], "sets must be paired"
    data = {}
    for m in models:
        for t, items in sets.items():
            cache = M.OUT / "stats" / f"{m['name'].replace('@', '__')}__dataset_{t}.json"
            if not cache.exists():
                raise SystemExit(f"missing cache {cache.name}: run eval_many.py first")
            st = json.loads(cache.read_text())["images"]
            imgs = [st[x["img"].name] for x in items]
            rec = [E.glioma_record(s) for s in imgs]
            data[m["name"], t] = {
                "rec": rec,
                "n_gt": np.array([r["n_gt"] for r in rec]),
                "hit": np.array([int(r["tp"][r["conf"] >= 0.25, 0].sum()) for r in rec]),
                "tumor_free": np.array([len(s["target_cls"]) == 0 for s in imgs]),
                "any_alarm": np.array([any(c >= 0.25 for c in s["conf"]) for s in imgs]),
                "glioma_alarm": np.array([bool((r["conf"] >= 0.25).any()) for r in rec]),
            }
    groups = defaultdict(list)
    for i, x in enumerate(real):
        groups[x["group"]].append(i)
    return models, real, data, groups


def score(d, idx, metric):
    """Selection score of one model on one test restricted to image indices idx."""
    if metric == "AP50":
        return E.metrics([d["rec"][i] for i in idx])["AP50"]
    n = d["n_gt"][idx].sum()
    return d["hit"][idx].sum() / n if n else np.nan


def outcomes(d, idx):
    """Clinic-facing numbers of one model on the REAL images idx."""
    n_gt = d["n_gt"][idx].sum()
    tf = d["tumor_free"][idx]
    non_glioma = d["n_gt"][idx] == 0
    return {"missed_per_100": 100 * (1 - d["hit"][idx].sum() / n_gt),
            "false_alarms_per_100_tumor_free": 100 * d["any_alarm"][idx][tf].mean(),
            "glioma_false_alarms_per_100_non_glioma": 100 * d["glioma_alarm"][idx][non_glioma].mean()}


OUTS = ("missed_per_100", "false_alarms_per_100_tumor_free", "glioma_false_alarms_per_100_non_glioma")


def choose(names, data, test, idx, metric):
    s = np.array([score(data[n, test], idx, metric) for n in names])
    best = np.nanmax(s)
    return [n for n, v in zip(names, s) if np.isclose(v, best, rtol=0, atol=1e-12)]


def mean_outcome(chosen, data, idx):
    o = [outcomes(data[n, "real"], idx) for n in chosen]
    return {k: float(np.mean([x[k] for x in o])) for k in OUTS}


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def variant_of(name):
    return name.split("@")[0].rsplit("_s", 1)[0]


def analyse(names, data, groups, n_items, rng):
    keys = sorted(groups)
    all_idx = np.arange(n_items)
    res = {}
    for metric in SELECT_BY:
        r = {"in_sample": {}, "split_half": {}}
        # ---------- in-sample point estimate + group bootstrap
        point_choice = {t: choose(names, data, t, all_idx, metric) for t in TESTS}
        point_out = {t: mean_outcome(c, data, all_idx) for t, c in point_choice.items()}
        boot = {t: {k: [] for k in OUTS} for t in TESTS}
        boot_d_share = {t: [] for t in TESTS}
        for _ in range(N_BOOT):
            idx = np.array([i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]])
            for t in TESTS:
                c = choose(names, data, t, idx, metric)
                o = mean_outcome(c, data, idx)
                for k in OUTS:
                    boot[t][k].append(o[k])
                boot_d_share[t].append(np.mean([variant_of(n) == "D" for n in c]))
        for t in TESTS:
            e = {"chosen": point_choice[t], "real_outcome": point_out[t],
                 "share_of_resamples_choosing_D": float(np.mean(boot_d_share[t]))}
            if t != "real":
                e["regret"] = {k: point_out[t][k] - point_out["real"][k] for k in OUTS}
                e["regret_ci95"] = {k: ci(np.subtract(boot[t][k], boot["real"][k])) for k in OUTS}
            r["in_sample"][t] = e
        # ---------- split-half: choose on half A, evaluate on real half B
        sh = {t: {k: [] for k in OUTS} for t in TESTS}
        sh_d = {t: [] for t in TESTS}
        n_gt = data[names[0], "real"]["n_gt"]
        strata = [[k for k in keys if n_gt[groups[k]].sum() > 0], [k for k in keys if n_gt[groups[k]].sum() == 0]]
        for _ in range(N_SPLIT):
            half_a, half_b = [], []
            for st in strata:
                perm = rng.permutation(len(st))
                cut = len(st) // 2 + (rng.random() < 0.5) * (len(st) % 2)  # odd stratum: the extra group goes either way
                half_a += [st[j] for j in perm[:cut]]
                half_b += [st[j] for j in perm[cut:]]
            a = np.array([i for g in half_a for i in groups[g]])
            b = np.array([i for g in half_b for i in groups[g]])
            for t in TESTS:
                c = choose(names, data, t, a, metric)
                o = mean_outcome(c, data, b)
                for k in OUTS:
                    sh[t][k].append(o[k])
                sh_d[t].append(np.mean([variant_of(n) == "D" for n in c]))
        for t in TESTS:
            e = {"real_outcome_mean": {k: float(np.mean(sh[t][k])) for k in OUTS},
                 "share_of_splits_choosing_D": float(np.mean(sh_d[t])), "n_splits": len(sh[t]["missed_per_100"])}
            if t != "real":
                diffs = {k: np.subtract(sh[t][k], sh["real"][k]) for k in OUTS}
                e["regret_mean"] = {k: float(v.mean()) for k, v in diffs.items()}
                e["regret_range95"] = {k: ci(v) for k, v in diffs.items()}
                e["share_of_splits_syn_worse_on_missed"] = float(np.mean(diffs["missed_per_100"] > 1e-9))
            r["split_half"][t] = e
        res[metric] = r
    return res


def main():
    rng = np.random.default_rng(0)
    models, real, data, groups = load()
    pools = {"best18": [m["name"] for m in models if m["ckpt"] == "best"],
             "all72": [m["name"] for m in models if m["epochs"] is None or m["epochs"] >= 25]}
    pools["best15_noD"] = [n for n in pools["best18"] if variant_of(n) != "D"]
    pools["all60_noD"] = [n for n in pools["all72"] if variant_of(n) != "D"]
    n_gl = int(sum(data[pools["best18"][0], "real"]["n_gt"] > 0))
    out = {"set": "dataset scenario (real test with dataset boxes; S-tests with blind twin boxes)",
           "n_images": len(real), "n_glioma_images": n_gl,
           "n_glioma_boxes": int(data[pools["best18"][0], "real"]["n_gt"].sum()), "n_groups": len(groups),
           "unit_missed": "per 100 annotated glioma boxes (not patients)",
           "bootstrap": N_BOOT, "splits": N_SPLIT, "pools": {}}
    for pool, names in pools.items():
        print(f"pool {pool}: {len(names)} models", flush=True)
        out["pools"][pool] = {"n_models": len(names), **analyse(names, data, groups, len(real), rng)}
    path = ROOT / "viz" / "regret.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    for pool, p in out["pools"].items():
        for metric in SELECT_BY:
            r = p[metric]
            print(f"\n== {pool}, choose by {metric}")
            for t in TESTS:
                e = r["in_sample"][t]
                o = e["real_outcome"]
                line = (f"  in-sample {t:14s} chosen {', '.join(e['chosen'])[:70]:70s} | real: missed {o['missed_per_100']:5.1f}"
                        f"/100, FA tumor-free {o['false_alarms_per_100_tumor_free']:5.1f}/100, glioma FA {o['glioma_false_alarms_per_100_non_glioma']:5.1f}/100")
                if t != "real":
                    g, c = e["regret"], e["regret_ci95"]
                    line += (f" | regret missed {g['missed_per_100']:+.1f} {np.round(c['missed_per_100'], 1).tolist()}"
                             f" FA {g['false_alarms_per_100_tumor_free']:+.1f} {np.round(c['false_alarms_per_100_tumor_free'], 1).tolist()}")
                print(line)
            for t in TESTS:
                e = r["split_half"][t]
                line = (f"  split-half {t:14s} real half B: missed {e['real_outcome_mean']['missed_per_100']:5.1f}/100,"
                        f" FA {e['real_outcome_mean']['false_alarms_per_100_tumor_free']:5.1f}/100, picks D {e['share_of_splits_choosing_D']:.0%}")
                if t != "real":
                    line += (f" | regret missed {e['regret_mean']['missed_per_100']:+.1f} {np.round(e['regret_range95']['missed_per_100'], 1).tolist()},"
                             f" syn worse in {e['share_of_splits_syn_worse_on_missed']:.0%} of splits")
                print(line)


if __name__ == "__main__":
    main()
