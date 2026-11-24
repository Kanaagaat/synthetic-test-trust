"""RQ1 and RQ2 over many models: every best.pt and intermediate checkpoint of the Colab runs.

    .venv/bin/python scripts/eval_many.py --runs colab/runs_A            # evaluate (cached) + summarize
    .venv/bin/python scripts/eval_many.py --runs colab/runs_A --runs colab/runs_BC

Models: <run>/weights/{best,epoch0,epoch25,epoch50,epoch75}.pt. Ultralytics counts epochs from 0, so
epochN.pt is the model after N+1 epochs. epoch0 (one epoch) is reported but kept out of the main RQ1
set (--rq1-min-epoch, default 25): it is barely trained and would only inflate rank agreement.
The bonus-task model is added as a reference (same recipe as A_full_s0, trained earlier in Colab).

Per-image statistics come from `ultralytics val` exactly as in eval_tests.py (same sets, same
matching, same AP formula). Three evaluation sets per model:
    dataset      real test (dataset boxes) vs S-test (blind twin boxes), all labeled pairs     -> RQ1
    samelabeler  real test gliomas relabeled blind by the twins' annotator vs S-test, 45 pairs -> RQ1
    full         the whole real test, dataset boxes                                            -> RQ2

Uncertainty: one joint bootstrap (2000 resamples of whole leakage groups) shared by all models, so
model differences, gaps and rank agreement are all paired.

Output: viz/rq1_many/summary.json (+ per-image stats cache in viz/rq1_many/stats/)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_tests as E  # noqa: E402

ROOT = E.ROOT
OUT = ROOT / "viz" / "rq1_many"
SCENARIOS = {"dataset": dict(twin_labels="labels_blind", require_both=False, real_labels="dataset"),
             "samelabeler": dict(twin_labels="labels_blind", require_both=False, real_labels="blind")}
METRICS = ("AP50", "recall@0.25", "AP50-95")
# S-tests from other generators: tag -> manifest purpose of their twins. They use the same images and
# pairs as the main S-test (its OpenAI twins swapped for the other generator's twins of the same
# real glioma), so all three tests are paired. --syn picks which synthetic test summarize() compares
# with the real test; the summary keeps the key "s_test" for it and records the choice in "syn_set".
OTHER_SYN = {"s_test_gemini": "s_test_gemini"}


def alias_syn(sets, recs, syn):
    """Make `syn` (e.g. s_test_gemini) the synthetic test that summarize() sees under the key s_test."""
    if syn == "s_test":
        return sets, recs
    recs = {**recs, **{(n, scen, "s_test"): v for (n, scen, tag), v in recs.items() if tag == syn}}
    sets = {scen: ({**d, "s_test": d[syn]} if syn in d else d) for scen, d in sets.items()}
    return sets, recs
CKPTS = ("epoch0", "epoch25", "epoch50", "epoch75", "best")


def find_models(run_dirs):
    models = [{"name": "bonus", "weights": E.BONUS, "run": "bonus", "variant": "bonus", "seed": None,
               "ckpt": "best", "epochs": None}]
    for rd in run_dirs:
        for run in sorted(p for p in Path(rd).iterdir() if (p / "done.json").exists()):
            variant, seed = run.name.rsplit("_s", 1)
            if not seed.isdigit():  # e.g. A_full_s0_ctrl: environment control, compared separately
                print("skip (control run):", run.name)
                continue
            for c in CKPTS:
                w = run / "weights" / f"{c}.pt"
                if w.exists():
                    models.append({"name": f"{run.name}@{c}", "weights": w, "run": run.name, "variant": variant,
                                   "seed": int(seed), "ckpt": c,
                                   "epochs": None if c == "best" else int(c[5:]) + 1})
    return models


def full_items():
    split = {r["file"]: r for r in csv.DictReader(open(E.SUBSET / "split_manifest.csv")) if r["split"] == "test"}
    return [{"id": f, "img": E.SUBSET / "images" / "test" / f,
             "label": E.SUBSET / "labels" / "test" / (Path(f).stem + ".txt"), "group": r["group"]}
            for f, r in sorted(split.items())]


def evaluate(models, extra=()):
    """Run val for every model on every set (cached) and return per-image glioma records.
    extra: tags from OTHER_SYN to evaluate as well (only when asked: a default run needs no other twins)."""
    sets = {}
    for scen, kw in SCENARIOS.items():
        real, syn, _ = E.build_sets(**kw)
        sets[scen] = {"real": real, "s_test": syn}
        for tag in extra:  # second generator: same images, its own twins
            sets[scen][tag] = E.swap_twins(syn, OTHER_SYN[tag], kw["twin_labels"])
    sets["full"] = {"real": full_items()}
    yamls = {(scen, tag): E.write_dataset(items, f"many_{scen}_{tag}")
             for scen, d in sets.items() for tag, items in d.items()}
    recs, other = {}, {}
    for k, m in enumerate(models):
        for (scen, tag), y in yamls.items():
            cache = OUT / "stats" / f"{m['name'].replace('@', '__')}__{scen}_{tag}.json"
            st = E.val_stats(m["weights"], y, cache)
            recs[m["name"], scen, tag] = [E.glioma_record(st["images"][x["img"].name]) for x in sets[scen][tag]]
            if scen == "full":
                other[m["name"]] = {"ultralytics_glioma": st["ultralytics_glioma"],
                                    "images": [st["images"][x["img"].name] for x in sets[scen][tag]]}
        print(f"[{k + 1}/{len(models)}] {m['name']}", flush=True)
    return sets, recs, other


def background_fp(stats_list, labels):
    """Share of tumor-free test images with any prediction (any class) at conf >= 0.25."""
    n = fp = 0
    for s, lab in zip(stats_list, labels):
        if lab:
            continue
        n += 1
        fp += any(c >= 0.25 for c in s["conf"])
    return fp / n if n else float("nan")


def kendall_tau_b(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    conc = disc = tx = ty = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            conc += 1
        else:
            disc += 1
    den = np.sqrt((conc + disc + tx) * (conc + disc + ty))
    return float((conc - disc) / den) if den else float("nan")


def ci(v):
    v = np.asarray(v, float)
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def summarize(models, sets, recs, other, n_boot, min_epoch):
    names = [m["name"] for m in models]
    rq1_names = [m["name"] for m in models if m["variant"] != "bonus"
                 and (m["epochs"] is None or m["epochs"] >= min_epoch)]
    res = {"models": [{k: (str(v) if isinstance(v, Path) else v) for k, v in m.items()} for m in models],
           "rq1_models": rq1_names, "bootstrap": {"resamples": n_boot, "unit": "leakage group"}}
    rng = np.random.default_rng(0)

    # ---------------- RQ1, per scenario
    res["rq1"] = {}
    for scen in SCENARIOS:
        real = sets[scen]["real"]
        groups = defaultdict(list)
        for i, x in enumerate(real):
            groups[x["group"]].append(i)
        keys = sorted(groups)
        point = {n: {t: E.metrics(recs[n, scen, t]) for t in ("real", "s_test")} for n in names}
        boot = {n: {t: defaultdict(list) for t in ("real", "s_test")} for n in names}
        for _ in range(n_boot):
            idx = [i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]]
            for n in names:
                for t in ("real", "s_test"):
                    mt = E.metrics([recs[n, scen, t][i] for i in idx])
                    for k in METRICS:
                        boot[n][t][k].append(mt[k])
        per_model = {}
        for n in names:
            per_model[n] = {t: point[n][t] for t in ("real", "s_test")}
            per_model[n]["gap"] = {k: point[n]["s_test"][k] - point[n]["real"][k] for k in METRICS}
            per_model[n]["gap_ci95"] = {k: ci(np.subtract(boot[n]["s_test"][k], boot[n]["real"][k])) for k in METRICS}

        agg = {}
        for k in METRICS:
            r = np.array([point[n]["real"][k] for n in rq1_names])
            s = np.array([point[n]["s_test"][k] for n in rq1_names])
            br = np.array([boot[n]["real"][k] for n in rq1_names])   # models x resamples
            bs = np.array([boot[n]["s_test"][k] for n in rq1_names])
            gaps = bs - br
            # pairwise order
            pairs = {"n": 0, "same_order": 0, "real_sure": 0, "syn_sure": 0, "both_sure_same": 0,
                     "both_sure_opposite": 0, "syn_sure_real_not": 0, "ties": 0}
            for i, j in combinations(range(len(rq1_names)), 2):
                pairs["n"] += 1
                dr, ds = r[i] - r[j], s[i] - s[j]
                if dr == 0 or ds == 0:
                    pairs["ties"] += 1
                pairs["same_order"] += int(np.sign(dr) == np.sign(ds) and dr != 0)
                cr, cs = ci(br[i] - br[j]), ci(bs[i] - bs[j])
                sure_r = cr[0] > 0 or cr[1] < 0
                sure_s = cs[0] > 0 or cs[1] < 0
                pairs["real_sure"] += sure_r
                pairs["syn_sure"] += sure_s
                if sure_r and sure_s:
                    pairs["both_sure_same" if np.sign(dr) == np.sign(ds) else "both_sure_opposite"] += 1
                if sure_s and not sure_r:
                    pairs["syn_sure_real_not"] += 1
            # Kendall tau overall and per seed (bootstrap CI for the overall one)
            tau = kendall_tau_b(r, s)
            tau_boot = [kendall_tau_b(br[:, b], bs[:, b]) for b in range(0, n_boot, 4)]
            seeds = sorted({m["seed"] for m in models if m["name"] in rq1_names})
            tau_seed = {}
            for sd in seeds:
                ix = [i for i, n in enumerate(rq1_names) if next(m for m in models if m["name"] == n)["seed"] == sd]
                tau_seed[str(sd)] = kendall_tau_b(r[ix], s[ix])
            # does the gap depend on model quality?
            gap_pt = s - r
            agg[k] = {"mean_real": float(r.mean()), "mean_s_test": float(s.mean()),
                      "mean_gap": float(gap_pt.mean()), "mean_gap_ci95": ci(gaps.mean(0)),
                      "models_gap_ci_above_0": int(sum(ci(g)[0] > 0 for g in gaps)),
                      "models_gap_ci_below_0": int(sum(ci(g)[1] < 0 for g in gaps)),
                      "n_models": len(rq1_names),
                      "pairs": pairs, "kendall_tau_b": tau, "kendall_tau_b_ci95": ci(tau_boot),
                      "kendall_tau_b_per_seed": tau_seed,
                      "corr_gap_vs_real": float(np.corrcoef(r, gap_pt)[0, 1]) if r.std() > 0 else float("nan"),
                      "range_real": [float(r.min()), float(r.max())], "range_s_test": [float(s.min()), float(s.max())]}
        # variant contrasts (seed-mean of best.pt) on both tests: does the S-test show the same difference?
        best = defaultdict(list)
        for m in models:
            if m["ckpt"] == "best" and m["variant"] != "bonus":
                best[m["variant"]].append(m["name"])
        contrasts = {}
        for a, b in combinations(sorted(best), 2):
            c = {}
            for k in METRICS:
                d = {t: np.mean([boot[n][t][k] for n in best[b]], 0) - np.mean([boot[n][t][k] for n in best[a]], 0)
                     for t in ("real", "s_test")}
                pt = {t: float(np.mean([point[n][t][k] for n in best[b]]) - np.mean([point[n][t][k] for n in best[a]]))
                      for t in ("real", "s_test")}
                c[k] = {"real": pt["real"], "real_ci95": ci(d["real"]), "s_test": pt["s_test"], "s_test_ci95": ci(d["s_test"]),
                        "s_test_minus_real": pt["s_test"] - pt["real"], "s_test_minus_real_ci95": ci(d["s_test"] - d["real"])}
            contrasts[f"{b} - {a}"] = c
        res["rq1"][scen] = {"n_images": len(real), "n_groups": len(keys), "variant_contrasts": contrasts,
                            "n_glioma_images": sum(rr["n_gt"] > 0 for rr in recs[names[0], scen, "real"]),
                            "aggregate": agg, "per_model": per_model}

    # ---------------- RQ2: real test, best.pt per run
    items = sets["full"]["real"]
    labels = [E.read_label_text(x["label"]).strip() for x in items]
    groups = defaultdict(list)
    for i, x in enumerate(items):
        groups[x["group"]].append(i)
    keys = sorted(groups)
    runs = [m for m in models if m["ckpt"] == "best"]
    per_run, boot = {}, defaultdict(list)
    for _ in range(n_boot):
        idx = [i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]]
        for m in runs:
            boot[m["name"]].append(E.metrics([recs[m["name"], "full", "real"][i] for i in idx])["AP50-95"])
    for m in runs:
        n = m["name"]
        mt = E.metrics(recs[n, "full", "real"])
        ours, ul = mt["AP50-95"], other[n]["ultralytics_glioma"]["AP50-95"]
        per_run[n] = {**mt, "AP50-95_ci95": ci(boot[n]), "bg_fp_rate@0.25": background_fp(other[n]["images"], labels),
                      "check_vs_ultralytics_AP50-95": ul - ours}
    variants = sorted({m["variant"] for m in runs if m["variant"] != "bonus"})
    by_var = {}
    for v in variants:
        ns = [m["name"] for m in runs if m["variant"] == v]
        vals = {k: [per_run[n][k] for n in ns] for k in ("AP50", "AP50-95", "recall@0.25", "bg_fp_rate@0.25")}
        by_var[v] = {"runs": ns, **{k: {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                                        "values": x} for k, x in vals.items()},
                     "_boot_mean": np.mean([boot[n] for n in ns], axis=0)}
    contrasts = {}
    for a, b in combinations(variants, 2):
        d = by_var[b]["_boot_mean"] - by_var[a]["_boot_mean"]
        contrasts[f"{b} - {a}"] = {"AP50-95_mean_diff": by_var[b]["AP50-95"]["mean"] - by_var[a]["AP50-95"]["mean"],
                                   "ci95": ci(d)}
    for v in by_var.values():
        v.pop("_boot_mean")
    res["rq2"] = {"set": "full real test (dataset boxes)", "n_images": len(items), "n_groups": len(keys),
                  "per_run": per_run, "by_variant": by_var, "contrasts_seed_mean": contrasts}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True, help="folder with <variant>_s<seed>/ run dirs")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--rq1-min-epoch", type=int, default=25)
    ap.add_argument("--only-eval", action="store_true")
    ap.add_argument("--syn", default="s_test", choices=["s_test", *OTHER_SYN],
                    help="synthetic test to compare with the real one (default: the main OpenAI S-test)")
    args = ap.parse_args()
    models = find_models([ROOT / r for r in args.runs])
    print(len(models), "models")
    sets, recs, other = evaluate(models, () if args.syn == "s_test" else (args.syn,))
    if args.only_eval:
        return
    sets, recs = alias_syn(sets, recs, args.syn)
    res = summarize(models, sets, recs, other, args.boot, args.rq1_min_epoch)
    if args.syn != "s_test":  # the default summary.json keeps its old keys
        res = {"syn_set": args.syn, **res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("summary.json" if args.syn == "s_test" else f"summary_{args.syn}.json")).write_text(json.dumps(res, indent=1))
    for scen, d in res["rq1"].items():
        print(f"\nRQ1 [{scen}] {d['n_images']} images, {d['n_glioma_images']} with glioma, {d['n_groups']} groups")
        for k, a in d["aggregate"].items():
            p = a["pairs"]
            print(f"  {k:12s} real {a['mean_real']:.3f} s_test {a['mean_s_test']:.3f} gap {a['mean_gap']:+.3f} "
                  f"{np.round(a['mean_gap_ci95'], 3).tolist()} | gap>0 sure {a['models_gap_ci_above_0']}/{a['n_models']} "
                  f"| same order {p['same_order']}/{p['n']}, both sure same {p['both_sure_same']}, opposite "
                  f"{p['both_sure_opposite']}, syn sure/real not {p['syn_sure_real_not']} | tau {a['kendall_tau_b']:.2f} "
                  f"{np.round(a['kendall_tau_b_ci95'], 2).tolist()} per seed {({s: round(t, 2) for s, t in a['kendall_tau_b_per_seed'].items()})}")
        for c, cc in d["variant_contrasts"].items():
            for k, v in cc.items():
                print(f"  {c} {k:12s} real {v['real']:+.3f} {np.round(v['real_ci95'], 3).tolist()}  s_test {v['s_test']:+.3f} "
                      f"{np.round(v['s_test_ci95'], 3).tolist()}  difference {v['s_test_minus_real']:+.3f} "
                      f"{np.round(v['s_test_minus_real_ci95'], 3).tolist()}")
    print("\nRQ2 (full real test, best.pt)")
    for v, d in res["rq2"]["by_variant"].items():
        print(f"  {v:8s} AP50-95 {d['AP50-95']['mean']:.3f} ± {d['AP50-95']['sd']:.3f}  AP50 {d['AP50']['mean']:.3f}  "
              f"recall {d['recall@0.25']['mean']:.3f}  bg FP {d['bg_fp_rate@0.25']['mean']:.3f}")
    for c, d in res["rq2"]["contrasts_seed_mean"].items():
        print(f"  {c}: {d['AP50-95_mean_diff']:+.3f} {np.round(d['ci95'], 3).tolist()}")


if __name__ == "__main__":
    main()
