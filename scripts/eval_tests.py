"""RQ1 evaluation: one model on the real test vs the S-test (paired), glioma metrics only.

    .venv/bin/python scripts/eval_tests.py                                   # bonus model
    .venv/bin/python scripts/eval_tests.py --weights path/to/best.pt --name A_low_s1

Sets (both built from the real test split of subset_v1, same non-glioma images):
    real    the real test, minus glioma images whose twin is not labeled yet
    s_test  the same images, but every glioma image replaced by its labeled synthetic twin
Image i of one set pairs with image i of the other; they differ only in where gliomas come from.
Both are written as small YOLO datasets under datasets/ (symlinked images, copied labels).

How the numbers are made: `ultralytics val` itself (imgsz 512, conf 0.001, NMS IoU 0.7, its own
rect batching, multi-label NMS and box matching). We record the per-image statistics val passes to
its metrics (TP flags at IoU 0.50:0.95, confidence, classes) and compute from them:
    AP50, AP50-95   101-point interpolated AP, glioma class (same formula as ultralytics;
                    --check confirms it reproduces `val` on the full real test)
    recall@0.25     share of glioma boxes matched at IoU >= 0.5 by a glioma prediction with conf >= 0.25
Uncertainty: 2000 bootstrap resamples of whole leakage groups (split_manifest.csv `group`; a twin
belongs to its original's group); the gap s_test - real is computed on the same resample (paired).

Output: viz/rq1/<name>.json (+ cached per-image stats viz/rq1/stats_<name>_<set>.json)
"""
import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
BONUS = (ROOT / "weights" / "reference_detector.pt" if (ROOT / "weights" / "reference_detector.pt").exists() else ROOT.parent / "bonus_task" / "runs" / "exp_epochs100" / "weights" / "best.pt")
NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
GLIOMA, N_IOU = 0, 10


# ---------------------------------------------------------------- sets
def read_label_text(path: Path):
    return path.read_text() if path.exists() else ""


REAL_BLIND = ROOT / "generated" / "real_test_blind" / "labels_blind"


def build_sets(twin_labels="labels_blind", require_both=False, real_labels="dataset"):
    split = {r["file"]: r for r in csv.DictReader(open(SUBSET / "split_manifest.csv")) if r["split"] == "test"}
    twins = {}  # original file -> latest labeled twin row (manifest is chronological)
    for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv")):
        has = lambda d: (ROOT / r["file"]).parent.joinpath(d, Path(r["file"]).stem + ".txt").exists()  # noqa: E731
        if r["purpose"] == "s_test" and r["status"] == "labeled" and has(twin_labels) and \
                (not require_both or (has("labels") and has("labels_blind"))):
            twins[r["refs"]] = r
    real, syn, missing = [], [], []
    for f, r in sorted(split.items()):
        img = SUBSET / "images" / "test" / f
        lab = SUBSET / "labels" / "test" / (Path(f).stem + ".txt")
        is_glioma = any(l.split()[0] == "0" for l in read_label_text(lab).splitlines() if l.strip())
        if is_glioma and real_labels in ("blind", "dataset_subset"):
            blind = REAL_BLIND / (Path(f).stem + ".txt")
            if not blind.exists():  # not labeled yet, or the annotator could not see a tumor
                missing.append(f)
                continue
            if real_labels == "blind":  # same annotator as the twins
                lab = blind
        if is_glioma and f not in twins:
            missing.append(f)
            continue
        real.append({"id": f, "img": img, "label": lab, "group": r["group"]})
        if is_glioma:
            tp = ROOT / twins[f]["file"]
            syn.append({"id": f, "img": tp, "label": tp.parent / twin_labels / (tp.stem + ".txt"), "group": r["group"]})
        else:
            syn.append({"id": f, "img": img, "label": lab, "group": r["group"]})
    return real, syn, missing


def write_dataset(items, name):
    d = ROOT / "datasets" / name
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "test").mkdir(parents=True)
    (d / "labels" / "test").mkdir(parents=True)
    for x in items:
        (d / "images" / "test" / x["img"].name).symlink_to(x["img"])
        (d / "labels" / "test" / (x["img"].stem + ".txt")).write_text(read_label_text(x["label"]))
    yaml = d / "data.yaml"
    yaml.write_text(f"path: {d}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n"
                    + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()))
    return yaml


# ---------------------------------------------------------------- ultralytics val with per-image stats
def val_stats(weights, yaml, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    from ultralytics.utils import metrics as um

    captured = {}
    original = um.DetMetrics.update_stats

    def capture(self, stat):
        captured[stat["im_name"]] = {"tp": np.asarray(stat["tp"]).tolist(), "conf": np.asarray(stat["conf"]).tolist(),
                                     "pred_cls": np.asarray(stat["pred_cls"]).astype(int).tolist(),
                                     "target_cls": np.asarray(stat["target_cls"]).astype(int).tolist()}
        return original(self, stat)

    um.DetMetrics.update_stats = capture
    try:
        r = YOLO(str(weights)).val(data=str(yaml), split="test", imgsz=512, batch=16, conf=0.001, iou=0.7,
                                   plots=False, verbose=False, project=str(ROOT / "viz" / "rq1" / "_val"),
                                   name=yaml.parent.name, exist_ok=True)
    finally:
        um.DetMetrics.update_stats = original
    out = {"images": captured, "ultralytics_glioma": {"AP50": float(r.box.ap50[GLIOMA]), "AP50-95": float(r.box.ap[GLIOMA])}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- metrics
def glioma_record(s):
    pc = np.asarray(s["pred_cls"], dtype=int)
    keep = pc == GLIOMA
    tp = np.asarray(s["tp"], dtype=bool).reshape(-1, N_IOU)[keep] if len(pc) else np.zeros((0, N_IOU), bool)
    conf = np.asarray(s["conf"], dtype=float)[keep] if len(pc) else np.zeros(0)
    return {"conf": conf, "tp": tp, "n_gt": int((np.asarray(s["target_cls"]) == GLIOMA).sum())}


def ap_from(records):
    conf = np.concatenate([r["conf"] for r in records])
    tp = np.concatenate([r["tp"] for r in records])
    n_gt = sum(r["n_gt"] for r in records)
    if n_gt == 0:
        return np.full(N_IOU, np.nan)
    order = np.argsort(-conf, kind="stable")
    tpc = tp[order].cumsum(0)
    fpc = (~tp[order]).cumsum(0)
    recall = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc)
    x = np.linspace(0, 1, 101)
    aps = np.zeros(N_IOU)
    for k in range(N_IOU):
        # same sentinels as ultralytics 8.4 compute_ap: precision drops to 0 right after the last recall
        last = recall[-1, k] if len(recall) else 1.0
        mrec = np.concatenate(([0.0], recall[:, k], [last], [1.0]))
        mpre = np.concatenate(([1.0], precision[:, k], [0.0], [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        aps[k] = np.trapezoid(np.interp(x, mrec, mpre), x)
    return aps


def metrics(records):
    aps = ap_from(records)
    n_gt = sum(r["n_gt"] for r in records)
    hit = sum(int(r["tp"][r["conf"] >= 0.25, 0].sum()) for r in records)
    return {"AP50": float(aps[0]), "AP50-95": float(np.mean(aps)), "recall@0.25": hit / n_gt if n_gt else float("nan")}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BONUS))
    ap.add_argument("--name", default="bonus_exp_epochs100")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--check", action="store_true", help="also run val on the full real test and compare")
    ap.add_argument("--fresh", action="store_true", help="ignore cached stats")
    ap.add_argument("--twin-labels", default="labels_blind", choices=["labels_blind", "labels"],
                    help="labels_blind = drawn without model drafts (default); labels = draft-assisted")
    ap.add_argument("--real-labels", default="dataset", choices=["dataset", "blind", "dataset_subset"],
                    help="blind = real test gliomas relabeled blind by the twins' annotator; "
                         "dataset_subset = dataset boxes, but only on the images the annotator could label")
    ap.add_argument("--require-both", action="store_true",
                    help="only twins that have both blind and assisted labels (for the label-source comparison)")
    args = ap.parse_args()

    out_dir = ROOT / "viz" / "rq1"
    real, syn, missing = build_sets(args.twin_labels, args.require_both, args.real_labels)
    if args.real_labels == "blind":
        args.name += "__samelabeler"
    elif args.real_labels == "dataset_subset":
        args.name += "__samelabeler_subset_datasetboxes"
    if args.twin_labels != "labels_blind":
        args.name += "__assisted"
    if args.require_both:
        args.name += "__common"
    stats = {}
    for tag, items in (("real", real), ("s_test", syn)):
        cache = out_dir / f"stats_{args.name}_{tag}.json"  # name carries the label source
        if args.fresh:
            cache.unlink(missing_ok=True)
        stats[tag] = val_stats(args.weights, write_dataset(items, f"rq1_{tag}"), cache)
    R = [glioma_record(stats["real"]["images"][x["img"].name]) for x in real]
    S = [glioma_record(stats["s_test"]["images"][x["img"].name]) for x in syn]

    res = {"model": args.name, "weights": args.weights, "twin_labels": args.twin_labels,
           "real_labels": args.real_labels, "n_images": len(real),
           "n_glioma_images": sum(r["n_gt"] > 0 for r in R),
           "n_glioma_boxes": {"real": sum(r["n_gt"] for r in R), "s_test": sum(r["n_gt"] for r in S)},
           "excluded_originals_without_twin": missing,
           "real": metrics(R), "s_test": metrics(S),
           "ultralytics_val": {k: v["ultralytics_glioma"] for k, v in stats.items()}}
    res["gap"] = {k: res["s_test"][k] - res["real"][k] for k in res["real"]}

    groups = defaultdict(list)
    for i, x in enumerate(real):
        groups[x["group"]].append(i)
    keys = sorted(groups)
    rng = np.random.default_rng(0)
    boot = {s: defaultdict(list) for s in ("real", "s_test", "gap")}
    for _ in range(args.boot):
        idx = [i for g in rng.choice(len(keys), len(keys)) for i in groups[keys[g]]]
        mr, ms = metrics([R[i] for i in idx]), metrics([S[i] for i in idx])
        for k in mr:
            boot["real"][k].append(mr[k])
            boot["s_test"][k].append(ms[k])
            boot["gap"][k].append(ms[k] - mr[k])
    res["ci95"] = {s: {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in d.items()}
                   for s, d in boot.items()}
    res["bootstrap"] = {"resamples": args.boot, "unit": "leakage group", "n_groups": len(keys)}

    if args.check:
        full = val_stats(args.weights, SUBSET / "data.yaml" if False else write_dataset(
            [{"img": p, "label": SUBSET / "labels" / "test" / (p.stem + ".txt")}
             for p in sorted((SUBSET / "images" / "test").glob("*.jpg"))], "rq1_real_full"),
            out_dir / f"stats_{args.name}_real_full.json")
        res["check_full_real_test"] = {"ours": metrics([glioma_record(s) for s in full["images"].values()]),
                                       "ultralytics": full["ultralytics_glioma"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "excluded_originals_without_twin"}, indent=1))


if __name__ == "__main__":
    main()
