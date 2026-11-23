"""RQ3: can a simple classifier tell real glioma MRIs from synthetic ones? (domain gap)

    .venv/bin/python scripts/rq3_classifier.py

Classes: real = glioma images of subset_v1; synthetic = every non-rejected image of the main batches
(s_test, train_low, train_full; pilots excluded, other prompts). Only gliomas on both sides, so the
classifier cannot use "which tumor class / no tumor" as a cue.

Two feature sets, both on the same preprocessing (grayscale, 512x512 LANCZOS; nothing about the
original file size, colour mode or JPEG tables is used):
  handcrafted  brightness/contrast/sharpness/head fraction (as in analyze_pilot.py), intensity
               histogram of the head, gradient-magnitude percentiles, high-pass residual (noise)
               inside and outside the head, background level
  detector     embedding of the bonus-task detector (YOLO11n trained on real data; ultralytics
               `embed`). Real images here are only val + test gliomas: the detector saw the train ones.
Classifier: L2 logistic regression (fixed strength, no tuning). Evaluation: 5-fold cross-validation
by leakage group (a synthetic image belongs to the group of its reference), out-of-fold AUC with a
2000x bootstrap over groups. Also single-feature AUCs (handcrafted) for interpretation.

Output: viz/rq3_classifier.json
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
BONUS = (ROOT / "weights" / "reference_detector.pt" if (ROOT / "weights" / "reference_detector.pt").exists() else ROOT.parent / "bonus_task" / "runs" / "exp_epochs100" / "weights" / "best.pt")
BATCHES = ("s_test", "train_low", "train_full")


def load_items():
    split = {r["file"]: r for r in csv.DictReader(open(SUBSET / "split_manifest.csv"))}
    items = []
    for f, r in split.items():
        lab = SUBSET / "labels" / r["split"] / (Path(f).stem + ".txt")
        if lab.exists() and any(l.split()[0] == "0" for l in lab.read_text().splitlines() if l.strip()):
            items.append({"path": SUBSET / "images" / r["split"] / f, "y": 0, "source": f"real_{r['split']}",
                          "group": r["group"]})
    for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv")):
        if r["purpose"] in BATCHES and r["status"] != "rejected":
            items.append({"path": ROOT / r["file"], "y": 1, "source": r["purpose"], "group": split[r["refs"]]["group"]})
    return items


def gray512(p):
    return np.asarray(Image.open(p).convert("L").resize((512, 512), Image.LANCZOS), dtype=np.float32)


HC_NAMES = (["brain_mean", "brain_std", "edge_var", "head_fraction", "bg_mean", "bg_std",
             "resid_std_head", "resid_std_bg"] + [f"hist_{i}" for i in range(16)]
            + [f"grad_p{q}" for q in (50, 90, 99)])


def handcrafted(p):
    a = gray512(p)
    im = Image.fromarray(a.astype(np.uint8))
    fg = a > 20  # head mask, same threshold idea as analyze_pilot.intensity
    edges = np.asarray(im.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    blur = np.asarray(im.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    resid = a - blur
    bg = ~fg
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)[fg]
    hist = np.histogram(a[fg], bins=16, range=(20, 256))[0] / max(fg.sum(), 1)
    f = [a[fg].mean(), a[fg].std(), edges[fg].var(), fg.mean(),
         a[bg].mean() if bg.any() else 0, a[bg].std() if bg.any() else 0,
         resid[fg].std(), resid[bg].std() if bg.any() else 0]
    return np.array(f + hist.tolist() + np.percentile(g, [50, 90, 99]).tolist(), dtype=np.float64)


def detector_embeddings(paths):
    from ultralytics import YOLO
    model = YOLO(str(BONUS))
    out = []
    for i in range(0, len(paths), 32):
        # same preprocessing as the handcrafted set: grayscale -> 512 -> 3 channels
        ims = [np.stack([gray512(p).astype(np.uint8)] * 3, -1) for p in paths[i:i + 32]]
        out += [e.cpu().numpy() for e in model.embed(ims, imgsz=512, verbose=False)]
    return np.array(out, dtype=np.float64)


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sv = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def logreg_fit_predict(Xtr, ytr, Xte, lam=1.0, iters=500):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w, b = np.zeros(Xtr.shape[1]), 0.0
    # class weights so the larger class does not dominate
    cw = np.where(ytr == 1, len(ytr) / (2 * ytr.sum()), len(ytr) / (2 * (len(ytr) - ytr.sum())))
    lr = 0.5
    for _ in range(iters):  # plain gradient descent on the weighted, L2-penalized log-loss
        z = Xtr @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        gr = cw * (p - ytr)
        w -= lr * (Xtr.T @ gr / len(ytr) + lam * w / len(ytr))
        b -= lr * gr.mean()
    return Xte @ w + b, w


def group_folds(groups, k=5, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.array(sorted(set(groups)))
    rng.shuffle(ug)
    fold_of = {g: i % k for i, g in enumerate(ug)}
    return np.array([fold_of[g] for g in groups])


def cv_auc(X, y, groups, n_boot=2000):
    folds = group_folds(groups)
    oof = np.zeros(len(y))
    for f in range(folds.max() + 1):
        tr, te = folds != f, folds == f
        oof[te], _ = logreg_fit_predict(X[tr], y[tr], X[te])
    _, w_full = logreg_fit_predict(X, y, X)
    gidx = defaultdict(list)
    for i, g in enumerate(groups):
        gidx[g].append(i)
    keys = sorted(gidx)
    rng = np.random.default_rng(1)
    boots = []
    for _ in range(n_boot):
        idx = [i for g in rng.choice(len(keys), len(keys)) for i in gidx[keys[g]]]
        boots.append(auc(y[idx], oof[idx]))
    return {"auc": auc(y, oof), "ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            "n_real": int((y == 0).sum()), "n_synthetic": int(y.sum()), "n_groups": len(keys)}, oof, w_full


def main():
    items = load_items()
    y = np.array([x["y"] for x in items])
    groups = [x["group"] for x in items]
    src = np.array([x["source"] for x in items])
    print({s: int((src == s).sum()) for s in sorted(set(src))})
    res = {"counts": {s: int((src == s).sum()) for s in sorted(set(src))}}

    Xh = np.array([handcrafted(x["path"]) for x in items])
    res["handcrafted"], oof, w = cv_auc(Xh, y, groups)
    res["handcrafted"]["per_batch_auc"] = {b: auc(y[(src == b) | (y == 0)], oof[(src == b) | (y == 0)]) for b in BATCHES}
    res["handcrafted"]["top_coefficients"] = sorted(((HC_NAMES[i], round(float(w[i]), 3)) for i in range(len(w))),
                                                    key=lambda t: -abs(t[1]))[:8]
    single = []
    for i, n in enumerate(HC_NAMES):  # in-sample, one feature, direction-free AUC
        a = auc(y, Xh[:, i])
        single.append((n, round(max(a, 1 - a), 3), "synthetic higher" if a > 0.5 else "real higher"))
    res["single_feature_auc"] = sorted(single, key=lambda t: -t[1])[:10]
    res["feature_means"] = {n: {"real": round(float(Xh[y == 0, i].mean()), 3), "synthetic": round(float(Xh[y == 1, i].mean()), 3)}
                            for i, n in enumerate(HC_NAMES) if not n.startswith("hist_")}

    # detector embeddings: real only from val/test (the detector was trained on the train images)
    keep = np.array([x["y"] == 1 or x["source"] in ("real_val", "real_test") for x in items])
    Xd = detector_embeddings([x["path"] for x, k in zip(items, keep) if k])
    yd, gd, sd = y[keep], [g for g, k in zip(groups, keep) if k], src[keep]
    res["detector"], oofd, _ = cv_auc(Xd, yd, gd)
    res["detector"]["per_batch_auc"] = {b: auc(yd[(sd == b) | (yd == 0)], oofd[(sd == b) | (yd == 0)]) for b in BATCHES}
    res["detector"]["embedding_dim"] = int(Xd.shape[1])
    # same real subset with handcrafted features, for a like-for-like comparison
    res["handcrafted_valtest_real"], _, _ = cv_auc(Xh[keep], yd, gd)

    out = ROOT / "viz" / "rq3_classifier.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k != "feature_means"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
