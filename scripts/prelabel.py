"""Draft boxes for synthetic images with the bonus-task model, plus a real-image comparison.

    .venv/bin/python scripts/prelabel.py --purpose pilot_v2

What it writes (under generated/<purpose>/):
    labels_draft/<name>.txt   YOLO draft label, class forced to 0 (glioma by construction);
                              empty file = the model found nothing, the box must be drawn by hand
    prelabel_preds.csv        every prediction with conf >= 0.05 (class the model *thought* it was)
and viz/<purpose>_drafts.jpg with the draft boxes drawn.

Comparison: the same model and thresholds on the real **val** glioma images (not test, to keep the
test untouched). Image-level numbers only, because synthetic images have no ground truth yet:
    detected@0.25   share of images with any box at conf >= 0.25
    glioma@0.25     share with a box predicted as class 0 at conf >= 0.25
    top_conf        highest confidence on the image
The drafts are NOT labels: every box is checked by a person (box_checked in the manifest).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SUBSET = (ROOT / "data" / "subset_v1" if (ROOT / "data" / "subset_v1").exists() else ROOT.parent / "bonus_task" / "data" / "subset_v1")
WEIGHTS = (ROOT / "weights" / "reference_detector.pt" if (ROOT / "weights" / "reference_detector.pt").exists() else ROOT.parent / "bonus_task" / "runs" / "exp_epochs100" / "weights" / "best.pt")
NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
CONF_DRAFT, CONF_LOG, IOU_NMS, IMGSZ = 0.25, 0.05, 0.7, 512


def predict(model, paths):
    out = {}
    for i in range(0, len(paths), 16):
        for p, r in zip(paths[i:i + 16], model.predict([str(x) for x in paths[i:i + 16]], imgsz=IMGSZ,
                                                         conf=CONF_LOG, iou=IOU_NMS, verbose=False)):
            b = r.boxes
            out[p] = [{"cls": int(c), "conf": float(s), "xywhn": [float(v) for v in xy]}
                      for c, s, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xywhn.tolist())]
    return out


def image_level(preds):
    det = [any(d["conf"] >= CONF_DRAFT for d in v) for v in preds.values()]
    gl = [any(d["conf"] >= CONF_DRAFT and d["cls"] == 0 for d in v) for v in preds.values()]
    top = [max([d["conf"] for d in v], default=0.0) for v in preds.values()]
    return {"n": len(preds), "detected@0.25": round(float(np.mean(det)), 3),
            "glioma@0.25": round(float(np.mean(gl)), 3),
            "top_conf_median": round(float(np.median(top)), 3),
            "top_conf_mean": round(float(np.mean(top)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--purpose", default="pilot_v2")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(ROOT / "generated" / "manifest.csv"))
            if r["purpose"] == args.purpose and r["status"] == "accepted_needs_box"]
    syn = [ROOT / r["file"] for r in rows]
    gen_dir = syn[0].parent
    model = YOLO(str(WEIGHTS))

    syn_preds = predict(model, syn)
    val_gl = sorted(p for p in (SUBSET / "images" / "val").glob("*.jpg")
                    if open(SUBSET / "labels" / "val" / (p.stem + ".txt")).read().startswith("0 "))
    val_preds = predict(model, val_gl)

    # draft labels + log
    (gen_dir / "labels_draft").mkdir(exist_ok=True)
    with open(gen_dir / "prelabel_preds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "pred_class", "pred_name", "conf", "x", "y", "w", "h"])
        for p, ds in syn_preds.items():
            for d in ds:
                w.writerow([p.name, d["cls"], NAMES[d["cls"]], round(d["conf"], 4), *[round(v, 6) for v in d["xywhn"]]])
            # drafts use glioma predictions only: a box the model calls Meningioma/Pituitary on a
            # synthetic glioma image is a model false positive (e.g. a normal pituitary gland), not the tumor
            keep = [d for d in ds if d["conf"] >= CONF_DRAFT and d["cls"] == 0]
            (gen_dir / "labels_draft" / (p.stem + ".txt")).write_text(
                "".join(f"0 {' '.join(f'{v:.6f}' for v in d['xywhn'])}\n" for d in keep))

    # sheet with drafts
    T, cols = 320, 4
    n = len(syn)
    sheet = Image.new("RGB", (cols * T, ((n + cols - 1) // cols) * (T + 18)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, p in enumerate(syn):
        im = Image.open(p).convert("RGB").resize((T, T))
        d = ImageDraw.Draw(im)
        for b in syn_preds[p]:
            if b["conf"] < CONF_DRAFT:
                continue
            x, y, w_, h_ = (v * T for v in b["xywhn"])
            color = (60, 220, 60) if b["cls"] == 0 else (255, 160, 0)
            d.rectangle([x - w_ / 2, y - h_ / 2, x + w_ / 2, y + h_ / 2], outline=color, width=2)
            d.text((x - w_ / 2 + 2, y - h_ / 2 - 11), f"{NAMES[b['cls']][:4]} {b['conf']:.2f}", fill=color)
        X, Y = (i % cols) * T, (i // cols) * (T + 18)
        sheet.paste(im, (X, Y + 18))
        top = max([b["conf"] for b in syn_preds[p]], default=0)
        dr.text((X + 4, Y + 3), f"{p.stem[-4:]}  top {top:.2f}" + ("  NO DRAFT" if top < CONF_DRAFT else ""), fill="black")
    sheet_path = ROOT / "viz" / f"{args.purpose}_drafts.jpg"
    sheet.save(sheet_path, quality=90)

    summary = {"weights": str(WEIGHTS.relative_to(ROOT.parent)), "conf_draft": CONF_DRAFT,
               "synthetic": image_level(syn_preds), "real_val_glioma": image_level(val_preds),
               "synthetic_pred_classes@0.25": {NAMES[c]: sum(1 for v in syn_preds.values() for d in v
                                                             if d["conf"] >= CONF_DRAFT and d["cls"] == c)
                                               for c in NAMES}}
    (ROOT / "viz" / f"{args.purpose}_prelabel_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("sheet:", sheet_path)


if __name__ == "__main__":
    main()
