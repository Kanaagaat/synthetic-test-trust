"""Write colab/train_rq2_colab.ipynb (and a .py copy of its code for a local smoke test).

    python3 scripts/make_colab_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MD_INTRO = """# RQ2: YOLO11n on real vs synthetic gliomas

This notebook trains the RQ2 variants of the synthetic-data study, 3 seeds each, with the same settings as the bonus-task model.

**This version trains only C_full and D** (3 seeds each) **plus one control run `A_full_s0_ctrl`**. The 12 runs of A_full / A_low / B_low / C_low were trained on 21–22.11 in an earlier Colab session and are already on the Mac; the control run repeats A_full_s0 here to check that this environment reproduces the earlier one. Edit `TRAIN_VARIANTS` / `CONTROL` in section 3 to change this.

**Before you start**
1. Upload `rq2_data.zip` (from `synthetic_study/colab/` on the Mac) to Google Drive as **`MyDrive/synthetic_study/rq2_data.zip`**, or its pieces `rq2_data.zip.part01`, `part02`, ... into the same folder (they are joined automatically). Upload **all** pieces of one build: pieces of different builds joined together give a broken zip.
2. *Runtime → Change runtime type →* **T4 GPU**.
3. *Runtime → Run all.*

**If the session stops** (time limit, disconnect): just *Run all* again. Finished runs are skipped, an interrupted run continues from its `last.pt`. Runs go in seed order (every variant with seed 0 first), so a short session still gives one complete seed.

**At the end** the last cell writes `MyDrive/synthetic_study/rq2_runs.zip` (weights and logs of all runs). Download it and put it into `synthetic_study/colab/` on the Mac; evaluation on the real test and the S-test happens there.

Validation and test are always the **real** subset_v1 images; only the train list differs between variants (see `variants.json` in the zip)."""

CELLS = [
    ("markdown", MD_INTRO),
    ("markdown", "## 1. Install"),
    ("code", "%pip install -q ultralytics==8.4.190"),
    ("markdown", "## 2. Setup: Drive, data, weights"),
    ("code", r'''import hashlib
import json
import os
import platform
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import torch
import ultralytics
from ultralytics import YOLO

assert ultralytics.__version__ == "8.4.190", "Restart the runtime after the install cell."

try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

SMOKE = os.environ.get("RQ2_SMOKE") == "1"  # tiny local test of this notebook's code
if IN_COLAB:
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive/synthetic_study")
    WORK = Path("/content")
else:
    DRIVE = Path(os.environ["RQ2_DRIVE"])
    WORK = Path(os.environ["RQ2_WORK"])
ZIP = DRIVE / "rq2_data.zip"
RUNS = DRIVE / ("runs_smoke" if SMOKE else "runs")
DATA = WORK / "rq2_data"
PARTS = sorted(DRIVE.glob("rq2_data.zip.part*"))  # the zip may be uploaded in <10 MB parts
if PARTS:
    joined = WORK / "rq2_data.zip"
    with open(joined, "wb") as out:
        for part in PARTS:
            out.write(part.read_bytes())
    sources, ZIP = PARTS, joined
else:
    assert ZIP.is_file(), f"Upload rq2_data.zip (or its .partNN pieces) to {DRIVE}"
    sources = [ZIP]

# extract (again) only when the zip on Drive changed
zip_id = hashlib.sha1("|".join(f"{p.name}-{p.stat().st_size}-{p.stat().st_mtime}" for p in sources).encode()).hexdigest()
marker = DATA / ".zip_id"
if not marker.exists() or marker.read_text() != zip_id:
    shutil.rmtree(DATA, ignore_errors=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(WORK)
    marker.write_text(zip_id)
VARIANT_INFO = json.loads((DATA / "variants.json").read_text())

device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(device)
ENV = {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
       "gpu": gpu, "python": platform.python_version()}
print(ENV)

WEIGHTS = Path(os.environ.get("RQ2_INITIAL_WEIGHTS", WORK / "yolo11n.pt"))
if not WEIGHTS.is_file():  # same COCO checkpoint as the bonus task
    urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt", WEIGHTS)
for name, info in VARIANT_INFO.items():
    print(f"{name:7s} {info['train_images']:4d} train images, {info['glioma_images']:3d} with glioma - {info['note']}")'''),
    ("markdown", "## 3. Settings\n\nSame as the bonus-task model (`exp_epochs100`), except the seed and `save_period` (intermediate checkpoints are extra models for RQ1)."),
    ("code", r'''# A_full/A_low/B_low/C_low (12 runs) were trained 21-22.11 in an earlier Colab session and are on the Mac.
# Here: C_full and D, plus a control rerun of A_full_s0 (folder "A_full_s0_ctrl") that checks this
# environment against the earlier one. D goes first in every seed: it is the main experiment.
TRAIN_VARIANTS = ["D", "C_full"]
CONTROL = [("A_full", 0)]
SEEDS = [0, 1, 2]
assert all(v in VARIANT_INFO for v in TRAIN_VARIANTS + [c[0] for c in CONTROL]), "variant missing from the zip"
SETTINGS = dict(imgsz=512, epochs=100, batch=16, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
                weight_decay=0.0005, warmup_epochs=3.0, cos_lr=False, patience=50, deterministic=True,
                close_mosaic=10, workers=2, save_period=25, plots=False)
if SMOKE:
    SETTINGS.update(imgsz=256, epochs=2, batch=8, fraction=0.05, workers=0, save_period=1)
    TRAIN_VARIANTS, SEEDS = TRAIN_VARIANTS[:1], [0]
VAL = dict(conf=0.001, iou=0.7, batch=16, imgsz=SETTINGS["imgsz"])

def data_yaml(variant):
    y = DATA / f"{variant}.yaml"
    y.write_text(f"path: {DATA}\ntrain: {variant}.txt\nval: real/images/val\ntest: real/images/test\n"
                 "names:\n  0: Glioma\n  1: Meningioma\n  2: Pituitary\n")
    return y

QUEUE = [(v, s, "") for s in SEEDS for v in TRAIN_VARIANTS]  # seed-major: one full seed first
QUEUE[len(TRAIN_VARIANTS):len(TRAIN_VARIANTS)] = [(v, s, "_ctrl") for v, s in CONTROL]  # control after seed 0
print(len(QUEUE), "runs:", [f"{v}_s{s}{t}" for v, s, t in QUEUE])'''),
    ("markdown", "## 4. Train (resumable)"),
    ("code", r'''def resumable(ckpt_path):
    """Ultralytics strips the optimizer from last.pt when training ends; such a file cannot be
    resumed (train(resume=True) would silently start a brand-new default training instead)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0


def run_one(variant, seed, tag=""):
    name = f"{variant}_s{seed}{tag}"
    run_dir = RUNS / name
    done = run_dir / "done.json"
    if done.exists():
        print("skip (done):", name)
        return json.loads(done.read_text())
    last = run_dir / "weights" / "last.pt"
    t0 = time.time()
    if last.exists() and resumable(last):
        print("resume:", name)
        YOLO(str(last)).train(resume=True)
    elif last.exists():  # training finished, the session died before done.json was written
        print("finished earlier, validating only:", name)
    else:
        shutil.rmtree(run_dir, ignore_errors=True)
        print("train:", name)
        YOLO(str(WEIGHTS)).train(data=str(data_yaml(variant)), device=device, project=str(RUNS), name=name,
                                 exist_ok=False, seed=seed, **SETTINGS)
    best = run_dir / "weights" / "best.pt"
    r = YOLO(str(best)).val(data=str(data_yaml(variant)), split="val", device=device, plots=False,
                            project=str(run_dir), name="val_best", exist_ok=True, **VAL)
    info = {"run": name, "variant": variant, "seed": seed, "variant_info": VARIANT_INFO[variant],
            "val_all": {"mAP50": float(r.box.map50), "mAP50-95": float(r.box.map)},
            "val_glioma": {"AP50": float(r.box.ap50[0]), "AP50-95": float(r.box.ap[0])},
            "minutes_this_session": round((time.time() - t0) / 60, 1), "settings": SETTINGS, "env": ENV,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    done.write_text(json.dumps(info, indent=1))
    return info

RESULTS = [run_one(v, s, t) for v, s, t in QUEUE]'''),
    ("markdown", "## 5. Validation summary (real val)\n\nThe real test and the S-test are evaluated on the Mac with `scripts/eval_tests.py`."),
    ("code", r'''rows = []
for info in RESULTS:
    rows.append(f"{info['run']:15s}  val mAP50-95 {info['val_all']['mAP50-95']:.3f}   "
                f"glioma AP50 {info['val_glioma']['AP50']:.3f}  AP50-95 {info['val_glioma']['AP50-95']:.3f}")
print("\n".join(rows))'''),
    ("markdown", "## 6. Pack runs for the Mac\n\nKeeps `best.pt` and the intermediate checkpoints (optimizer state stripped), plus logs. Download `MyDrive/synthetic_study/rq2_runs.zip`."),
    ("code", r'''from ultralytics.utils.torch_utils import strip_optimizer

out = DRIVE / ("rq2_runs_smoke.zip" if SMOKE else "rq2_runs.zip")
tmp = WORK / "rq2_runs_pack"
shutil.rmtree(tmp, ignore_errors=True)
for run_dir in sorted(p for p in RUNS.iterdir() if (p / "done.json").exists()):
    dst = tmp / run_dir.name
    (dst / "weights").mkdir(parents=True)
    for f in ["done.json", "args.yaml", "results.csv"]:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, dst / f)
    for w in sorted((run_dir / "weights").glob("*.pt")):
        if w.name == "last.pt":
            continue
        strip_optimizer(str(w), s=str(dst / "weights" / w.name))
shutil.make_archive(str(out.with_suffix("")), "zip", root_dir=tmp)
print(out, f"{out.stat().st_size / 1e6:.0f} MB")'''),
]


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    code = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
            if not src.startswith("%pip"):
                code.append(src)
        nb["cells"].append(cell)
    out = ROOT / "colab" / "train_rq2_colab.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (ROOT / "colab" / "train_rq2_colab_code.py").write_text("# generated by make_colab_notebook.py, for the local smoke test\n" + "\n\n".join(code) + "\n")
    print(out)


if __name__ == "__main__":
    main()
