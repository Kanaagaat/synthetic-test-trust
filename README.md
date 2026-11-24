# Can a synthetic test set be trusted?

Code, synthetic data and results of a paired study: **can a test set made by a general-purpose image
generator replace a real test set for brain-MRI tumor detection?**

> **All images under `generated/` are AI-generated** (OpenAI gpt-image; the second S-test: Google Gemini).
> They are not patient data and must not be used for any clinical purpose.

## The study in short

- **Data.** A fixed 1,350-image subset (train/val/test = 945/202/203) of the public Kaggle dataset
  [MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
  (CC0): glioma, meningioma, pituitary tumor and tumor-free 2D slices, split by leakage groups.
- **S-test.** For each of the 52 real glioma test images, one synthetic *twin* was generated from that
  image (same plane, tumor type, size and number of tumors; different anatomy and location). The S-test
  is the real test with every glioma image replaced by its twin; all other images are unchanged. Twin
  boxes were drawn blind. 51 twins were accepted.
- **Second S-test.** The same 52 references were given to a second general-purpose generator (Google
  Gemini, web app) with the same prompt and substitutions; all 52 Gemini twins were accepted and boxed
  blind. No model was trained on Gemini images, so this test checks whether a bias of the S-test is tied
  to the generator of the training images.
- **Models.** YOLO11n trained in six regimes (3 seeds each); with checkpoints after ~25/50/75 epochs and
  `best.pt`, 72 models are evaluated on both tests.

| Regime | Glioma training images | Real test AP50-95 | Real test AP50 | S-test AP50 |
|---|---|---:|---:|---:|
| A_full | 245 real | 0.621 | 0.864 | 0.976 |
| A_low | 61 real (25%) | 0.488 | 0.725 | 0.947 |
| B_low | 61 real + 182 copies | 0.457 | 0.672 | 0.940 |
| C_low | 61 real + 182 synthetic | 0.510 | 0.768 | 0.979 |
| C_full | 245 real + 245 synthetic | 0.621 | 0.834 | 0.967 |
| D | 245 synthetic, no real | **0.403** | **0.614** | **0.980** |

Glioma class, `best.pt`, mean of 3 seeds. AP50-95: full real test (203 images); AP50 columns: the 51
real/twin pairs (202 images).

**Findings.**
1. The S-test overstates almost every model (mean gap over 72 models: AP50 +0.25, recall +0.30), more
   for weaker models, and preserves their ranking poorly (Kendall τ = 0.23).
2. It changes the conclusions: it hides most of the loss from scarce real data, and it says synthetic
   gliomas fully replace the missing real ones, which the real test refutes.
3. Beyond this scale compression, it favors models trained on synthetic images: regime D, which never
   saw a real glioma, is the worst on the real test and on par with the best on the S-test
   (residual over never-synthetic models of equal real performance: AP50 +0.11, recall +0.22). The
   Gemini S-test gives a residual at least as large (AP50 +0.11, recall +0.31), so there is no evidence
   that the bias is tied to the generator of the training images.
4. Cost of choosing a model by a synthetic test: picking one of the 18 trained detectors by either
   S-test, instead of the real test, selects a model that misses 12-19 more of every 100 glioma boxes on
   held-out real images (stratified split-half, 1,000 splits; worse in 95-100% of splits). Without the
   synthetic-only regime D among the candidates the cost falls to 3-5 and is uncertain. Counted per
   annotated box, not per patient (the dataset has no patient IDs).
5. As training data, synthetic gliomas give a small, uncertain gain with scarce real data, none with
   full data, and cannot replace real gliomas.
6. A logistic regression separates synthetic from real images with AUC 0.98; we found no sign that the
   synthetic images copy their references or the validation/test images.

The scope is one training generator (two for the S-test), one prompt, one dataset and one detector. A
paper is in preparation.

## Repository layout

```
data/subset_v1/            real images and YOLO labels with the fixed split (split_manifest.csv: leakage groups)
weights/reference_detector.pt  YOLO11n trained on the real train split: box drafts, RQ3 embedding
prompts/                   generation prompts v1-v4 (v4 for all main batches; notes partly in Russian)
refs/pools.json            reference pools: train references for training images, test references only for twins
generated/
  manifest.csv             one row per generated image: reference, prompt version, revised prompt,
                           status, rejection reason, box review
  s_test/                  twins of the real test gliomas
                           labels_blind/ = evaluation boxes (drawn blind); labels/ = boxes reviewed from model drafts
  s_test_gemini/           Gemini twins of the same references (labels_blind/, prompts/ = exact prompt per twin,
                           spec.json, compare_sheet.jpg = reference | OpenAI twin | Gemini twin);
                           high_reasoning/ = 13 superseded images made with another reasoning level (not used)
  train_low/, train_full/  synthetic training gliomas (labels/ = reviewed boxes, labels_draft/ = model drafts)
  real_test_blind/         blind re-annotation of the real test gliomas by the same annotator (one-annotator scenario)
  pilot*/                  prompt pilots (not used in the experiments)
colab/                     training notebook (Colab, T4) and its code;
                           runs_*/<run>/ = args.yaml, results.csv, done.json (weights: see Releases)
scripts/                   all code
viz/                       results: rq1_many/summary.json, selfgen.json (and *_s_test_gemini.json for the
                           Gemini S-test), rq1_many/visible_control.json, regret.json (model-selection cost),
                           rq3_classifier.json, leak_check.json, batch_table.csv, figures (viz/article/),
                           contact sheets
```

## Reproducing

Python 3.13; `pip install -r requirements.txt`. All scripts are run from the repository root.

1. **Reference pools:** `python scripts/make_pools.py`.
2. **Generation** needs the Codex CLI with its image-generation tool (an OpenAI account and usage limits):
   ```bash
   python scripts/generate_codex.py --prompt prompts/glioma_t1c_v4.md --pool S_test_refs --n 0 --per-ref 1 --seed 11 --out generated/s_test --purpose s_test --prefix syn_gl_st
   python scripts/generate_codex.py --prompt prompts/glioma_t1c_v4.md --pool R_train --n 0 --per-ref 1 --seed 31 --out generated/train_full --purpose train_full --prefix syn_gl_fu
   ```
   `train_low` used the same prompt with `--pool low_keep --per-ref 3 --prefix syn_gl_lo`. The generator
   is not deterministic, so new images will differ; the images used in the study are included.
   The Gemini twins were made by hand in the Gemini web app (a new chat per image):
   `python scripts/gemini_s_test.py spec` writes the exact prompt for each twin, `... ingest NAME FILE`
   converts a downloaded image and adds it to the manifest.
3. **Boxes:** drafts with `python scripts/prelabel.py --purpose <batch>`, review with
   `python scripts/review_server.py --purpose <batch>` (local web tool).
4. **Training:** `python scripts/build_train_sets.py`, `python scripts/pack_colab.py`, then
   `colab/train_rq2_colab.ipynb` on a Colab T4 (Ultralytics 8.4.190, 100 epochs, seeds 0-2).
5. **Weights:** download `runs_weights.zip` from the
   [release](../../releases) and unzip it in the repository root; it fills `colab/runs_*/<run>/weights/`.
6. **Evaluation:**
   ```bash
   python scripts/eval_many.py --runs colab/runs_A --runs colab/runs_BC --runs colab/runs_FD
   python scripts/rq1_selfgen.py
   python scripts/plot_article.py
   # second S-test (Gemini), control without the invisible real gliomas, model-selection cost
   python scripts/eval_many.py --runs colab/runs_A --runs colab/runs_BC --runs colab/runs_FD --syn s_test_gemini
   python scripts/rq1_selfgen.py --syn s_test_gemini
   python scripts/rq1_visible_control.py
   python scripts/regret.py
   ```
7. **Synthetic data analysis:** `python scripts/rq3_classifier.py`, `python scripts/batch_table.py`,
   `python scripts/leak_check.py`.

The control run `colab/runs_ctrl/A_full_s0_ctrl` repeats `A_full_s0` in a new Colab session; its training
curves and weights are identical to the original.

## Licenses

- Code (`scripts/`, `colab/`): MIT, see `LICENSE`.
- Synthetic images, boxes, manifest and results: CC BY 4.0.
- Real images in `data/subset_v1/`: CC0, from the Kaggle dataset above (Ahmed Sorour).
- Model weights (`weights/`, release assets) are fine-tuned from Ultralytics YOLO11n and follow the
  Ultralytics license (AGPL-3.0).

## Acknowledgments

Ultralytics YOLO11 for training and evaluation. The OpenAI image generator was called through the Codex
CLI; the Gemini twins were generated in the Gemini web app, operated by Claude in Chrome under the
author's supervision. Anthropic's Claude assisted with analysis code, documentation and figures; all boxes were drawn or
reviewed by the author.
