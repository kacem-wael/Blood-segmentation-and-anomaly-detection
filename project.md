# Blood Cell Detection & Analysis: Project Specification

> **Course:** Mathematical Morphology and Medical Applications
> **Project:** Segment, separate, classify, count and measure blood cells in microscopic smear images, using mathematical morphology as the core and a light ML model for classification.
> **Goal of this file:** one single source of truth describing _what_ we build, _why_, and _every step_ from an empty folder to a deployed, documented, evaluated product.

> **Revision v2 (2026-10-15), changes after design review:**
>
> - Contours are drawn from the **label map** (morphological gradient of labels), not from the binary mask; otherwise watershed split lines are invisible.
> - `fill_holes` replaced by a **shape-filtered hole fill** (only small, round holes = RBC pale centers), so background enclosed between touching cells is not turned into fake cells.
> - **WBCs are segmented before the watershed** and removed from the RBC mask; the watershed then runs on RBCs only.
> - All size parameters are expressed as multiples of the **RBC radius `r`, estimated automatically by granulometry**, so the pipeline works at any magnification (BCCD, Raabin, ALL-IDB, user uploads).
> - **N/C ratio** is defined as nucleus / cytoplasm (hematology convention); WBC cytoplasm is found by a **gradient watershed seeded from the nucleus**.
> - BCCD **under-labelled RBCs** are handled explicitly in training and evaluation; the **official BCCD split** is used; manual masks are spread over train/val **and** test.
> - Raabin-WBC and ALL-IDB now have concrete roles (WBC segmentation Dice; blast-vs-normal stretch task).
> - Missing-marker objects, platelets stuck to RBCs, library API drift, deployment target and team size are addressed.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Scope](#2-scope)
3. [Requirements](#3-requirements)
4. [Tech stack](#4-tech-stack)
5. [Data](#5-data)
6. [System architecture](#6-system-architecture)
7. [Step-by-step roadmap (Phase 0 to Phase 10)](#7-step-by-step-roadmap)
8. [Pipeline specification (technical detail)](#8-pipeline-specification)
9. [Machine Learning specification](#9-machine-learning-specification)
10. [Evaluation protocol](#10-evaluation-protocol)
11. [Web application specification](#11-web-application-specification)
12. [Testing strategy](#12-testing-strategy)
13. [Documentation & report](#13-documentation--report)
14. [Timeline](#14-timeline)
15. [Risks and mitigations](#15-risks-and-mitigations)
16. [Definition of Done](#16-definition-of-done)
17. [CV / portfolio packaging](#17-cv--portfolio-packaging)
18. [References](#18-references)

---

## 1. Project overview

### 1.1 Problem

A blood smear examined under a microscope is used to diagnose anemia, infections, leukemia, etc. Counting and measuring cells by hand is slow, tedious and operator-dependent. Cells also overlap in the image, which makes naive automatic counting inaccurate.

### 1.2 Solution

An automatic pipeline that, given a microscope image:

1. cleans the image and **estimates its scale** (typical RBC size),
2. separates cells from the background,
3. finds WBC nuclei and platelets (stained purple) **first**,
4. **splits touching RBCs** (the hard part),
5. identifies each cell type (RBC, WBC, platelet),
6. counts them and measures shape/size features,
7. flags abnormal cells,
8. returns an annotated image, tables and statistics through a web app.

### 1.3 Cell types

| Cell                                   | Appearance in a stained smear                       | Typical role in analysis                                   |
| -------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| **RBC** (red blood cell / erythrocyte) | Numerous, pink-red, round, pale center              | Count, size distribution (anisocytosis), shape             |
| **WBC** (white blood cell / leukocyte) | Rare, larger, dark purple nucleus, pale cytoplasm   | Count, N/C ratio (nucleus area / cytoplasm area)           |
| **Platelet** (thrombocyte)             | Very small, purple fragments, often stuck to RBCs   | Count                                                      |

### 1.4 Where each course keyword is used

| Keyword                       | Where in the project                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------ |
| Traitement (preprocessing)    | Phase 3: green channel + purple-stain channel, median denoising, illumination correction (black top-hat) |
| Granulometry                  | Phase 3: pattern spectrum of openings gives the RBC radius `r` used to scale every parameter            |
| Segmentation                  | Phase 4: Otsu thresholds (cells / purple stain) + morphological refinement + labelling                 |
| Ouverture (opening)           | Phase 4: remove noise/debris; Phase 3: granulometry                                                    |
| Fermeture (closing)           | Phase 3: background estimate for illumination correction; Phase 4: smooth contours                     |
| Hole filling / reconstruction | Phase 4: shape-filtered filling of RBC pale centers (hole filling = reconstruction of the complement)  |
| Transformation de distance    | Phase 5: find RBC centers, h-maxima markers                                                            |
| Watershed                     | Phase 5: split touching RBCs (distance-based); Phase 6: WBC cytoplasm (gradient-based)                  |
| Gradient morphologique        | Phase 6: gradient image for the WBC watershed; contours of the label map for display                    |
| Squelettisation (bonus)       | Phase 6: SKIZ (skeleton by influence zones) / ultimate erosion for clump analysis                       |
| Machine Learning              | Phase 8: classify cell type, detect anomalies                                                          |

---

## 2. Scope

### 2.1 In scope (must have)

- Segmentation of RBC, WBC, platelets on 2D RGB smear images.
- Automatic scale estimation by granulometry (parameters expressed in units of the RBC radius).
- Splitting of touching RBCs with distance transform + h-maxima + watershed.
- Counting per class, per-cell measurements, CSV export.
- Cell-type classification: rule-based baseline (must) and ML classifier compared to it (must, see §2.4).
- Evaluation against ground truth with quantified metrics.
- Ablation study of each morphological operation.
- Streamlit web application and GitHub repository.
- Written report.

### 2.2 Nice to have (stretch)

- WBC nucleus/cytoplasm segmentation evaluated on **Raabin-WBC** masks.
- Blast vs. normal lymphocyte classification on **ALL-IDB2** (leukemia anomaly bonus).
- Baseline comparison with a deep-learning model (Cellpose or YOLO).
- SKIZ / ultimate-erosion analysis of clumps.
- Anomaly flagging (anisocytosis, abnormal shapes), qualitative.
- Public deployment.
- Batch processing of a folder of images.

### 2.3 Out of scope

- Real clinical diagnosis or medical certification (this is an educational/research tool).
- Fine-grained WBC subtype classification (neutrophil, lymphocyte...), unless time remains.
- Training large deep-learning models from scratch.
- 3D or video processing.

### 2.4 Alignment with the course description (`Projet_Cellules_Sanguines.docx`)

| Topic        | Decision                                                                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| ML           | The docx says "éventuellement"; we keep it as must-have but always compared to a rule-based baseline.      |
| Datasets     | BCCD = main; Raabin-WBC = WBC segmentation evaluation; ALL-IDB = stretch anomaly task.                      |
| Top-hat      | Illumination correction = black top-hat (closing − image) with an SE larger than a WBC.                     |
| N/C ratio    | "Rapport noyau/cytoplasme" = nucleus area / cytoplasm area (cytoplasm = cell − nucleus).                   |
| Team         | 2 people (Wael Kacem, Talel Ben Mbarek); role split in §14.2.                                              |
| Language     | Code and this spec in English; report language to confirm with the instructor.                             |

---

## 3. Requirements

### 3.1 Functional requirements

| ID   | Requirement                                                                                                                                                         |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FR1  | The system loads an RGB image (JPG/PNG) of a blood smear.                                                                                                           |
| FR2  | The system estimates the typical RBC radius `r` (pixels) by granulometry; the user may override it.                                                                 |
| FR3  | The system produces a binary mask of all cells and a mask of purple-stained structures (nuclei, platelets).                                                         |
| FR4  | The system separates touching cells into individual labelled objects.                                                                                               |
| FR5  | The system classifies each object as RBC, WBC or Platelet.                                                                                                          |
| FR6  | The system counts objects per class.                                                                                                                                |
| FR7  | The system computes per-object features: area, perimeter, circularity, eccentricity, solidity, color, purple fraction; for WBC: nucleus area, cytoplasm area, N/C ratio. |
| FR8  | The system draws an annotated overlay (contours of every labelled object + class colors + IDs).                                                                     |
| FR9  | The system exports results to CSV and the overlay to PNG.                                                                                                           |
| FR10 | The system flags abnormal objects (size/shape outliers), excluding clumps and border cells.                                                                         |
| FR11 | The web app lets the user upload an image, tune main parameters and view results.                                                                                   |
| FR12 | An evaluation script reproduces all reported metrics from the command line.                                                                                         |

### 3.2 Non-functional requirements

| ID   | Requirement                                                                              |
| ---- | ---------------------------------------------------------------------------------------- |
| NFR1 | Processing time under about 3 seconds per 640x480 image on a laptop CPU.                 |
| NFR2 | Code is modular, documented (docstrings), and PEP8-compliant.                            |
| NFR3 | Results are reproducible (fixed random seeds, pinned dependencies, keyword arguments for library calls). |
| NFR4 | All parameters live in one config file, not hard-coded; size parameters are in units of `r`. |
| NFR5 | A new user can install and run the project in under 10 minutes using the README.         |

---

## 4. Tech stack

| Purpose                | Tool                                                              |
| ---------------------- | ----------------------------------------------------------------- |
| Language               | Python 3.10+                                                      |
| Image processing       | `scikit-image`, `opencv-python`, `scipy.ndimage`                  |
| Numerical              | `numpy`, `pandas`                                                 |
| ML                     | `scikit-learn` (+ `joblib` for saving models)                     |
| Plots                  | `matplotlib` (optionally `seaborn`)                               |
| Web app                | `streamlit`                                                       |
| Annotation (if needed) | LabelMe or CVAT                                                   |
| Data download          | `git` (BCCD), `kaggle` CLI (Raabin-WBC, ALL-IDB mirror)           |
| Tests                  | `pytest`                                                          |
| Quality                | `black`, `flake8` (optional: `pre-commit`)                        |
| Version control        | Git + GitHub                                                      |
| Deployment             | Streamlit Community Cloud, or Hugging Face Spaces (Docker template) |
| Optional baseline      | `cellpose` or `ultralytics` (YOLO)                                |

> scikit-image has been renaming/deprecating arguments in `morphology` (e.g. size arguments of `remove_small_objects`). Pin the version in `requirements.txt` and always pass keyword arguments.

---

## 5. Data

### 5.1 Datasets

| Dataset                                   | Use                                                   | Notes |
| ----------------------------------------- | ----------------------------------------------------- | ----- |
| **BCCD** (Blood Cell Count and Detection) | **Main dataset.** Counting, detection, classification | ~364 annotated images (640x480), bounding boxes for RBC / WBC / Platelets (Pascal VOC XML in `BCCD/Annotations/`), official split in `BCCD/ImageSets/Main/`. MIT license. GitHub: Shenggan/BCCD_Dataset. **RBCs are under-labelled** (many visible RBCs have no box); usually one WBC per image. |
| **Raabin-WBC**                            | WBC nucleus/cytoplasm segmentation evaluation         | **Kaggle version (downloaded):** 16,633 crops (575×575; TestB 488×488, other microscope) centred on one WBC, 5 classes (neutrophil 65 %, basophil 2 %), split Train / TestA / TestB. **No segmentation masks**: the nucleus/cytoplasm ground-truth subset is only on raabindata.com and must be requested/downloaded by hand for the Dice of the WBC branch (Phase 6). Usable now as a *detection* check: every crop contains exactly one WBC near its centre. |
| **ALL-IDB**                               | Leukemia stretch task                                 | Official: ALL-IDB1 = full smears with blast coordinates; ALL-IDB2 = crops labelled blast / normal. **Kaggle mirror (downloaded) is unreliable:** folders L1/L2/L3 are *not* FAB subtypes but L1 = 50 healthy-named (`_0`) smears 2592×1944, L2 = 50 leukemic-named (`_1`) smears 1712×1368, L3 = 50 single-cell crops 257×257 (blasts only); only **109 of 150 files are distinct** (re-encoded, renamed or rotated copies); no blast coordinates, no normal crops, so the blast-vs-normal task cannot be trained on it. Official source requires a request to Univ. degli Studi di Milano (cite Labati et al., 2011); images **must not be redistributed** (no demo images from it in the app or repo). |

> **Important:** BCCD only has **bounding boxes**, not pixel masks, and its RBC boxes are incomplete. So:
>
> - Counting and detection metrics use the boxes directly; RBC precision is reported with a caveat (§10).
> - For a true **Dice/IoU on segmentation**, we manually annotate **18 images** (LabelMe/CVAT): **10 from train/val** (for tuning) and **8 from test** (for final reporting). On these images **every** cell is annotated, so they also give exact RBC counts and clean labels.

### 5.2 Data rules

- Never commit raw data to Git (add `data/` to `.gitignore`); `scripts/download_data.py` and the README explain how to get it.
- **Use the official BCCD split** (`ImageSets/Main/train.txt`, `val.txt`, `test.txt`) so results are comparable with other work; images without a valid annotation file are excluded and listed.
- **Remove same-field leakage** (found in Phase 2): BCCD contains shifted frames of the same microscope field, 11 such pairs across official splits. The test set is kept unchanged; train/val images sharing a field with a test image are excluded (5), val images sharing a field with train move to train (6). Final split: **209 / 78 / 72**. `field_groups` in `splits.json` is the group key for `GroupKFold`.
- **Images without any RBC box** (15: 8 train, 4 val, 3 test; `no_rbc_labels` in the index) are excluded from RBC counting metrics and RBC training labels. Overall ≈ 30 % of RBCs have no box (crude estimate in notebook 01).
- **Repeated boxes** (same class, IoU ≥ 0.7; 8 boxes) are flagged `duplicate` and ignored.
- **Split by image, never by cell** (prevents leakage).
- **Raabin-WBC:** crops of the same photo share their background and 278 photos (26 of 60 slides) appear in both Train and TestA; use the `photo` (or `slide`) column of `raabin_files.csv` as the group key. All basophils come from 4 slides of a separate session (yellowish background): a classifier can learn the stain instead of the cell.
- **ALL-IDB mirror:** keep one file per duplicate `group` (`keep` column of `all_idb_files.csv`).
- Tune parameters on train/validation only; touch the test set **once**, at the end.
- Keep a `data/splits.json` listing which image is in which split and which images have manual masks.
- Validate annotations when parsing: unknown class names, empty or inverted boxes, boxes outside the image are logged and fixed/skipped.

---

## 6. System architecture

### 6.1 Data flow

```
Input RGB image
      |
      v
[1] Preprocessing      --> green channel (median, black top-hat illumination correction)
                       --> purple-stain channel (Lab -b* or HSV saturation)
      |
      v
[2] Scale estimation   --> granulometry on the cell mask --> RBC radius r
      |
      v
[3] Segmentation       --> Otsu (green) --> cell mask ; Otsu (purple) --> purple mask
      |                    shape-filtered hole fill (pale centers), opening, small-object removal
      v
[4] Purple branch      --> large purple objects = WBC nuclei ; small = platelet candidates
      |
      v
[5] WBC cells          --> gradient watershed seeded from each nucleus --> nucleus + cytoplasm
      |
      v
[6] RBC mask           --> cell mask minus WBC cells minus platelets
      |
      v
[7] RBC separation     --> distance transform --> h-maxima markers --> watershed
      |                    (+ objects without marker kept as their own label)
      v
[8] Label map          --> one integer per object, class known per branch
      |
      v
[9] Feature extraction --> regionprops, normalized by r
      |
      v
[10] Classifier        --> rule baseline / ML (RBC / WBC / Platelet / Artefact) + anomaly flags
      |
      v
[11] Outputs           --> overlay (gradient of label map), CSV, statistics, web app
```

### 6.2 Repository structure

```
blood-cell-analysis/
├── README.md
├── project.md                   <- this file
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.yaml                  <- ALL parameters
├── data/                        <- not versioned
│   ├── raw/
│   │   ├── BCCD_Dataset/
│   │   ├── raabin_wbc/
│   │   └── all_idb/
│   ├── masks_manual/            <- our hand-annotated masks
│   ├── index/                   <- parsed annotations (CSV)
│   └── splits.json
├── src/
│   └── bloodcells/
│       ├── __init__.py
│       ├── config.py            <- load config.yaml
│       ├── io_utils.py          <- load images, parse VOC XML, save outputs
│       ├── datasets.py          <- dataset indexes (BCCD, Raabin, ALL-IDB)
│       ├── preprocessing.py
│       ├── scale.py             <- granulometry, RBC radius estimate
│       ├── segmentation.py      <- thresholds + hole fill + opening
│       ├── separation.py        <- distance transform + h-maxima + watershed
│       ├── wbc.py               <- nucleus + gradient watershed cytoplasm
│       ├── platelets.py
│       ├── features.py          <- regionprops + custom features
│       ├── classifier.py        <- rules + train / predict ML model
│       ├── anomaly.py
│       ├── visualization.py     <- overlays, plots
│       ├── pipeline.py          <- end-to-end function
│       └── evaluate.py          <- metrics + ablation
├── scripts/
│   ├── download_data.py
│   ├── build_index.py           <- parse annotations, dataset stats
│   ├── make_splits.py
│   ├── check_scale.py           <- r vs box sizes + zoom check
│   ├── run_pipeline.py          <- CLI: one image / folder
│   ├── build_features.py
│   ├── train_classifier.py
│   └── run_evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_scale.ipynb
│   ├── 03_pipeline_step_by_step.ipynb
│   └── 04_results_figures.ipynb
├── app/
│   └── streamlit_app.py
├── models/                      <- saved classifier (.joblib, small, versioned)
├── results/
│   ├── figures/
│   ├── tables/
│   └── ablation/
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_scale.py
│   ├── test_segmentation.py
│   ├── test_separation.py
│   └── test_features.py
└── docs/
    └── report/                  <- final report
```

---

## 7. Step-by-step roadmap

> Tick the boxes as you go. Each phase has an **output** (what must exist before moving on).

### Phase 0: Project setup (Day 1)

- [ ] Create the GitHub repository `blood-cell-analysis`, add MIT license (code only; datasets keep their own licenses).
- [x] Clone it locally; create the folder structure from section 6.2.
- [x] Create a virtual environment: `python -m venv .venv` then activate it (`make setup`).
- [x] Create `requirements.txt` (numpy, scipy, scikit-image, opencv-python, pandas, scikit-learn, matplotlib, streamlit, pyyaml, joblib, pytest, jupyter, kaggle); pin versions with `pip freeze` once installed.
- [x] Add `.gitignore` (`data/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `results/` large files). The trained model in `models/` is small and **is** versioned (needed for deployment).
- [x] `pip install -e .` so `import bloodcells` works everywhere.
- [ ] Agree on Git workflow: `main` is always working; each feature on a branch `feat/<name>`; merge through pull requests; small commits with clear messages.
- [ ] Split work between teammates (see section 14).
- [ ] Create the GitHub Project board (To do / Doing / Done) with one card per phase.

**Output:** runnable repository, everyone can `pip install` and import `bloodcells`.

---

### Phase 1: Literature and background (Days 1 to 3)

- [ ] Review the course notes on: erosion, dilation, opening, closing, morphological gradient, top-hat, granulometry, morphological reconstruction (hole filling, h-maxima), distance transform, watershed, skeletonization / SKIZ.
- [ ] Read about blood smear analysis (what RBC/WBC/platelets look like, what anisocytosis, leukocytosis and N/C ratio mean).
- [ ] Read the original watershed paper (Vincent & Soille, 1991) and a tutorial on marker-controlled watershed.
- [ ] Find 3 to 5 related works (classical and deep learning) on blood cell segmentation; note their metrics and which BCCD split they use.
- [x] Write a 1-page summary of the state of the art (it becomes the report's related-work section).

**Output:** `docs/state_of_the_art.md`.

---

### Phase 2: Data acquisition and exploration (Days 2 to 4)

- [x] `scripts/download_data.py`: clone BCCD into `data/raw/BCCD_Dataset/`; download Raabin-WBC and the ALL-IDB mirror with the Kaggle CLI if credentials exist (otherwise print instructions).
- [x] `io_utils.py`: load an image, parse a VOC XML into a list of `(class, xmin, ymin, xmax, ymax)` with validation (§5.2).
- [x] `scripts/build_index.py`: one CSV of all BCCD boxes (`data/index/bccd_boxes.csv`) + one row per image (`bccd_images.csv`), duplicate-box flags, same-field pairs and field groups, (Raabin / ALL-IDB: `scripts/audit_extra_datasets.py`).
- [x] Notebook `01_data_exploration.ipynb`:
  - [x] show 10 random images with their boxes,
  - [x] count boxes per class over the whole dataset and per split,
  - [x] plot the distribution of box sizes per class (in pixels **and** relative to the median RBC box),
  - [x] check color variation (staining, brightness) across images,
  - [x] choose the purple-stain channel (Fisher ratio: −b* best),
  - [x] estimate how many RBCs are missing labels (automatic estimate: ≈ 30 %; 15 images with none),
  - [x] note difficult cases (heavy overlap, cells on the border, platelets stuck to RBCs),
  - [x] detect shifted frames of the same field across splits (leakage).
- [x] `scripts/make_splits.py`: read the official split, remove same-field leakage, save `data/splits.json`.
- [x] Download Raabin-WBC and ALL-IDB (Kaggle token); audit them with `scripts/audit_extra_datasets.py` (`make audit-extra` → `data/index/raabin_files.csv`, `all_idb_files.csv`).
- [ ] Choose **18 images** (mixed difficulty): 10 from train/val, 8 from test; annotate **every** cell (LabelMe/CVAT), one polygon per cell with its class, nucleus polygon for WBCs. Save in `data/masks_manual/`; record the list in `splits.json`.
- [x] Measure typical cell sizes (in pixels) from the box statistics to validate the granulometry estimate later: RBC 104 px (r = 52 px), WBC 194 px (1.87 r), platelet 39.5 px (0.38 r); see `results/tables/typical_sizes.csv`.

**Output:** download script, annotation index, exploration notebook, `splits.json`, manual masks, a table of "typical sizes per class".

---

### Phase 3: Preprocessing and scale (Days 4 to 5)

- [x] Implement `preprocessing.py`:
  - [x] `green_channel(img)`: **green channel** (hemoglobin absorbs green: best RBC contrast).
  - [x] `purple_channel(img)`: stain channel for nuclei and platelets: `-b*` of Lab (best Fisher ratio in Phase 2, 7.6) or HSV saturation.
  - [x] `denoise(gray)`: median filter (disk radius 2) to remove salt-and-pepper noise while keeping edges.
  - [x] `black_tophat(gray, radius)`: cells are darker than the background, so estimate the background with a **closing** by a disk **larger than a WBC** (radius ≈ 2.5 r, computed at 1/4 resolution with edge padding) and compute the black top-hat (`closing − image`); cells become bright on a flat dark background.
  - [x] **Relative top-hat** `(closing − image) / closing` (`tophat_relative: true`): transmitted light is illumination × transmittance, so vignetting also lowers the cell contrast. Under synthetic vignetting the plain top-hat collapsed on 1 of 10 images (Dice 0.34); the relative one stays ≥ 0.90.
  - [x] `normalize(gray)`: percentile contrast stretch to [0, 1].
- [x] Implement `scale.py`:
  - [x] `estimate_rbc_radius(mask)`: **granulometry**: areas of openings by disks of radius 1..R (each opening = two Euclidean distance transforms, so no downsampling is needed); the pattern spectrum (area removed between consecutive radii) peaks at the dominant RBC radius.
  - [x] The raw spectrum is dominated by WBCs and clumps (35 % of val images within 15 %): weight it by 1/ρ and smooth it in log-radius (σ = 0.08, scale-free) → 90 %.
  - [x] The peak is the **inscribed** radius; `scale.calibration` = 1.24 (median box radius / inscribed radius, fitted on train) converts it to the box radius used as the unit of `config.yaml`.
  - [x] Bootstrapping: a coarse Otsu mask with a plain `fill_holes` is good enough to estimate `r`; then illumination correction and the final segmentation use `r`.
  - [x] Check `r` against the BCCD RBC box sizes (`scripts/check_scale.py`, `make scale-check`): median error 3.7 % on val, 90 % of images within 15 %. Since BCCD has one magnification (a constant guess also scores well), a **zoom check** was added: rescaling by 0.5–2 multiplies `r` by the same factor (median ratio 0.997–1.007).
- [x] Visual check on 10 images: each step saved as a figure; pattern spectrum plot (`results/figures/preprocessing/`).
- [x] Compare with and without illumination correction on 5 images, plus synthetic vignetting (notebook 02, section 3).
- [x] Unit tests: `tests/test_preprocessing.py`, `tests/test_scale.py`.

Known failures (5 / 74 val images off by > 30 %): RBC rings whose pale centre touches the background (under-estimate), large clumps or a WBC dominating a sparse field (over-estimate). Re-estimating `r` on the Phase 4 mask is an option if they hurt segmentation.

**Output:** `preprocess(img, cfg, r) -> channels`, `estimate_scale(img, cfg) -> r`, before/after figure, pattern-spectrum figure, `results/tables/scale_check.csv`.

---

### Phase 4: Segmentation and morphological cleaning (Days 5 to 8)

- [x] Implement `segmentation.py`:
  - [x] `threshold_cells(gray)`: Otsu threshold on the corrected green channel → cell mask (RBC + WBC + platelets).
  - [x] `threshold_purple(p)`: Otsu (or multi-Otsu) on the purple channel → nuclei + platelets mask.
  - [x] `clean_mask(mask, r)`, **in this order**:
    - [x] **Shape-filtered hole fill** first (before opening, so ring-shaped RBCs are not broken): fill a hole only if it is **small** (area ≤ 0.5·π r²) **and round** (circularity ≥ 0.6). RBC pale centers are round; the gaps between touching cells have concave sides and very low circularity (~0.2 for three touching discs), so they stay background.
    - [x] **Opening** with a disk (radius ≈ 0.1 r, at least 1): remove small noise and thin bridges.
    - [x] **Closing** with a small disk (radius ≈ 0.05 r): smooth the contour.
    - [x] `remove_small_objects` with `min_size ≈ 0.02·π r²` (keyword argument, see §4); platelets come from the purple mask, not from here.
    - [x] (border cells are kept and flagged `touches_border`; excluded from statistics and anomalies) Optionally `clear_border` for statistics (but keep a version with border cells for display).
- [x] Visual check: overlay the mask contours on the original for 20 images.
- [x] Record observations: where does Otsu fail (low contrast, bright images)? If necessary, test an adaptive/local threshold as an alternative.
- [x] (done on box proxies and detection F1; Dice sweep needs the manual masks) Parameter sweep on train/val images: structuring element radius (in units of r) vs. Dice on the 10 train/val manual masks.
- [x] Unit tests (`test_segmentation.py`): synthetic disk on a bright background gives one object of the expected area; a ring (pale center) is filled; the triangular gap between three touching discs is **not** filled.

**Output:** `segment(img, r) -> cell_mask, purple_mask`, a figure of masks over images, first parameter values in `config.yaml`.

---

### Phase 5: WBC first, then splitting touching RBCs (Days 8 to 11)

This is the **core** of the project.

- [x] Remove non-RBC objects before separating: `rbc_mask = cell_mask & ~wbc_cells & ~platelets` (WBC cells from Phase 6 functions, which are called first in the pipeline; platelets dilated by 1–2 px so a platelet stuck to an RBC does not leave a notch that creates a false marker).
- [x] Implement `separation.py`:
  - [x] **Distance transform** of `rbc_mask` (`scipy.ndimage.distance_transform_edt`).
  - [x] Smooth it slightly (Gaussian sigma ≈ 0.05 r) to avoid many tiny peaks.
  - [x] Find **markers**: h-maxima (`skimage.morphology.h_maxima`, h ≈ 0.15 to 0.3 r) or `peak_local_max` (min_distance ≈ 0.6 to 0.8 r), then label them.
  - [x] **Watershed** on the inverted distance map: `skimage.segmentation.watershed(-dist, markers, mask=rbc_mask)`.
  - [x] **Missing markers:** watershed leaves label 0 on connected components that received no marker; add those components back as their own labels (and count them in a debug log).
  - [x] Post-process: remove fragments smaller than ≈ 0.1·π r²; merge them into the neighbour sharing the longest border.
- [x] Test on the hardest images (large clumps): visualise markers on top of the mask.
- [x] Study **over-segmentation** (one cell split in two: h too small) vs **under-segmentation** (two cells merged: h too large). Pick values on the validation set by counting error (on the manual-mask images, where RBC counts are exact).
- [x] Alternative marker strategy to test: **ultimate erosion** (regional maxima of the distance function / residues of successive erosions), and compare.
- [x] Tests: two synthetic discs of radius r with centers 1.5 r apart give exactly 2 labels; one disc gives 1; a tiny disc below `h` is still labelled (missing-marker rule).
- [x] Save before/after-watershed figures (very strong visual for the report).

**Output:** `separate(rbc_mask, r) -> label_map`, tuned `h` / `min_distance` (in units of r), comparison figure with and without watershed.

---

### Phase 6: WBC and platelet branches, contours (Days 11 to 14)

> In the pipeline these functions run **before** Phase 5 separation (see §6.1); they are a separate phase only for planning.

- [x] **WBC branch** (`wbc.py`):
  - [x] **Nucleus**: connected components of the purple mask with area ≥ ≈ 0.3·π r² (smaller ones are platelet candidates). Clean with opening (remove specks) then closing (join lobes of segmented nuclei; neutrophil lobes may need a closing of ≈ 0.15 r to be grouped as one nucleus).
  - [x] **Whole cell (nucleus + cytoplasm)** by **gradient watershed**: morphological gradient of the green channel (`dilation − erosion`, disk 1); markers = each nucleus + one background marker (outside the cell mask or farther than ≈ 2 r from the nucleus); `watershed(gradient, markers)`. The cytoplasm is the cell region minus the nucleus.
  - [x] **N/C ratio** = nucleus area / cytoplasm area. If the cytoplasm is almost empty (pale cytoplasm lost), flag `cytoplasm_uncertain` instead of reporting a huge ratio.
  - [x] (detection part done: `scripts/eval_raabin.py`; Dice still needs the masks) Evaluate nucleus and cell Dice on the Raabin-WBC masks (stretch, §2.2; masks not in the Kaggle download, see §5.1) and on the manual masks. Without masks: WBC detection rate on Raabin crops (one WBC per crop).
- [x] **Platelet branch** (`platelets.py`):
  - [x] Platelets = small purple components with area in `[a_min, a_max]` ≈ `[0.02, 0.3]·π r²`, not inside a WBC cell.
  - [x] Platelets stuck to an RBC: they are taken from the purple mask, so they are detected even when the cell mask merges them with the RBC; they are then subtracted from the RBC mask (Phase 5).
- [x] **Contours** (`visualization.py`): **morphological gradient of the label map**, `dilation(labels) != erosion(labels)` (disk 1). Unlike the gradient of the binary mask, this also draws the watershed lines between touching cells.
- [x] **Bonus: SKIZ / ultimate erosion** of large unsplit clumps: the SKIZ of the ultimate-erosion markers gives the zones of influence (it is the morphological basis of the watershed); the number of ultimate-erosion components estimates the number of cells in a clump.
- [x] Resolve conflicts: an object must not be both WBC and platelet; precedence: WBC cell, then platelet, then RBC.

**Output:** nucleus and WBC cell masks, N/C ratios, platelet detections, contour images, list of objects per branch.

---

### Phase 7: Feature extraction (Days 14 to 15)

- [x] Implement `features.py` with `skimage.measure.regionprops` plus extras.
- [x] For **each** object, compute:

| Feature                  | Description                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `area`, `area_rel`       | Pixel count; and `area / (π r²)` (scale-free, used by rules and ML)                 |
| `perimeter`              | Contour length (`perimeter_crofton`, less biased on a pixel grid)                   |
| `circularity`            | `4*pi*area / perimeter^2`, clipped to [0, 1] (1 = perfect circle)                   |
| `eccentricity`           | Elongation (0 = circle, close to 1 = line)                                          |
| `solidity`               | `area / convex_area` (low = irregular or merged cells)                              |
| `equivalent_diameter_rel`| Diameter of the equal-area circle divided by `2r`                                    |
| `L, a, b` (mean, std)    | Mean/std color in Lab (more robust to stain variation than raw RGB)                 |
| `purple_fraction`        | Fraction of the object's pixels in the purple mask (defined for **every** object)   |
| `nucleus_area`, `cytoplasm_area`, `nc_ratio` | WBC branch only (NaN otherwise); **not** used as ML inputs      |
| `branch`                 | Which pipeline branch produced the object (rbc / wbc / platelet)                    |
| `touches_border`         | Boolean                                                                             |

- [x] Build a DataFrame; save as CSV.
- [x] **Label assignment for training:** match each detected object to a ground-truth box (IoU ≥ 0.5 with one-to-one Hungarian matching; fallback: centroid inside box).
  - On **manual-mask images**: unmatched objects are true `Artefact`.
  - On **other BCCD images**: unmatched objects are `Unlabelled` (BCCD misses many RBCs) and are **excluded** from training, not labelled as artefact.
- [x] Script `build_features.py` that runs the pipeline on all train/val/test images and writes `features_{split}.csv`.

**Output:** feature tables with ground-truth labels.

---

### Phase 8: Machine Learning (Days 15 to 18)

See [section 9](#9-machine-learning-specification) for details.

- [x] Build a **rule-based baseline** first (e.g., `purple_fraction` high and `area_rel` ≥ 1.2 = WBC; `purple_fraction` high and `area_rel` < 0.3 = platelet; else RBC) and measure its accuracy.
- [x] Train a **Random Forest** (and compare with SVM and logistic regression) on scale-free features only.
- [x] Handle class imbalance (RBC >> WBC, platelets).
- [x] Evaluate with cross-validation grouped by image; then on the validation set.
- [x] Analyze feature importance and errors (confusion matrix, example of misclassified cells).
- [x] Save the best model to `models/classifier.joblib`.
- [x] Implement `anomaly.py`: flag abnormal objects (§9.3).

**Output:** trained model, confusion matrix, feature importance plot, rule-vs-ML comparison table.

---

### Phase 9: Integration, evaluation and ablation (Days 18 to 22)

- [x] Implement `pipeline.py`: `analyze(image, cfg, r=None) -> {r, labels, classes, features_df, overlay, counts}`.
- [x] Implement `visualization.py`: overlay with colored contours from the label-map gradient (e.g., RBC red, WBC blue, platelet green), cell IDs optional, legend.
- [x] Implement `run_pipeline.py` (CLI) for one image or a whole folder.
- [x] Implement `evaluate.py` and `run_evaluation.py` (see [section 10](#10-evaluation-protocol)):
  - [x] counting error, detection precision/recall/F1, Dice/IoU on manual masks, classification metrics.
- [x] Run the **ablation study** and fill the ablation table.
- [x] Run the **baseline comparison** (simple Otsu without morphology; optional Cellpose/YOLO).
- [x] **Only now** evaluate once on the test set; save all tables to `results/tables/`.
- [x] Error analysis: collect 10 failure cases, explain each cause.

**Output:** reproducible evaluation script, result tables, ablation table, failure-case gallery.

---

### Phase 10: Web app, documentation and release (Days 22 to 28)

> **Scope change (2026-11-04):** the web app is dropped (decision of 2026-11-04); the CLI `scripts/run_pipeline.py` and the step images `scripts/export_steps.py` replace it. The remaining items (README, cleanup, report, slides, release) stay.

- [x] ~~Build the Streamlit app (see [section 11](#11-web-application-specification)).~~ (dropped)
- [x] ~~Test the app manually on 10 different images (including one at a different magnification) and a wrong-format file.~~ (dropped)
- [x] ~~Deploy (Streamlit Community Cloud, or Hugging Face Spaces with the Docker template); include the model file and a few **BCCD** demo images (MIT license; no ALL-IDB images).~~ (dropped)
- [x] Write a polished `README.md`: description, GIF/screenshots, install, usage, results table, dataset links, limitations, license.
- [x] Clean the repo: remove dead code, run `black` and `flake8`, check that all tests pass, notebooks run top to bottom.
- [ ] Write the final report (see [section 13](#13-documentation--report)).
- [ ] Prepare the presentation/demo (5 to 10 slides + live demo with a backup recorded video).
- [ ] Tag a release `v1.0` on GitHub.

**Output:** deployed app, finished repository, report, slides.

---

## 8. Pipeline specification

### 8.1 Default parameters (starting values, to be tuned in `config.yaml`)

> All size parameters are in units of the RBC radius **`r`** (pixels), estimated by granulometry (Phase 3) or overridden by the user. Areas are in units of `π r²`.

| Stage         | Parameter                       | Start value               | Tune on      |
| ------------- | ------------------------------- | ------------------------- | ------------ |
| Scale         | granulometry max radius         | image short side × 0.25   | vs. box sizes |
| Scale         | spectrum weighting / smoothing  | 1/ρ, σ = 0.08 in log ρ    | vs. box sizes |
| Scale         | calibration (box / inscribed r) | 1.24 (fitted on train)    | vs. box sizes |
| Preprocessing | median filter radius            | 2 px                      | visual       |
| Preprocessing | top-hat (background) SE radius  | 2.5 r (larger than a WBC) | visual       |
| Preprocessing | top-hat relative to background  | yes (flat-field)          | vignetting Dice |
| Preprocessing | purple channel                  | `-b*` (Lab) or saturation | nucleus Dice |
| Segmentation  | threshold method                | Otsu                      | Dice         |
| Cleaning      | hole fill: max area / min circ. | 0.5 π r² / 0.6            | Dice         |
| Cleaning      | opening SE (disk)               | 0.1 r (≥ 1 px)            | Dice         |
| Cleaning      | closing SE (disk)               | 0.05 r (≥ 1 px)           | Dice         |
| Cleaning      | min object area                 | 0.02 π r²                 | count error  |
| WBC           | nucleus min area                | 0.3 π r²                  | detection F1 |
| WBC           | cytoplasm search distance       | 2 r from the nucleus      | cell Dice    |
| Platelets     | area range `[a_min, a_max]`     | [0.02, 0.3] π r²          | detection F1 |
| Separation    | distance smoothing sigma        | 0.05 r                    | count error  |
| Separation    | h-maxima `h`                    | 0.15 to 0.3 r             | count error  |
| Separation    | alt: `min_distance`             | 0.6 to 0.8 r              | count error  |
| Separation    | min fragment area               | 0.1 π r²                  | count error  |
| Contours      | gradient SE (disk)              | 1 px                      | visual       |

### 8.2 Reference code skeleton

```python
# pipeline sketch: bloodcells/pipeline.py
import numpy as np
from scipy import ndimage as ndi
from skimage import color, filters, measure, morphology, segmentation, transform


def preprocess(img, cfg, r):
    g = filters.median(img[..., 1], morphology.disk(cfg["median_r"]))
    background = morphology.closing(g, morphology.disk(int(cfg["tophat_r"] * r)))
    green = background.astype(float) - g          # black top-hat: cells bright, background ~0
    purple = -color.rgb2lab(img)[..., 2]          # purple stain = negative b*
    return green, purple


def estimate_rbc_radius(mask, r_max, step=2):
    """Granulometry: radius at which the pattern spectrum peaks."""
    small = transform.rescale(mask.astype(float), 1 / step, order=0) > 0.5
    areas = [morphology.binary_opening(small, morphology.disk(k)).sum() for k in range(r_max // step)]
    spectrum = -np.diff(areas)                     # area removed between radius k and k+1
    return (int(np.argmax(spectrum)) + 0.5) * step


def fill_round_holes(mask, r, max_area=0.5, min_circ=0.6):
    """Fill RBC pale centers (small, round holes), not gaps between touching cells."""
    holes = measure.label(ndi.binary_fill_holes(mask) & ~mask)
    out = mask.copy()
    for p in measure.regionprops(holes):
        circ = 4 * np.pi * p.area / max(p.perimeter, 1) ** 2
        if p.area <= max_area * np.pi * r**2 and circ >= min_circ:
            out[tuple(p.coords.T)] = True
    return out


def wbc_cells(green, nuclei, cell_mask, r, cfg):
    """Gradient watershed seeded from each nucleus -> whole WBC (nucleus + cytoplasm)."""
    if nuclei.max() == 0:
        return np.zeros_like(nuclei)
    se = morphology.disk(1)
    grad = morphology.dilation(green, se) - morphology.erosion(green, se)
    near = morphology.binary_dilation(nuclei > 0, morphology.disk(int(cfg["wbc_search"] * r)))
    bg = nuclei.max() + 1
    markers = nuclei.copy()
    markers[~cell_mask | ~near] = bg
    cells = segmentation.watershed(grad, markers)
    cells[cells == bg] = 0
    return cells


def separate(rbc_mask, r, cfg):
    dist = ndi.distance_transform_edt(rbc_mask)
    dist = filters.gaussian(dist, sigma=cfg["dist_sigma"] * r)
    markers = measure.label(morphology.h_maxima(dist, cfg["h"] * r))
    labels = segmentation.watershed(-dist, markers, mask=rbc_mask)
    orphans = measure.label(rbc_mask & (labels == 0))   # components with no marker
    labels[orphans > 0] = orphans[orphans > 0] + labels.max()
    return labels


def contours(labels, radius=1):
    """Morphological gradient of the label map: also draws lines between touching cells."""
    se = morphology.disk(radius)
    return morphology.dilation(labels, se) != morphology.erosion(labels, se)
```

### 8.3 Edge cases to handle explicitly

- Cells **touching the image border**: kept for display, excluded from size statistics and anomaly flags.
- **Very large clumps** that watershed cannot split (low solidity, area > 3 π r²): flag as "clump" and estimate cell count by `clump_area / median_RBC_area`.
- **Objects without a marker**: kept as their own label (see `separate`), never silently dropped.
- **Platelets stuck to RBCs**: detected from the purple mask, then subtracted (dilated) from the RBC mask before the distance transform.
- **Segmented (multi-lobed) nuclei**: closing of the nucleus mask so one WBC = one nucleus object.
- **Granulometry failure** (flat spectrum, very few cells): fall back to the config default `r` and warn.
- **Empty or blank image**: return zero counts, no crash.
- **Very bright / very dark images**: the Otsu threshold is unstable; add a safety check on the histogram.
- **Non-RGB, RGBA or tiny images**: convert or reject with a clear error.

---

## 9. Machine Learning specification

### 9.1 Purpose

Morphology produces objects; ML decides **what each object is** and **whether it is abnormal**. ML never replaces the morphological segmentation.

### 9.2 Task A: cell-type classification

- **Input:** scale-free feature vector per object (Phase 7): `area_rel`, `circularity`, `eccentricity`, `solidity`, `equivalent_diameter_rel`, Lab mean/std, `purple_fraction`. Branch-specific outputs (`nucleus_area`, `nc_ratio`, `branch`) are **not** inputs, otherwise the model just copies the rules.
- **Output:** `RBC`, `WBC`, `Platelet`, `Artefact`.
- **Training labels:** matched BCCD boxes; `Unlabelled` objects excluded; `Artefact` examples only from manual-mask images (§7, Phase 7).
- **Models to compare:** rule-based baseline, Logistic Regression, SVM (RBF), Random Forest (final candidate).
- **Imbalance:** RBC greatly outnumbers the others. Use `class_weight="balanced"` and report per-class metrics (not only accuracy).
- **Validation:** `GroupKFold` with image ID as group (no image shared between train and validation folds).
- **Metrics:** per-class precision, recall, F1; macro-F1; confusion matrix; bootstrap 95% confidence intervals (the test set has only about one WBC per image).
- **Explainability:** feature importance plot; 10 misclassified examples with explanations.
- **Output artefact:** `models/classifier.joblib` plus the list of feature names used.

### 9.3 Task B (stretch): anomaly flagging

- **Pre-filter first:** exclude border cells, clumps, low-solidity objects and very small fragments; otherwise segmentation errors are flagged as "anomalies".
- **Simple version:** robust z-score (median / MAD) on RBC `area_rel` and circularity; flag objects beyond 3.
- **Statistic to report:** coefficient of variation of RBC area (a simple anisocytosis indicator, analogous to the RDW).
- **Evaluation:** no dataset provides RBC anomaly labels, so this part is **qualitative** (gallery of flagged cells, inspected by hand).

### 9.4 Task C (stretch): blast detection on ALL-IDB2

- Run the WBC branch on ALL-IDB2 crops; features: nucleus `area_rel`, N/C ratio, nucleus circularity/solidity, chromatin texture (std of L*).
- Classify blast vs. normal (Random Forest, grouped cross-validation); report accuracy and F1. This is the only quantitatively evaluated "anomaly" task.

### 9.5 What ML is **not** used for

- Segmentation, separation of touching cells and contours are purely morphological. This keeps the project aligned with the course and the results interpretable.

---

## 10. Evaluation protocol

### 10.1 Metrics

| Aspect         | Metric                                                                                                                                                                       | Ground truth                       |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| Counting       | MAE per class and per image; relative error. **Exact RBC counts** only on manual-mask images; on other BCCD images RBC "error" is reported against an incomplete reference. | Manual masks (RBC); BCCD boxes (WBC, platelets) |
| Detection      | Precision, Recall, F1 with one-to-one matching at **IoU ≥ 0.5** and at **IoU ≥ 0.3** (BCCD boxes are loose; split cells have smaller boxes). RBC precision on non-manual images is a lower bound (missing labels). | BCCD boxes; manual masks |
| Segmentation   | Dice and IoU per cell / per image                                                                                                                                            | Manual masks (8 test images)       |
| WBC branch     | Nucleus and cell Dice                                                                                                                                                        | Manual masks; Raabin-WBC (stretch) |
| Scale          | Relative error of granulometry `r` vs. median RBC box radius                                                                                                                 | BCCD boxes                         |
| Classification | Per-class precision/recall/F1, macro-F1, confusion matrix, bootstrap 95% CI                                                                                                  | BCCD labels                        |
| Performance    | Seconds per image                                                                                                                                                            | n/a                                |

### 10.2 Ablation study

Run the **same evaluation** for each configuration and fill a table:

| Config       | Preprocessing | Hole fill (shape) | Opening/Closing | WBC first | Distance + Watershed | Count MAE (RBC) | Det. F1 | Dice |
| ------------ | :-----------: | :---------------: | :-------------: | :-------: | :------------------: | --------------- | ------- | ---- |
| A (baseline) |       n       |         n         |        n        |     n     |          n           |                 |         |      |
| B            |       y       |         n         |        n        |     n     |          n           |                 |         |      |
| C            |       y       |         y         |        n        |     n     |          n           |                 |         |      |
| D            |       y       |         y         |        y        |     n     |          n           |                 |         |      |
| E            |       y       |         y         |        y        |     n     |          y           |                 |         |      |
| F (full)     |       y       |         y         |        y        |     y     |          y           |                 |         |      |

Extra comparisons: plain `fill_holes` vs. shape-filtered fill; fixed-pixel parameters vs. `r`-scaled parameters (on images rescaled ×0.5 and ×2). Also vary the SE size (0.05 r to 0.3 r) and `h` (0.05 r to 0.5 r) and plot the metric curves.

### 10.3 Baselines to compare against

1. Otsu threshold + connected components (no morphology, no watershed).
2. Otsu + opening/closing, but **no** watershed.
3. _(optional)_ Cellpose / YOLO pretrained model.

### 10.4 Rules

- Choose all parameters on train/validation (including the 10 train/val manual-mask images).
- Final test-set evaluation done **once**, results saved, never re-tuned afterwards.
- Everything reproducible by `python scripts/run_evaluation.py --split test`.

---

## 11. Web application specification

### 11.1 Screens and features

1. **Upload area:** accepts JPG/PNG; also a "use demo image" dropdown (BCCD images only).
2. **Sidebar parameters:** RBC radius `r` (auto-estimated, shown, with manual override), opening size, h (separation sensitivity), min area, all in units of `r`; "Reset to default" button.
3. **Result view (tabs):**
   - _Overlay:_ annotated image (class-colored contours from the label map), with a toggle for IDs.
   - _Steps:_ intermediate images (channels, pattern spectrum, mask, purple mask, distance map, markers, watershed labels) for teaching/debugging.
   - _Statistics:_ counts per class (metric cards), RBC size histogram, circularity histogram, WBC N/C ratios.
   - _Table:_ per-cell features with sortable columns; flagged anomalies highlighted.
4. **Downloads:** annotated PNG, results CSV.
5. **Footer:** clear disclaimer: _"Educational prototype, not a medical device."_

### 11.2 UX rules

- Show a spinner during processing; cache the model loading (`st.cache_resource`) and pipeline results per (image, parameters) (`st.cache_data`).
- Friendly error messages (wrong file, empty result, scale estimation failed).
- No state loss when changing a slider.

---

## 12. Testing strategy

| Level         | What                                                                                                                                                                                                                                                                                   | Tool      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| Unit (I/O)    | VOC parser on a small XML fixture: classes, coordinates, invalid boxes rejected; split loader                                                                                                                                                                                          | `pytest`  |
| Unit          | Synthetic images (drawn discs of radius r): 1 disc gives 1 object; 2 discs with centers 1.5 r apart give 2 objects after watershed; empty image gives 0 objects; a disc below the area threshold is removed; a ring is filled; the gap between 3 touching discs is not filled; a tiny unmarked component keeps a label | `pytest`  |
| Unit          | `contours(labels)` draws a line between two touching labels                                                                                                                                                                                                                            | `pytest`  |
| Unit          | Granulometry on synthetic discs of radius 20 returns `r` within ±10%                                                                                                                                                                                                                   | `pytest`  |
| Feature tests | Disc of radius 30: circularity in [0.95, 1.0]; known area equals pixel count                                                                                                                                                                                                           | `pytest`  |
| Integration   | `analyze()` on 3 sample images returns the expected keys and non-negative counts                                                                                                                                                                                                      | `pytest`  |
| Regression    | Counts on 5 fixed images stay within tolerance after code changes                                                                                                                                                                                                                      | `pytest`  |
| Manual        | App tested on 10 images and one invalid file                                                                                                                                                                                                                                           | checklist |

- [x] Add a GitHub Actions workflow running `pytest` on every push (optional but valuable).

---

## 13. Documentation & report

### 13.1 README.md must contain

Project summary, demo GIF, results table, install instructions, data download, quick start (CLI and app), dataset links and licenses, project structure, limitations, license, authors.

### 13.2 Final report outline (about 10 to 15 pages)

1. **Abstract**
2. **Introduction:** medical context, problem, objectives
3. **Theory:** erosion, dilation, opening, closing, top-hat, gradient, granulometry, morphological reconstruction (hole filling, h-maxima), distance transform, watershed, SKIZ (with formulas and small illustrations)
4. **Related work**
5. **Data:** datasets, label quality (missing RBC boxes), annotation protocol, splits
6. **Method:** pipeline stage by stage, with figures and parameters (in units of r)
7. **ML component:** features, models, validation
8. **Experiments and results:** metrics, ablation, baselines, scale robustness, runtime
9. **Discussion:** what worked, failure cases, limitations (staining variability, overlap, pale cytoplasm, dataset size and label quality)
10. **Conclusion and future work:** e.g. 3D, other stains, deep-learning hybrid, mobile deployment
11. **References**
12. **Appendix:** full parameter table, extra figures

### 13.3 Figures to produce (they make the report strong)

- Pipeline diagram.
- Before/after for each preprocessing step; pattern spectrum with the detected `r`.
- Plain vs. shape-filtered hole filling on a dense area.
- With vs. without watershed on a dense clump (label-map contours).
- Marker overlay (h-maxima of the distance function).
- WBC gradient watershed: nucleus, cytoplasm, N/C.
- Ablation curves (metric vs. structuring element size, in units of r).
- Confusion matrix and feature importance.
- Failure-case gallery.

---

## 14. Timeline

### 14.1 Overview (4 weeks)

| Week  | Phases                 | Milestone                                                                          |
| ----- | ---------------------- | ---------------------------------------------------------------------------------- |
| **1** | 0, 1, 2, 3, start of 4 | Repo ready, data downloaded and indexed, manual masks started, preprocessing + scale |
| **2** | 4, 5, 6                | Full morphological pipeline working: segmentation, WBC branch, watershed, platelets |
| **3** | 7, 8, 9                | Features, ML, evaluation, ablation, baselines                                      |
| **4** | 10                     | App deployed, README, report, slides, release                                      |

> If time is short, protect this order of priority: **(1) segmentation + watershed + counting**, **(2) evaluation + ablation**, **(3) classification (rules, then ML)**, **(4) app**, **(5) stretch goals**.

### 14.2 Role split (team of 2)

| Person    | Responsibilities                                                                                          |
| --------- | --------------------------------------------------------------------------------------------------------- |
| Member A  | Image processing: Phases 3 to 6 (preprocessing, scale, segmentation, WBC branch, watershed), contours     |
| Member B  | Data, evaluation, ML, app: Phases 2, 7, 8, 9, 10 (data, indexes, features, classifier, metrics, Streamlit) |
| Both      | Phase 0–1, manual annotation (9 images each), ablation runs, report writing, review of each other's PRs   |

---

## 15. Risks and mitigations

| Risk                                       | Impact                     | Mitigation                                                                            |
| ------------------------------------------ | -------------------------- | ------------------------------------------------------------------------------------- |
| Heavy cell overlap breaks watershed        | Wrong counts               | Tune `h`, try ultimate-erosion markers, flag clumps and estimate by area              |
| Different magnification between images     | Fixed parameters fail      | Granulometry-based `r`, all parameters in units of `r`, manual override in the app    |
| Staining/lighting varies between images    | Otsu fails                 | Top-hat illumination correction, Lab color, adaptive threshold fallback               |
| Gaps between touching cells filled as holes| Fake cells, wrong counts   | Shape-filtered hole fill (small and round only)                                        |
| Pale WBC cytoplasm lost                    | Wrong N/C ratio            | Gradient watershed from the nucleus; `cytoplasm_uncertain` flag                       |
| Platelets removed or merged with RBCs      | Missed platelets           | Platelets from the purple mask, subtracted from the RBC mask                          |
| BCCD RBCs under-labelled                   | Misleading precision, noisy training | Exclude unmatched objects from training; exact counts from manual masks; caveats in report |
| Few WBC samples                            | Weak classifier            | `class_weight`, rule-based fallback, per-class metrics with confidence intervals      |
| BCCD has no masks                          | No Dice                    | 18 manual masks (train/val + test); Raabin-WBC for WBC masks                          |
| Over-tuning on test set                    | Misleading results         | Strict official split, test set used once                                             |
| Same field in several splits (shifted frames) | Inflated test scores    | Registration-based detection; leaking train/val images excluded; `GroupKFold` on field groups |
| Library API changes (scikit-image)         | Broken code                | Pinned versions, keyword arguments, CI tests                                          |
| Kaggle downloads need credentials          | Optional data missing      | BCCD (no credentials) covers all must-have work; instructions printed by the script   |
| Time overrun                               | Incomplete deliverable     | Priority order in section 14.1; weekly check-ins                                      |
| Deployment issues                          | No live demo               | Record a demo video as backup; keep the app runnable locally                          |

---

## 16. Definition of Done

The project is **finished** when all boxes are ticked:

- [x] Pipeline runs end-to-end on any BCCD image and on a user-uploaded image at another magnification.
- [x] Cells are segmented, touching cells are split, classes are assigned, counts and features exported.
- [x] All course keywords are used and explained: opening, closing, top-hat, granulometry, reconstruction, gradient, distance transform, watershed, segmentation, preprocessing (and SKIZ / skeleton as bonus).
- [ ] Evaluation done: counting MAE, detection F1, Dice, classification macro-F1, on a held-out test set. (done except Dice: needs the 18 manual masks)
- [x] Ablation table and baseline comparison completed and discussed.
- [x] Unit tests pass; code is formatted; config file holds all parameters.
- [x] ~~Streamlit app works locally and is deployed publicly.~~ (dropped 2026-11-04)
- [ ] README is clear enough for a stranger to run the project in 10 minutes.
- [ ] Final report and slides delivered.
- [ ] GitHub release `v1.0` tagged.

---

## 17. CV / portfolio packaging

> Personal section: **not** part of the course submission.

**CV bullet (fill in the real numbers):**

> _Built an automatic blood-smear analysis pipeline using mathematical morphology (granulometry, top-hat, opening/closing, distance transform, marker-controlled watershed) and a Random Forest classifier; achieved X% F1 detection and Y RBC counting error on BCCD; deployed as a Streamlit web app (Python, scikit-image, OpenCV, scikit-learn)._

**Portfolio checklist:**

- [ ] Pinned GitHub repo with a strong README and demo GIF.
- [ ] Live demo link in the repo description.
- [ ] One "hero" figure: original image next to the annotated result.
- [ ] A short technical post or LinkedIn write-up explaining the watershed splitting and the ablation findings.

---

## 18. References

- N. Otsu, _A Threshold Selection Method from Gray-Level Histograms_, IEEE Trans. SMC, 1979.
- L. Vincent and P. Soille, _Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion Simulations_, IEEE TPAMI, 1991.
- L. Vincent, _Morphological Grayscale Reconstruction in Image Analysis: Applications and Efficient Algorithms_, IEEE TIP, 1993.
- J. Serra, _Image Analysis and Mathematical Morphology_, Academic Press, 1982.
- P. Soille, _Morphological Image Analysis: Principles and Applications_, Springer.
- G. Matheron, _Random Sets and Integral Geometry_, Wiley, 1975 (granulometry).
- R. D. Labati, V. Piuri, F. Scotti, _ALL-IDB: the acute lymphoblastic leukemia image database for image processing_, IEEE ICIP, 2011.
- scikit-image documentation: `morphology`, `segmentation`, `measure` modules.
- BCCD dataset (Shenggan/BCCD_Dataset on GitHub); Raabin-WBC (raabindata.com); ALL-IDB.
