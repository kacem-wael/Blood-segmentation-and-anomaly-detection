# Blood cell detection and analysis with mathematical morphology

**Course:** Morphologie mathématique et application médicale, ENSI
**Authors:** Wael Kacem, Talel Ben Mbarek
**Code:** `git@github.com:kacem-wael/Blood-segmentation-and-anomaly-detection.git`

> Draft generated from the measured results of the repository (`results/tables/`). Every number
> below is reproducible with the commands of the README. Sections marked **[TODO masks]** are
> completed once the 18 manual masks exist (`docs/manual_tasks.md`). Figures are referenced by
> their path in the repository.

---

## Abstract

We present an automatic pipeline that segments, separates, classifies, counts and measures red
blood cells (RBC), white blood cells (WBC) and platelets in stained blood-smear images. The
segmentation is entirely morphological: granulometry estimates the RBC radius `r` of each image
and every parameter is a multiple of `r`; a relative black top-hat corrects the illumination;
Otsu thresholds, a shape-filtered hole filling, openings and closings clean the masks; WBCs are
segmented first by a marker-controlled watershed of the morphological gradient; touching RBCs
are split by h-maxima of the distance transform and a watershed. A random forest on 14
scale-free features decides the class of each object, overriding the morphological branch only
when confident. On the official BCCD test split (72 images, with same-field leakage removed,
evaluated once) the pipeline reaches a detection F1 of 0.90 for WBCs, 0.65 for platelets and an
RBC recall of 0.59 at IoU 0.5 (0.76 at IoU 0.3), in 1.6 s per image on a laptop CPU. An ablation
study quantifies the contribution of each morphological operator, and the results are stable
when the image is zoomed by 0.5 or 2.

## 1. Introduction

A blood smear is examined under a microscope to detect anaemia, infections or leukaemia.
Counting and measuring cells by hand is slow and operator-dependent, and cells overlap, which
makes naive automatic counting inaccurate. Three cell types are visible in a stained smear:

| Cell | Appearance | Clinical use |
| ---- | ---------- | ------------ |
| RBC (erythrocyte) | numerous, pink, round, pale centre | count, size variation (anisocytosis), shape |
| WBC (leukocyte) | rare, large, dark purple nucleus, pale cytoplasm | count, nucleus / cytoplasm (N/C) ratio |
| Platelet | very small purple fragments, often stuck to RBCs | count |

**Objectives.** (1) segment the cells and split touching ones with morphological operators,
(2) classify, count and measure them, (3) flag abnormal cells, (4) evaluate every step
quantitatively against the BCCD annotation, with an ablation of each operator.

## 2. Theory

Let `f` be a grey image and `B` a structuring element (here a disk of radius ρ, `B_ρ`).

- **Erosion / dilation**: `(f ⊖ B)(x) = min_{b∈B} f(x+b)`, `(f ⊕ B)(x) = max_{b∈B} f(x−b)`.
- **Opening** `γ_B = (f ⊖ B) ⊕ B` removes bright structures smaller than `B`;
  **closing** `φ_B = (f ⊕ B) ⊖ B` removes dark ones. Openings are increasing, anti-extensive
  and idempotent; closings increasing, extensive and idempotent.
- **Black top-hat** `BTH(f) = φ_B(f) − f`: dark objects smaller than `B` on a flat background.
  We use the *relative* top-hat `(φ_B(f) − f) / φ_B(f)`: transmitted light is illumination ×
  transmittance, so dividing by the background estimate also removes multiplicative vignetting.
- **Granulometry** (Matheron): the family `γ_{B_ρ}` for growing ρ. The **pattern spectrum**
  `PS(ρ) = A(γ_{B_ρ}(X)) − A(γ_{B_{ρ+1}}(X))` is the area of the objects whose largest inscribed
  disk has radius ≈ ρ; its maximum gives the dominant object size. Binary openings by Euclidean
  disks are computed with two distance transforms (`X ⊖ B_ρ = {d_X > ρ}`, then a dilation by ρ),
  which makes the spectrum exact and fast for every ρ.
- **Morphological gradient** `g(f) = (f ⊕ B_1) − (f ⊖ B_1)`: high on edges.
- **Reconstruction and hole filling**: the holes of a binary set `X` are the complement of the
  reconstruction by dilation of the image border into `X^c`. **h-maxima** `HMAX_h(f)` are the
  regional maxima of `R^δ_f(f − h)`: maxima whose dynamic is at least `h`.
- **Distance transform** `d_X(x) = min_{y∉X} ||x − y||`: maximal at cell centres, minimal at the
  necks between touching cells. **Ultimate erosion** = regional maxima of `d_X`.
- **Watershed** (Vincent & Soille, 1991): flooding of a relief from markers; the dams are the
  segmentation lines. Marker-controlled: only the given markers create basins. The **SKIZ**
  (skeleton by influence zones) is the binary watershed of the distance to the markers.

Illustrations of every operator on our data: `outputs/NN-<step>/` (40 steps, see section 6).

## 3. Related work

See `docs/state_of_the_art.md`. Classical pipelines (colour-space choice, Otsu, morphological
cleaning, distance-transform watershed or circular Hough transform, hand-crafted features + SVM
/ random forest) are interpretable but sensitive to staining, illumination and magnification;
deep-learning detectors (YOLO, Faster R-CNN on BCCD) and segmenters (U-Net, Mask R-CNN,
Cellpose) need annotated training data and are usually evaluated on random, non-official
splits. **[TODO: add 3-5 papers with their dataset, split and metric.]**

## 4. Data

**BCCD** (Shenggan/BCCD_Dataset, MIT license): 364 annotated 640×480 smear images with boxes for
RBC, WBC and platelets (Pascal VOC). Findings of the exploration (notebook 01,
`results/figures/exploration/`):

- **Official split** (`ImageSets/Main`), with **same-field leakage** removed: 11 pairs of shifted
  frames of the same microscope field cross the official splits
  (`results/figures/exploration/same_field_pairs.png`). Test unchanged; train/val images sharing a
  field with a test image are excluded (5); val images sharing a field with train move to train
  (6). Final split **209 / 78 / 72**; the field group is the key of every grouped
  cross-validation.
- **Under-labelled RBCs**: ≈ 30 % of the visible RBCs have no box
  (`unlabelled_rbcs.png`); 15 images have no RBC box at all (excluded from RBC metrics). RBC
  precision and count errors are therefore measured against an incomplete reference.
- 8 duplicate boxes (same class, IoU ≥ 0.7) are ignored.
- Typical sizes from the boxes: RBC 104 px (radius 52), WBC 194 px (1.87 r), platelet 40 px
  (0.38 r).
- **Stain channel**: −b\* of CIE Lab separates WBC nuclei from RBCs best (Fisher ratio 7.6,
  `channel_separability.png`).

**Raabin-WBC** (Kaggle version, 16 633 crops centred on one WBC, another microscope and stain) is
used as an out-of-distribution check of the WBC branch. **ALL-IDB** (Kaggle mirror) was audited
but cannot support the blast-vs-normal task (no normal crops, 41 duplicates).

**Manual masks [TODO masks].** 18 images (10 train/val, 8 test) are being annotated exhaustively
(every cell, nucleus polygons) with LabelMe, starting from drafts derived from the BCCD boxes (not
from our pipeline, to avoid biasing the evaluation).

## 5. Method

Every size is a multiple of the RBC radius `r` (areas of `π r²`); all values are in
`config.yaml`. Pipeline (figure: `results/figures/pipeline/final_overlays.png`):

### 5.1 Scale: granulometry
Coarse mask = Otsu of the median-filtered green channel + hole filling. Its pattern spectrum is
weighted by `1/ρ` (the raw area spectrum is dominated by one WBC or one clump; `1/ρ²` by
debris), smoothed by a Gaussian of constant width in `log ρ` (σ = 0.08, scale-free), and radii
below 5 px are ignored (thin rims of open RBC rings). The peak is the *inscribed* radius; a
calibration factor of 1.24 (median box radius / inscribed radius, fitted on train) converts it to
the box radius. **Accuracy** (`results/tables/scale_check.csv`): median relative error 3.8 % on
val, 90.5 % of the images within 15 % (5 / 74 off by more than 30 %). **Zoom check**: when the
image is rescaled by 0.5, 0.75, 1.5, 2, `r` follows (median ratio 1.000 to 1.007, 97-100 % within
15 %). Re-estimating `r` on the cleaned Phase-4 mask was tested and rejected (85 % within 15 %,
`results/tables/scale_refinement_trial.csv`).

When `r` is far from the working radius (more than ±25 % from 50 px), the image is resampled so
that `r` = 50 px before the morphology and the label map is mapped back: the few operations
defined on the pixel grid (median 2 px, gradient 1 px) then stay valid at any magnification.

### 5.2 Preprocessing
Green channel (haemoglobin absorbs green: best RBC contrast), median filter (disk 2 px),
background = grey closing by a disk of 2.5 r (larger than a WBC, computed at 1/4 resolution
with edge padding), relative black top-hat, percentile stretch. Stain channel: −b\* (median
filtered, stretched). `results/figures/preprocessing/preprocessing_steps.png`,
`illumination_correction.png` (under synthetic vignetting the plain top-hat collapses on 1 / 10
images, Dice 0.34; the relative one stays ≥ 0.90).

### 5.3 Segmentation and cleaning
- Cell mask: Otsu of the top-hat ∪ stain mask. Stain mask: Otsu of −b\*; when the stain covers
  more than half of the cells (whole population stained, e.g. Raabin-WBC: 0.95 vs 0.11 on BCCD) a
  3-class Otsu keeps only the top class. Safety checks return empty masks on blank images.
- **Shape-filtered hole fill**, before the opening (else the opening breaks the thin rim of a
  pale-centred RBC): a hole is filled only if small (≤ 0.5 π r²), round (circularity ≥ 0.6,
  Crofton perimeter) and convex (solidity ≥ 0.9). Gaps between three or four touching cells have
  concave sides (solidity ≤ 0.8) and stay background, whereas a plain `fill_holes` glues the
  cells together (`results/figures/pipeline/hole_fill_plain_vs_shape.png`). A closing of 0.1 r
  first seals RBC rims broken by a thin gap (C shapes), only to find the holes.
- Opening 0.1 r (noise, thin bridges), closing 0.05 r, removal of objects < 0.02 π r²; the stain
  mask has its own finer cleaning (platelet cores are only 0.005-0.04 π r²).

### 5.4 WBC branch (before the RBC watershed)
- **Nucleus**: stain components grouped by a closing of 0.15 r (the lobes of a neutrophil
  nucleus become one object), kept if ≥ 0.3 π r².
- **Whole cell**: marker-controlled watershed of the morphological gradient of the green and
  stain channels. Markers: the nucleus grown by 3 px (past its own edge ridge, otherwise the
  cytoplasm is lost to the background), a background marker (farther than 1 r from the nucleus,
  or clearly background in the top-hat), and the centres of touching RBCs (so that an RBC is not
  absorbed). Pale cytoplasm fainter than the Otsu threshold is left unassigned, and the gradient
  decides where the cell ends (`results/figures/pipeline/wbc_gradient_watershed.png`).
- **N/C ratio** = nucleus area / cytoplasm area; a cytoplasm smaller than 10 % of the cell sets a
  `cytoplasm_uncertain` flag instead of a meaningless ratio.
- These two changes raised the WBC F1@0.5 on train/val from 0.864 to 0.982
  (`results/tables/detection_wbc.csv`).

### 5.5 Platelets
Stain components outside the WBCs, grouped by a closing of 0.1 r, with area in
[0.005, 0.3] π r². **Enclosure filter**: purple specks *inside* an RBC (stain deposits, bluish
pale centres) are surrounded by cell pixels, platelets are not. The share of non-purple cell
pixels in a ring of 0.08-0.18 r around each candidate is measured (one feature transform for all
candidates); above 0.4 the candidate is rejected. On train this keeps 179 / 187 labelled
platelets and removes 80 % of the other candidates
(`results/figures/pipeline/platelet_enclosure.png`); platelet F1@0.5 on train/val went from 0
to 0.72. The segmented object is the stained core; BCCD boxes the whole platelet with its halo,
so platelet boxes are padded by 0.2 r for matching only (core dilated by 0.2 r: IoU ≥ 0.5 for
94 % of labelled platelets).

### 5.6 Splitting touching RBCs
RBC mask = cell mask − WBCs − platelets dilated by 2 px (a platelet stuck to an RBC would leave a
notch that creates a false marker), opened. Distance transform, Gaussian smoothing (0.05 r),
**h-maxima** with `h = 0.15 r` as markers, **watershed of the inverted distance** inside the mask.
Components without a marker keep their own label; fragments < 0.1 π r² are merged into the
neighbour sharing the longest border (`results/figures/pipeline/watershed_before_after.png`,
`h_effect.png`). `h` was chosen on train/val (section 6.3). Clumps the watershed cannot split
(area > 2 π r², solidity < 0.9) are flagged and counted by area / median RBC area. Bonus: the
ultimate erosion and the geodesic SKIZ of the markers (`skiz_ultimate_erosion.png`,
`outputs/33-ultimate_erosion`, `outputs/35-skiz`).

### 5.7 Contours
Morphological gradient of the **label map**, `δ(L) ≠ ε(L)`: unlike the contour of the binary
mask it also draws the watershed lines between touching cells
(`results/figures/pipeline/contours_mask_vs_labels.png`).

## 6. Machine learning

- **Features** (per object, scale-free): area / π r², circularity (Crofton perimeter),
  eccentricity, solidity, extent, equivalent diameter / 2r, mean and std of L\*, a\*, b\*, stain
  fraction and mean stain. The branch and the N/C ratio are *not* inputs (the model would copy
  the rules).
- **Labels**: one-to-one Hungarian matching of object and BCCD boxes (IoU ≥ 0.5), then a
  centroid-in-box fallback (IoU ≥ 0.2). Unmatched objects are *Unlabelled* and excluded (they are
  often real RBCs without a box); on manual-mask images they become *Artefact*.
- **Models**: rule baseline (stain fraction and area), logistic regression, RBF SVM, random
  forest (300 trees), all with balanced class weights; GroupKFold (5 folds) over train + val
  grouped by microscope field; bootstrap 95 % CI over images.

| Model (GroupKFold, train+val, 3 104 objects) | macro-F1 [95 % CI] | F1 RBC | F1 WBC | F1 Platelet |
| --- | --- | --- | --- | --- |
| rule baseline | 0.979 [0.971, 0.987] | 0.995 | 0.956 | 0.987 |
| logistic regression | 0.992 [0.986, 0.996] | 0.999 | 0.993 | 0.983 |
| SVM (RBF) | 0.989 [0.982, 0.995] | 0.999 | 0.990 | 0.978 |
| **random forest** (selected) | **0.994 [0.989, 0.998]** | 0.999 | 0.995 | 0.989 |
| pipeline branch | 0.992 [0.985, 0.997] | 0.999 | 0.997 | 0.980 |

Feature importance: stain fraction, mean stain, relative area and diameter dominate
(`results/figures/classification/feature_importance.png`). On *matched* objects the problem is
nearly separable. End to end, however, the model also sees unmatched objects (small RBC
fragments) it was never trained on: used without restriction it lowered the platelet F1 on val
(0.655 vs 0.671 for the branch). The final classifier therefore overrides the branch **only when
its probability is ≥ 0.8**, which is the best end-to-end variant (val platelet F1 0.675; table
`results/tables/eval_val_classifiers.csv`).

**Anomalies**: on eligible RBCs (not on the border, not clumps, solidity ≥ 0.9) a robust z-score
(median / MAD) of size and circularity flags outliers (|z| > 3); the coefficient of variation of
the RBC area is reported as an anisocytosis indicator (analogous to the RDW). No dataset has RBC
anomaly labels: this part is qualitative (`outputs/40-anomalies/`).

## 7. Experiments and results

Detection: one-to-one Hungarian box matching per class at IoU ≥ 0.5 (and 0.3: BCCD boxes are
loose). Images without RBC boxes are excluded from RBC metrics. **The test set was evaluated
once**, after all choices were made on train/val (stamp `results/tables/TEST_SET_EVALUATED.txt`).

### 7.1 Main results

| Class | Val P / R / F1 | **Test P / R / F1** | Test R@0.3 | Test count MAE / image |
| --- | --- | --- | --- | --- |
| WBC | 0.99 / 0.99 / 0.99 | **0.83 / 0.97 / 0.90** | 1.00 | 0.17 |
| Platelet | 0.63 / 0.73 / 0.68 | **0.63 / 0.67 / 0.65** | 0.70 | 0.53 |
| RBC | 0.41 / 0.60 / 0.49 | **0.38 / 0.59 / 0.47** | 0.76 | 7.5 |

RBC precision and count error are bounded by the missing RBC boxes (≈ 30 %); the predicted /
labelled RBC ratio is 1.54 on test, close to the 1 / 0.7 ≈ 1.43 expected for an exhaustive
detection. Classification of the matched test objects: macro-F1 **0.989 [0.970, 1.000]**.
Runtime: **1.6 s** per 640×480 image (one CPU process, mean of 10 test images).

**Pixel scores [TODO masks]**: Dice / IoU of the cell mask and per-cell F1 on the 8 test masks
(`results/tables/eval_test_masks.csv`).

### 7.2 Ablation (test set; classes from the pipeline branch, so that only morphology changes)

| Configuration | RBC count ×(pred/gt) | RBC R@0.5 | RBC F1@0.5 | WBC F1@0.5 | Platelet F1@0.5 |
| --- | --- | --- | --- | --- | --- |
| A Otsu + connected components | 5.34 | 0.371 | 0.117 | 0.609 | 0.607 |
| B + preprocessing (top-hat) | 3.00 | 0.398 | 0.199 | 0.667 | 0.638 |
| C + shape-filtered hole fill | 2.90 | 0.393 | 0.201 | 0.667 | 0.652 |
| D + opening / closing | 1.13 | 0.393 | 0.370 | 0.680 | 0.652 |
| E + distance transform + watershed | 1.63 | 0.590 | 0.448 | 0.831 | 0.652 |
| **F + WBC first (full)** | 1.55 | 0.592 | 0.464 | **0.896** | 0.662 |
| F with plain `fill_holes` | 1.51 | 0.606 | 0.484 | 0.896 | 0.657 |

- Without morphology the threshold yields 5× too many RBC objects (noise); the opening / closing
  brings the count to the right order (D), and the **watershed raises the RBC recall from 0.39 to
  0.59** (E): touching cells are split.
- **Segmenting the WBC first** raises the WBC F1 from 0.83 to 0.90 (test) and from 0.92 to 0.99
  (val): a WBC split by the distance watershed or merged with an RBC is no longer a WBC box.
- Plain vs shape-filtered hole filling: shape-filtered is better on val (RBC F1 0.489 vs 0.478)
  but not on test (0.464 vs 0.484). At box level the difference is small: the plain fill glues
  cells, but the watershed then cuts most of the glued blobs again. The pixel Dice on the manual
  masks is the right metric to settle it **[TODO masks]**.
- Baselines of project.md 10.3: (1) Otsu + components = row A, (2) Otsu + opening / closing
  without watershed = row D. A deep-learning baseline (Cellpose) was not run (disk space).

### 7.3 Parameter sweeps (val; `results/figures/evaluation/eval_val_sweeps.png`)
- `h` (h-maxima dynamic): from 0.05 r to 0.5 r, RBC recall falls from 0.68 to 0.41 and the
  predicted / labelled ratio from 1.82 to 1.15: small `h` over-segments, large `h` merges.
  F1 is maximal around 0.10-0.15 r (0.488-0.489); 0.15 r keeps the count ratio closest to the
  expected ≈ 1.43.
- Opening radius: 0.05-0.1 r best (RBC F1 0.495 / 0.488); above 0.2 r small and thin cells are
  destroyed (0.35 at 0.3 r).

### 7.4 Scale robustness (val, 20 images rescaled; `eval_val_scale_robustness.csv`)

| Zoom | RBC F1@0.5, estimated r | RBC F1@0.5, fixed pixels (r = 50) | WBC F1, est. / fixed |
| --- | --- | --- | --- |
| ×0.5 | **0.540** | 0.517 | **1.00** / 0.89 |
| ×1 | 0.540 | 0.541 | 1.00 / 1.00 |
| ×2 | **0.546** | 0.419 | **1.00** / 0.86 |

With the granulometric `r` (and the resampling to the working radius) the results do not depend
on the magnification; fixed-pixel parameters degrade at ×2 (RBC count ×2.2, platelet F1 0.30).

### 7.5 Other microscope and stain: Raabin-WBC (720 crops, `raabin_detection_summary.csv`)
The WBC at the centre of the crop is detected in **90 %** of the crops; exactly one WBC is found
in 75 % (other cells cut by the crop border or stained RBCs account for the rest). Without the
stain-adaptive threshold (section 5.3) only 37 % had exactly one WBC, because the RBCs of this
stain are purple-grey.

## 8. Discussion

- **What worked**: scale normalisation by granulometry (zoom invariance), the relative top-hat,
  the shape-filtered hole fill, WBC-first gradient watershed, the platelet enclosure filter, and
  the marker-controlled distance watershed, each measurable in the ablation.
- **Test WBC precision (0.83 vs 0.99 on val)**: of the 14 false positives, 9 come from three
  images where the granulometry failed (r = 14-17 px instead of ≈ 50, e.g. BloodImage_00169: RBCs
  with open pale centres look like thin rings), and in 3 of the 4 test images without any WBC box
  the detected object **is** a real WBC that BCCD did not box (BloodImage_00135, 00178, 00327). The
  WBC precision is therefore a lower bound too (`results/figures/evaluation/eval_test_failures.png`).
- **Failure cases**: granulometry failures (≈ 5 % of the images; ring-shaped RBCs, a WBC or a
  clump dominating a sparse field), heavily overlapping RBCs left as clumps, pale cytoplasm
  partly lost, purple debris taken for platelets.
- **Limitations**: box-only reference (missing RBC boxes; no pixel ground truth until the masks
  exist), one stain and microscope for tuning (BCCD), no RBC anomaly labels (qualitative flags),
  ALL-IDB task impossible with the public mirror.

## 9. Conclusion and future work

A fully morphological segmentation, with a light classifier on top, detects WBCs reliably, finds
two thirds of the platelets and most RBCs, is explainable step by step (40 inspectable stages)
and is independent of the magnification. Future work: pixel evaluation on the manual masks,
a more robust scale estimate (e.g. combining the pattern spectrum with the WBC size, or several
peaks), a deep-learning comparison (Cellpose / YOLO), other stains, and a hybrid where a learned
model only proposes markers for the watershed.

## References

See `docs/state_of_the_art.md` (Matheron 1975; Serra 1982; Meyer 1979; Maragos 1989; Otsu 1979;
Vincent & Soille 1991; Vincent 1993; Beucher & Meyer 1993; Soille; Labati et al. 2011;
Kouzehkanan et al. 2022; Stringer et al. 2021), BCCD dataset (Shenggan/BCCD_Dataset),
scikit-image.

## Appendix A: parameters

All parameters with their justification: `config.yaml` (sizes in units of `r`).

## Appendix B: the 40 pipeline steps

`outputs/README.md` lists the 40 steps (`01-original` … `40-anomalies`) with their parameters;
`make outputs` regenerates one folder per step for 20 train/val images.
