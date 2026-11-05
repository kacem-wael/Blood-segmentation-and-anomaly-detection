# State of the art (Phase 1)

Draft of the report's related-work section. It deliberately quotes **no metric values
from other papers**: before citing a number, check it in the paper and note which BCCD
split it was measured on (most deep-learning papers use their own random split, which is
not comparable with the official split used here).

## 1. Mathematical morphology foundations

- **Mathematical morphology** (Matheron, 1975; Serra, 1982): erosion, dilation, opening and
  closing by a structuring element; openings are increasing, anti-extensive and idempotent,
  which is what makes **granulometries** (families of openings of growing size, Matheron) a
  size distribution. The derivative of the opened area, the **pattern spectrum** (Maragos, 1989),
  peaks at the dominant object size: our RBC radius `r`.
- **Top-hat transforms** (Meyer, 1979): image minus opening (white top-hat) or closing minus
  image (black top-hat) extract objects smaller than the structuring element from a slowly
  varying background; used for illumination correction.
- **Morphological reconstruction** (Vincent, 1993): fast algorithms for reconstruction by
  dilation; hole filling and regional / h-maxima are special cases.
- **Distance transform and watershed**: the watershed by immersion (Vincent & Soille, 1991)
  and marker-controlled watershed (Beucher & Meyer, 1993) are the standard way to split
  touching convex objects: markers from the maxima of the distance function, flooding of the
  inverted distance. Over-segmentation is controlled by the marker choice (h-maxima dynamic,
  ultimate erosion). The SKIZ (skeleton by influence zones) is the binary counterpart.
- **Morphological gradient** (Beucher): dilation minus erosion, the usual relief for a
  watershed on grey-level images (here: the WBC cytoplasm boundary).

## 2. Classical blood-smear analysis

Classical pipelines follow the same pattern as this project:
1. colour-space choice to separate stained nuclei from erythrocytes (green channel, HSV
   saturation, CIE Lab or CMYK components are common),
2. global or adaptive thresholding (Otsu, 1979) plus morphological cleaning,
3. separation of overlapping RBCs by distance transform + watershed, or by the circular
   Hough transform (RBCs are nearly circular),
4. hand-crafted shape / colour / texture features and a classical classifier (SVM, random
   forest, k-NN) for cell types or abnormalities.

Their known weaknesses, which motivated our design choices: sensitivity to staining and
illumination (relative top-hat, Lab channel), fixed pixel parameters (granulometric `r`),
merging of touching cells by plain hole filling (shape-filtered hole fill), and WBC/RBC
contact (WBC segmented first).

Leukaemia-oriented work uses **ALL-IDB** (Labati, Piuri & Scotti, 2011): ALL-IDB1 (full
smears with blast positions) and ALL-IDB2 (single-cell crops, blast vs normal), typically
with nucleus segmentation + shape/texture features + a classifier.

## 3. Deep learning

- **Object detection** (YOLO family, Faster R-CNN, SSD) is the common baseline on **BCCD**
  (Shenggan/BCCD_Dataset, boxes for RBC / WBC / platelets). Box-level mAP is reported; the
  under-labelling of RBCs in BCCD affects these numbers too.
- **Instance segmentation**: U-Net (Ronneberger et al., 2015) for semantic masks; Mask R-CNN
  (He et al., 2017); **Cellpose** (Stringer et al., 2021), a generalist cell segmentation model
  predicting flow fields, usable without training as an optional baseline (project.md 10.3).
- **WBC classification** on crops (e.g. **Raabin-WBC**, Kouzehkanan et al., 2022) is dominated
  by CNNs; the dataset also provides nucleus / cytoplasm ground truth for a subset.

## 4. Positioning of this project

| Aspect | Typical deep-learning work | This project |
| ------ | -------------------------- | ------------ |
| Segmentation | learned (needs masks / boxes) | morphological, no training |
| Scale | fixed by the training data | estimated per image (granulometry), resampling to a working scale |
| Interpretability | low | every step is an explainable operator, ablation per operator |
| Learning | end to end | only cell-type decision (random forest on 14 scale-free features) |
| Evaluation | random split, mAP | official BCCD split, leakage removed, P/R/F1 per class with caveats on missing RBC boxes |

## References

- G. Matheron, *Random Sets and Integral Geometry*, Wiley, 1975.
- J. Serra, *Image Analysis and Mathematical Morphology*, Academic Press, 1982.
- F. Meyer, "Iterative image transformations for an automatic screening of cervical smears",
  *J. Histochem. Cytochem.*, 1979 (top-hat).
- P. Maragos, "Pattern spectrum and multiscale shape representation", *IEEE TPAMI*, 1989.
- N. Otsu, "A threshold selection method from gray-level histograms", *IEEE Trans. SMC*, 1979.
- L. Vincent and P. Soille, "Watersheds in digital spaces: an efficient algorithm based on
  immersion simulations", *IEEE TPAMI*, 1991.
- L. Vincent, "Morphological grayscale reconstruction in image analysis: applications and
  efficient algorithms", *IEEE TIP*, 1993.
- S. Beucher and F. Meyer, "The morphological approach to segmentation: the watershed
  transformation", in *Mathematical Morphology in Image Processing*, 1993.
- P. Soille, *Morphological Image Analysis: Principles and Applications*, Springer.
- R. D. Labati, V. Piuri, F. Scotti, "ALL-IDB: the acute lymphoblastic leukemia image database
  for image processing", *IEEE ICIP*, 2011.
- O. Ronneberger, P. Fischer, T. Brox, "U-Net", *MICCAI*, 2015.
- K. He, G. Gkioxari, P. Dollár, R. Girshick, "Mask R-CNN", *ICCV*, 2017.
- C. Stringer, T. Wang, M. Michaelos, M. Pachitariu, "Cellpose: a generalist algorithm for
  cellular segmentation", *Nature Methods*, 2021.
- Z. M. Kouzehkanan et al., "A large dataset of white blood cells containing cell locations
  and types, along with segmented nuclei and cytoplasm", *Scientific Reports*, 2022 (Raabin-WBC).
