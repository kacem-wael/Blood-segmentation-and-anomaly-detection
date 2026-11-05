# What is left to do by hand

Two tasks only: **drawing the 18 manual masks** and **committing / pushing to GitHub**.
Everything else (code, ML, tests, evaluation, step images, report draft, slides) is done.

All commands are run in PowerShell from the project folder:

```powershell
cd "C:\Users\MSI\Desktop\ENSI\3eme ENSI\DS-CV\morphologie mathématique et application médicale\projet"
.venv\Scripts\activate
```

---

## Task 1: the 18 manual masks (~20 min per image)

BCCD only has boxes and ~30 % of its RBCs have none. 18 fully annotated images give the pixel
**Dice / IoU** of the segmentation and **exact RBC counts**. Everything is prepared:

- LabelMe 6.3.1 is installed in `.venv-labelme` (it was missing `onnxruntime`: fixed);
- a **draft** for each image is in `data/masks_manual/drafts/`: one ellipse per BCCD box with
  its class, and the WBC nucleus. The drafts come from the BCCD boxes, **not** from our
  pipeline, so correcting them does not bias the evaluation;
- one command opens the next image, with only the 4 allowed labels, and checks it afterwards;
- a file only counts once you tick **reviewed**: an unfinished draft never enters the evaluation.

### 1.1 Who does which image

| Person | train/val (tuning) | test (final report) |
|--------|--------------------|---------------------|
| Wael  | 00109, 00168, 00195, 00200, 00206 | 00021, 00205, 00233, 00293 |
| Talel | 00207, 00214, 00235, 00264, 00272 | 00320, 00341, 00366, 00407 |

Do **one** train/val image each first, compare the two results, agree on the rules, then continue.

### 1.1b Working as a team (each on their own computer)

The masks, their drafts and `data/splits.json` are versioned; the images are not (each
computer downloads BCCD). Each person only touches their own 9 files, so Git never conflicts.

**Wael (once):** push the project (Task 2 below), then add Talel as collaborator on GitHub
(repository > Settings > Collaborators).

**Talel (once, ~15 min):**

```powershell
git clone git@github.com:kacem-wael/Blood-segmentation-and-anomaly-detection.git
cd Blood-segmentation-and-anomaly-detection
py -3.11 -m venv .venv
.venv\Scriptsctivate
pip install -r requirements.txt
pip install -e .
python scripts\download_data.py --only bccd     # BCCD images (GitHub, no account needed)
python scriptsuild_index.py                   # local index with his own paths (~1 min)
py -3.11 -m venv .venv-labelme
.venv-labelme\Scripts\pip install labelme onnxruntime
```

Do **not** run `make_splits.py`: the split comes with the repository (`data/splits.json`).

**Each person, for every image:**

```powershell
git pull                                         # get the other person's finished masks
python scriptsnnotate.py --who talel           # or --who wael
git add data/masks_manual/BloodImage_00207.json   # only your file
git commit -m "Mask BloodImage_00207"
git push
```

`python scriptsnnotate.py status` shows the 18 images and who has finished what (after a
`git pull`). When all 18 are reviewed, one person runs `make with-masks` (section 1.3) and
commits the new results.

### 1.2 Annotate one image

```powershell
python scripts\annotate.py next --who wael       # or --who talel  (or: make annotate WHO=wael)
```

It prints the status of the 18 images, copies the draft of your next image to
`data\masks_manual\<image_id>.json` and opens it in LabelMe. In LabelMe:

1. **Fix every draft polygon** so that it follows the real cell outline: select it, drag its
   points (`Ctrl+J` = edit mode; right-click a point to delete it; click on an edge to add one).
   Draft ellipses come from loose boxes: almost all need adjusting.
2. **Add the cells BCCD forgot** (often 5 to 10 RBCs per image): `Ctrl+N` (create polygon),
   click around the outline every few pixels, click the first point to close, choose the label.
3. Rules:
   - **every** cell gets one polygon, including cells cut by the image border (follow the edge);
   - **RBC**: outer outline, pale centre included (do not draw the hole);
   - **WBC**: one `WBC` polygon for the whole cell (outer edge of the pale cytoplasm) **and**
     `nucleus` polygon(s) inside it (one per lobe is fine). Check the draft nucleus carefully;
   - **Platelet**: the purple fragment and its pale halo;
   - **overlapping cells**: the polygon drawn *later* wins the shared pixels; draw the cell
     underneath first. To bring a polygon to the front, delete and redraw it;
   - delete a polygon that is not a cell (`Delete` key); do not annotate debris.
4. When the image is completely done: tick **reviewed** in the *Flags* panel (top right), save
   (`Ctrl+S`), close LabelMe.

`annotate.py` then runs the checker: cells per class next to the BCCD boxes, and warnings
(hidden polygon, RBC of strange size, WBC without nucleus, nucleus outside a WBC, draft
polygons never edited). It writes a preview `outputs\masks_preview\<image_id>.png`: open it and
check every contour. To fix something, reopen the same image:

```powershell
python scripts\annotate.py BloodImage_00109       # reopens the existing file, nothing is lost
python scripts\annotate.py status                 # what is done / left
python scripts\check_masks.py                     # check all files at once
```

### 1.3 When masks are finished: one command re-runs everything

```powershell
make with-masks          # or the commands below
```

```powershell
python scripts\build_features.py                  # mask images: exact labels, incl. Artefact
python scripts\train_classifier.py                # the classifier also learns the Artefact class
python scripts\run_evaluation.py --split val --ablation
python scripts\run_evaluation.py --split test --ablation --force
python tests\test_regression.py --update
python scripts\export_steps.py
```

The Dice / IoU / per-cell F1 then appear in `results\tables\eval_*_masks.csv` and in the
`Dice` / `cell F1` columns of `eval_*_ablation.csv`. `--force` is needed because the test set was
already evaluated once; re-running with **unchanged parameters** only to add the masks keeps the
"test set used once" rule (say so in the report). Then copy the new numbers into the README
results table and section 6 of `docs/report/report.md`.

---

## Task 2: commit and push to GitHub

Remote: `git@github.com:kacem-wael/Blood-segmentation-and-anomaly-detection.git`. Nothing is
committed yet. `.gitignore` keeps the datasets, environments (`.venv*`) and `outputs/` out; the
model (`models/classifier.joblib`, 250 KB), report figures and summary tables are included.

```powershell
git add .                                         # includes data/splits.json and data/masks_manual/
git status                                        # check: no data/raw, no .venv*, no kaggle.json
git commit -m "Blood cell analysis: morphology pipeline, classifier, evaluation, report"
git remote add origin git@github.com:kacem-wael/Blood-segmentation-and-anomaly-detection.git
git push -u origin main
```

On github.com, **Actions** tab: the `tests` workflow (flake8, black, pytest on synthetic images,
no dataset needed) should be green. For the final version:

```powershell
git tag -a v1.0 -m "Final version"
git push origin v1.0
```

---

## Notes

- Disk space: drive C: was full (63 MB free). The pip download cache was purged (1.6 GB freed);
  keep at least 1 GB free before installing anything else. Cellpose (optional deep-learning
  baseline, project.md 10.3) was not installed for this reason (~1 GB with torch).
- Before citing a reference of `docs/state_of_the_art.md` in the report, check it.
- Ask the instructor which language the report must be in (the draft is in English).
