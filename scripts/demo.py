"""Live demo: walk through the 40 pipeline steps of one image (`make demo`).

    python scripts/demo.py                          # BloodImage_00226 (test set), manual pace
    python scripts/demo.py --image path/to/img.jpg  # any smear image (or a BCCD id)
    python scripts/demo.py --auto 3                 # advance automatically every 3 s

The pipeline is computed once, then every step is announced in the terminal, saved to
demo/NN-step.png at that moment and shown in a window (input on the left, step on the
right). Window keys: Right / Space / Enter = next step, Left = previous step,
A = automatic on/off, Q / Escape = save the remaining steps and show the result.
At the end demo/ also holds the final overlay, one CSV row per cell and a summary.
The demo/ folder is recreated on every run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from bloodcells.classifier import load_model
from bloodcells.config import PROJECT_ROOT, load_config
from bloodcells.datasets import rbc_radius_per_image
from bloodcells.io_utils import load_image
from bloodcells.pipeline import PIPELINE_STEPS, pipeline_step_images, run
from bloodcells.segmentation import STEPS
from bloodcells.visualization import overlay, save_image, spectrum_figure

ALL_STEPS = STEPS + PIPELINE_STEPS
DEFAULT_IMAGE = "BloodImage_00226"  # test image: 1 WBC and 1 platelet matched, 87 % RBC recall

# (first step, last step, phase title, colour)
PHASES = [
    (1, 1, "Image d'entrée", "#3a3f4d"),
    (2, 5, "Phase 1 · Échelle : granulométrie", "#3a3f4d"),
    (6, 8, "Phase 2 · Prétraitement : correction de l'éclairage", "#b8730a"),
    (9, 15, "Phase 3 · Seuillage des cellules et de la coloration", "#2f6db3"),
    (16, 22, "Phase 4 · Nettoyage morphologique du masque", "#2f6db3"),
    (23, 27, "Phase 5 · Globules blancs", "#5b3fb0"),
    (28, 30, "Phase 6 · Plaquettes", "#0f8a76"),
    (31, 36, "Phase 7 · Séparation des globules rouges", "#c0283f"),
    (37, 40, "Phase 8 · Étiquetage, classification, anomalies", "#14161e"),
]

# step name -> (French title, what it does, is a morphological operator)
TEXT = {
    "original": ("Image originale", "frottis coloré, 3 canaux RGB", False),
    "green_channel": ("Canal vert", "l'hémoglobine absorbe le vert : les cellules sont sombres", False),
    "median_filter": ("Filtre médian", "enlève le bruit impulsionnel sans flouter les bords", True),
    "scale_coarse_mask": ("Masque grossier", "Otsu + remplissage des trous, uniquement pour mesurer r", True),
    "scale_granulometry": (
        "Granulométrie",
        "spectre de motifs (ouvertures par disques) → rayon r des GR",
        True,
    ),
    "background_closing": ("Fermeture", "un disque de 2,5 r efface les cellules : il reste le fond", True),
    "relative_tophat": ("Top-hat noir relatif", "(fond − image) / fond : éclairage corrigé, fond plat", True),
    "cells_normalized": ("Normalisation", "étirement du contraste entre percentiles", False),
    "cells_otsu": ("Seuillage d'Otsu", "masque brut des cellules", False),
    "purple_lab_b": ("Canal violet −b*", "espace CIE Lab : noyaux et plaquettes ressortent", False),
    "purple_median_normalized": ("Médian + normalisation", "canal violet débruité", True),
    "purple_otsu": ("Seuillage d'Otsu (violet)", "masque brut de la coloration", False),
    "purple_opening": ("Ouverture", "supprime les petits points violets isolés", True),
    "purple_closing": ("Fermeture", "recolle les morceaux d'un même objet violet", True),
    "purple_remove_small_objects": ("Suppression des petits objets", "→ masque violet final", True),
    "cells_union_purple": ("Union", "masque des cellules ∪ masque violet", False),
    "gap_closing": ("Fermeture des anneaux", "referme les GR ouverts en forme de C", True),
    "hole_fill": (
        "Remplissage filtré par la forme",
        "comble les centres pâles (petits, ronds, convexes), pas les espaces entre cellules",
        True,
    ),
    "opening": ("Ouverture", "coupe les ponts fins et retire le bruit", True),
    "closing": ("Fermeture", "lisse les contours", True),
    "remove_small_objects": ("Suppression des petits objets", "→ masque final des cellules", True),
    "segmentation_overlay": ("Superposition", "cellules (vert) et coloration (magenta) sur l'image", False),
    "wbc_lobe_grouping": ("Fermeture des lobes", "les lobes d'un même noyau deviennent un seul objet", True),
    "wbc_nuclei": ("Noyaux des GB", "grands objets violets, une couleur par GB", True),
    "wbc_gradient": ("Gradient morphologique", "dilatation − érosion : fort sur les bords", True),
    "wbc_markers": ("Marqueurs", "noyau, fond et GR voisins concurrents", True),
    "wbc_cells": ("Ligne de partage des eaux", "GB entier (cyan) et noyau (magenta) → rapport N/C", True),
    "platelet_candidates": (
        "Candidats plaquettes",
        "petits objets violets hors GB, regroupés par fermeture",
        True,
    ),
    "platelet_enclosure": (
        "Filtre d'encerclement",
        "anneau autour de chaque candidat : gardé (vert), enfermé dans un GR (rouge)",
        True,
    ),
    "platelets": ("Plaquettes retenues", "cœurs colorés des plaquettes", False),
    "rbc_mask": ("Masque des GR", "cellules − GB − plaquettes dilatées, puis ouverture", True),
    "distance_transform": (
        "Transformée en distance",
        "maximale au centre de chaque GR, minimale aux cols",
        True,
    ),
    "ultimate_erosion": ("Érosion ultime", "maxima régionaux : trop de marqueurs (comparaison)", True),
    "hmaxima_markers": ("h-maxima", "un seul marqueur par GR (dynamique ≥ 0,15 r)", True),
    "skiz": ("SKIZ", "zones d'influence des marqueurs", True),
    "watershed_rbc": ("Ligne de partage des eaux", "les GR collés sont séparés, une couleur par GR", True),
    "label_map": ("Carte d'étiquettes", "GB, plaquettes et GR réunis", False),
    "label_contours": ("Gradient de la carte d'étiquettes", "contours + lignes de séparation", True),
    "classification": ("Classification", "GR rouge, GB bleu, plaquette verte (forêt aléatoire)", False),
    "anomalies": ("Anomalies", "anomalie (jaune), amas (orange), bord (gris)", False),
}
assert [n for n, _ in ALL_STEPS] == list(TEXT), "TEXT must follow the pipeline steps"


# ---------------------------------------------------------------- terminal
os.system("")  # enables ANSI colours in the Windows console
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"


def rgb(hex_colour: str) -> str:
    r, g, b = (int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"\033[38;2;{r};{g};{b}m"


GREEN, RED, CYAN = rgb("#2fb79f"), rgb("#f06a83"), rgb("#7fb6ff")


def phase_of(i: int) -> tuple[str, str]:
    return next((title, col) for a, b, title, col in PHASES if a <= i <= b)


def banner(title: str) -> None:
    width = 78
    print(f"\n{BOLD}{'═' * width}\n  {title}\n{'═' * width}{RESET}")


# ---------------------------------------------------------------- data
def resolve_image(arg: str, cfg: dict) -> tuple[str, Path]:
    path = Path(arg)
    if path.exists():
        return path.stem, path
    index = pd.read_csv(cfg["paths"]["index"] / "bccd_images.csv").set_index("image_id")
    if arg not in index.index:
        raise SystemExit(f"Image introuvable : {arg} (chemin ou identifiant BCCD attendu)")
    return arg, Path(index.loc[arg, "image_path"])


def reference_radius(image_id: str, cfg: dict) -> float | None:
    try:
        r = rbc_radius_per_image(pd.read_csv(cfg["paths"]["index"] / "bccd_boxes.csv"))
        return float(r[image_id]) if image_id in r.index else None
    except (OSError, KeyError):
        return None


def step_details(res) -> dict[str, str]:
    """Numbers measured on this image, appended to some steps."""
    a, f = res.analysis, res.features
    wbc = f[f["cls"] == "WBC"]
    details = {
        "scale_granulometry": f"r = {res.r:.1f} px",
        "wbc_nuclei": f"{int(a.wbc.nuclei.max())} noyau(x)",
        "platelets": f"{int(a.platelets.max())} plaquette(s)",
        "rbc_mask": f"{ndi.label(a.rbc_mask)[1]} composantes connexes",
        "watershed_rbc": f"{res.counts['RBC']} globules rouges après séparation",
        "classification": "{RBC} GR · {WBC} GB · {Platelet} plaquette(s)".format(**res.counts),
    }
    if len(wbc) and np.isfinite(wbc["nc_ratio"]).any():
        details["wbc_cells"] = "N/C = " + ", ".join(f"{v:.2f}" for v in wbc["nc_ratio"].dropna())
    return details


def write_step(i: int, name: str, images: dict, res, out: Path, r_ref, image_id: str) -> Path:
    target = out / f"{i:02d}-{name}.png"
    seg = res.analysis.seg
    if name == "scale_granulometry":
        if seg.scale is None:
            save_image(target, np.zeros((240, 320), np.uint8))
        else:
            spectrum_figure(seg.scale, r_ref, title=image_id).savefig(target, dpi=110)
    elif name == "segmentation_overlay":
        save_image(target, overlay(seg.steps["original"], seg.cell_mask, seg.purple_mask))
    else:
        save_image(target, images[name])
    return target


# ---------------------------------------------------------------- window
class Viewer:
    """One matplotlib window updated in place; returns the user's choice for each step."""

    def __init__(self, original: np.ndarray, image_id: str, auto: float):
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt, self.auto, self.action = plt, auto, None
        plt.rcParams["toolbar"] = "None"
        for key in ("quit", "back", "forward", "home", "save", "grid", "yscale", "xscale"):
            plt.rcParams[f"keymap.{key}"] = []
        self.fig = plt.figure(figsize=(15, 7.6), facecolor="#f6f4ef")
        self.fig.canvas.manager.set_window_title(f"Démo pipeline · {image_id}")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.75], left=0.02, right=0.98, top=0.84, bottom=0.06)
        self.ax_in, self.ax_out = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        for ax in (self.ax_in, self.ax_out):
            ax.set_axis_off()
        self.ax_in.imshow(original)
        self.ax_in.set_title("Image d'entrée", fontsize=13, color="#3a3f4d")
        self.phase = self.fig.text(0.02, 0.94, "", fontsize=14, fontweight="bold")
        self.title = self.fig.text(0.02, 0.885, "", fontsize=20, fontweight="bold", color="#14161e")
        self.help = self.fig.text(
            0.98,
            0.94,
            "→ / Espace : suivant   ← : précédent   A : auto   Q : terminer",
            fontsize=10,
            color="#717686",
            ha="right",
        )
        self.im = None
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        plt.show(block=False)

    @property
    def alive(self) -> bool:
        return self.plt.fignum_exists(self.fig.number)

    def _key(self, event) -> None:
        k = (event.key or "").lower()
        if k in ("right", " ", "enter", "down", "pagedown"):
            self.action = "next"
        elif k in ("left", "up", "pageup", "backspace"):
            self.action = "prev"
        elif k == "a":
            self.auto = 0 if self.auto else 3.0
            self.action = "next" if self.auto else None
        elif k in ("q", "escape"):
            self.action = "quit"

    def show(self, path: Path, i: int, n: int, title: str, phase: str, colour: str, detail: str) -> None:
        img = np.asarray(Image.open(path).convert("RGB"))
        if self.im is None or self.im.get_array().shape != img.shape:
            self.ax_out.clear()
            self.ax_out.set_axis_off()
            self.im = self.ax_out.imshow(img)
        else:
            self.im.set_data(img)
        self.phase.set_text(phase)
        self.phase.set_color(colour)
        self.title.set_text(f"Étape {i}/{n} · {title}" + (f"   —   {detail}" if detail else ""))
        self.ax_out.set_title(path.name, fontsize=11, color="#717686")
        self.fig.canvas.draw_idle()

    def wait(self) -> str:
        self.action, t0 = None, time.monotonic()
        while self.alive and self.action is None:
            self.plt.pause(0.05)
            if self.auto and time.monotonic() - t0 >= self.auto:
                return "next"
        return self.action or "closed"


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="BCCD id or image path")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo")
    parser.add_argument("--auto", type=float, default=0, help="seconds per step (0 = keys)")
    parser.add_argument("--no-window", action="store_true", help="terminal only, no image window")
    args = parser.parse_args()

    cfg = load_config()
    image_id, path = resolve_image(args.image, cfg)
    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    banner("Détection et analyse des cellules sanguines · morphologie mathématique")
    print(f"  Image   : {BOLD}{image_id}{RESET}  ({path})")
    print(f"  Sortie  : {BOLD}{out}{RESET}  (une image par étape, ajoutée au fur et à mesure)")

    img = load_image(str(path))
    print(f"\n  {DIM}Chargement du modèle…{RESET}", flush=True)
    model = load_model(PROJECT_ROOT / cfg["classifier"]["model_file"])
    print(f"  {DIM}Calcul des 40 étapes…{RESET}", end=" ", flush=True)
    t0 = time.perf_counter()
    res = run(img, cfg, model=model)
    seconds = time.perf_counter() - t0
    print(f"{GREEN}terminé en {seconds:.1f} s{RESET}  (r = {res.r:.1f} px)")

    images = dict(res.analysis.seg.steps)
    if res.analysis.seg.scale is not None:
        images["scale_coarse_mask"] = res.analysis.seg.scale.mask
    images.update(pipeline_step_images(res, cfg))
    details = step_details(res)
    r_ref = reference_radius(image_id, cfg)

    viewer = None
    if not args.no_window:
        try:
            viewer = Viewer(img, image_id, args.auto)
            print(f"\n  {CYAN}Fenêtre ouverte : cliquez dessus puis → / Espace pour avancer.{RESET}")
        except Exception as exc:  # no display available: terminal only
            print(f"  {RED}Pas de fenêtre graphique ({exc}) : mode terminal.{RESET}")

    n, written, i, current_phase = len(ALL_STEPS), {}, 1, None
    while i <= n:
        name = ALL_STEPS[i - 1][0]
        title, what, morpho = TEXT[name]
        phase, colour = phase_of(i)
        first = i not in written
        if first:
            written[i] = write_step(i, name, images, res, out, r_ref, image_id)
            if phase != current_phase:
                current_phase = phase
                print(f"\n{rgb(colour)}{BOLD}▌ {phase}{RESET}")
            tag = f"{CYAN}◆{RESET}" if morpho else f"{DIM}·{RESET}"
            extra = f"  {GREEN}{details[name]}{RESET}" if name in details else ""
            print(f"  {tag} {BOLD}[{i:02d}/{n}] {title}{RESET} — {what}{extra}")
            print(f"        {DIM}→ demo/{written[i].name}{RESET}", flush=True)
        if viewer is None or not viewer.alive:
            i += 1
            continue
        viewer.show(written[i], i, n, title, phase, colour, details.get(name, ""))
        action = viewer.wait()
        if action == "prev" and i > 1:
            i -= 1
            print(f"        {DIM}↩ retour à l'étape {i}{RESET}")
        elif action == "quit":
            viewer.plt.close(viewer.fig)
            i += 1
        else:
            i += 1

    # final results
    final = out / "resultat_final.png"
    save_image(final, res.overlay(show_ids=True))
    res.features.to_csv(out / "cellules.csv", index=False)
    s = res.summary()
    pd.DataFrame([{"image": image_id, **s, "seconds": round(seconds, 2)}]).to_csv(
        out / "resume.csv", index=False
    )

    banner("Résultat")
    c = res.counts
    print(f"  Rayon d'un GR (granulométrie) : {BOLD}{res.r:.1f} px{RESET}")
    print(
        f"  {RED}Globules rouges{RESET} : {BOLD}{c['RBC']}{RESET}"
        + (
            f"  (avec amas comptés par aire : {c['RBC_with_clumps']})"
            if c["RBC_with_clumps"] != c["RBC"]
            else ""
        )
    )
    print(
        f"  {rgb('#9b85ff')}Globules blancs{RESET} : {BOLD}{c['WBC']}{RESET}"
        + (
            f"  (rapport N/C moyen {s['wbc_nc_ratio_mean']:.2f})"
            if np.isfinite(s["wbc_nc_ratio_mean"])
            else ""
        )
    )
    print(f"  {GREEN}Plaquettes{RESET}      : {BOLD}{c['Platelet']}{RESET}")
    if np.isfinite(s.get("rbc_area_cv", np.nan)):
        print(
            f"  Anisocytose (CV de l'aire des GR) : {s['rbc_area_cv']:.2f}"
            f"   ·   GR signalés : {s['rbc_flagged']}"
        )
    print(f"  Temps de calcul : {seconds:.1f} s")
    print(f"\n  Dossier {BOLD}{out}{RESET} :")
    print(f"    {len(written)} étapes (01-… à {n:02d}-…), resultat_final.png, cellules.csv, resume.csv\n")

    if viewer is not None and viewer.alive:
        viewer.show(
            final,
            n,
            n,
            "Résultat final",
            "Résultat",
            "#c0283f",
            "{RBC} GR · {WBC} GB · {Platelet} plaquette(s)".format(**c),
        )
        viewer.title.set_text(
            "Résultat final — " + "{RBC} GR · {WBC} GB · {Platelet} plaquette(s)".format(**c)
        )
        viewer.help.set_text("Fermez la fenêtre pour terminer")
        viewer.fig.canvas.draw_idle()
        while viewer.alive:
            viewer.plt.pause(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
