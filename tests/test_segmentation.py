import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from bloodcells.config import load_config
from bloodcells.segmentation import (
    STEPS,
    clean_mask,
    closing_by_disk,
    fill_round_holes,
    segment,
    threshold_cells,
)

CFG = load_config()


def ring(shape, center, outer, inner):
    mask = np.zeros(shape, bool)
    mask[disk(center, outer, shape=shape)] = True
    mask[disk(center, inner, shape=shape)] = False
    return mask


def test_disc_on_background_gives_one_object_of_the_right_area():
    img = np.zeros((200, 200))
    img[disk((100, 100), 30)] = 1.0
    img += np.random.default_rng(0).normal(0, 0.05, img.shape)
    mask = clean_mask(threshold_cells(img), r=30, cfg=CFG)
    _, n = ndi.label(mask)
    assert n == 1
    assert abs(mask.sum() / (np.pi * 30**2) - 1) < 0.05


def test_pale_centre_ring_is_filled():
    mask = ring((120, 120), (60, 60), 30, 12)
    filled = fill_round_holes(mask, r=30)
    assert filled[60, 60] and filled.sum() > mask.sum()


@pytest.mark.parametrize(
    "centers",
    [
        [(70, 70), (70, 128), (120, 99)],  # 3 discs, neighbours 58 px apart (slight overlap)
        [(60, 60), (60, 116), (116, 60), (116, 116)],  # 4 discs in a square
    ],
)
def test_gap_between_touching_discs_is_not_filled(centers):
    r = 30
    mask = np.zeros((200, 200), bool)
    for c in centers:
        mask[disk(c, r, shape=mask.shape)] = True
    gap = (int(np.mean([c[0] for c in centers])), int(np.mean([c[1] for c in centers])))
    assert not mask[gap]  # the discs enclose a background gap
    assert not fill_round_holes(mask, r)[gap]


def test_large_round_hole_is_not_filled():
    # A round hole bigger than 0.5 pi r^2 is background between cells, not a pale centre.
    mask = ring((200, 200), (100, 100), 80, 40)
    assert not fill_round_holes(mask, r=30)[100, 100]


def test_segment_records_every_step_in_pipeline_order():
    # Synthetic smear: pink cells with pale centres and one purple nucleus on a bright background.
    img = np.full((240, 320, 3), (225, 225, 220), np.uint8)
    for c in [(60, 60), (60, 160), (170, 90), (170, 250)]:
        img[disk(c, 30, shape=img.shape[:2])] = (200, 120, 140)
        img[disk(c, 10, shape=img.shape[:2])] = (215, 175, 180)
    img[disk((80, 260), 25, shape=img.shape[:2])] = (110, 60, 170)
    seg = segment(img, CFG)
    figures = {"scale_granulometry", "segmentation_overlay"}  # drawn by the exporter
    expected = [name for name, _ in STEPS if name not in figures]
    assert list(seg.steps) == expected
    assert seg.steps["remove_small_objects"] is seg.cell_mask
    assert seg.steps["purple_remove_small_objects"] is seg.purple_mask
    assert seg.purple_mask[80, 260] and not seg.purple_mask[60, 60]
    assert seg.cell_mask[60, 60]  # pale centre filled
    assert "scale_coarse_mask" not in segment(img, CFG, r=30).steps


def test_closing_by_disk_keeps_border_cells_and_fills_small_gaps():
    mask = np.zeros((60, 60), bool)
    mask[:, :20] = True  # touches the left border
    mask[:, 22:40] = True  # 2 px gap
    closed = closing_by_disk(mask, 2)
    assert closed[:, :20].all()  # border not eroded
    assert closed[30, 20:22].all()  # gap closed
    assert not closed[:, 45:].any()


def test_auto_purple_threshold_switches_when_every_cell_is_stained():
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from synthetic import smear

    from bloodcells.segmentation import purple_coverage

    img = smear(r=30)
    # Make the RBCs bluish: the whole population is stained, only the nucleus is dark purple.
    rbc = (np.abs(img.astype(int) - (205, 120, 118)).sum(-1) < 20) | (
        np.abs(img.astype(int) - (218, 160, 155)).sum(-1) < 20
    )
    img[rbc] = (150, 130, 185)
    seg = segment(img, CFG, r=30)
    nucleus = np.abs(img.astype(int) - (105, 55, 165)).sum(-1) < 25
    assert purple_coverage(seg.purple_mask, seg.cell_mask) < 0.5
    assert seg.purple_mask[nucleus].mean() > 0.9 and seg.purple_mask[rbc].mean() < 0.1
