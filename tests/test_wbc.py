import numpy as np
from skimage import morphology
from skimage.draw import disk

from bloodcells.config import load_config
from bloodcells.preprocessing import grey_morphology
from bloodcells.wbc import find_nuclei, morphological_gradient, nc_ratio, wbc_cells

CFG = load_config()
R = 30


def test_grey_morphology_matches_skimage():
    x = np.random.default_rng(0).random((80, 90))
    se = morphology.disk(5)
    import cv2

    ours = grey_morphology(x, cv2.MORPH_CLOSE, 5)
    ref = morphology.closing(x, se)
    assert np.abs(ours - ref)[6:-6, 6:-6].max() < 1e-6
    grad = morphological_gradient(x)
    ref_g = morphology.dilation(x, morphology.disk(1)) - morphology.erosion(x, morphology.disk(1))
    assert np.abs(grad - ref_g)[1:-1, 1:-1].max() < 1e-6


def test_lobes_of_one_nucleus_are_grouped_and_small_specks_ignored():
    purple = np.zeros((200, 200), bool)
    for c in ((100, 85), (100, 115), (80, 100)):  # three lobes 2-3 px apart
        purple[disk(c, 13, shape=purple.shape)] = True
    purple[disk((20, 20), 4, shape=purple.shape)] = True  # platelet-sized speck
    nuclei = find_nuclei(purple, R, CFG)
    assert nuclei.max() == 1
    assert nuclei[100, 85] == nuclei[100, 115] == nuclei[80, 100] == 1
    assert nuclei[20, 20] == 0


def test_gradient_watershed_finds_cytoplasm_and_stops_at_touching_rbc():
    shape = (240, 300)
    green = np.full(shape, 0.9)
    purple = np.zeros(shape)
    cell = np.zeros(shape, bool)
    cell[disk((120, 110), 48, shape=shape)] = True  # WBC
    green[cell] = 0.72  # pale cytoplasm
    rbc = np.zeros(shape, bool)
    rbc[disk((120, 185), 30, shape=shape)] = True  # RBC touching the WBC on the right
    green[rbc] = 0.45
    nuc = np.zeros(shape, bool)
    nuc[disk((120, 110), 28, shape=shape)] = True
    green[nuc], purple[nuc] = 0.3, 1.0
    purple[cell & ~nuc] = 0.4
    cell_mask = cell | rbc
    res = wbc_cells(green, purple, nuc.astype(np.int32), cell_mask, R, CFG, cells=1 - green)
    found = res.cells == 1
    dice = 2 * (found & cell).sum() / (found.sum() + cell.sum())
    assert dice > 0.9
    assert (found & rbc & ~cell).sum() < 0.05 * rbc.sum()  # the RBC is not absorbed


def test_nc_ratio_and_uncertain_cytoplasm():
    assert nc_ratio(300, 1000) == (300 / 700, False)
    ratio, uncertain = nc_ratio(980, 1000)
    assert np.isnan(ratio) and uncertain


def test_no_nucleus_gives_no_wbc():
    z = np.zeros((50, 50))
    res = wbc_cells(z, z, np.zeros((50, 50), np.int32), np.zeros((50, 50), bool), R, CFG)
    assert res.cells.max() == 0
