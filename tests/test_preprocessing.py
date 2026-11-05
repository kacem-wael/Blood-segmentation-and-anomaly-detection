import numpy as np
from skimage import morphology
from skimage.draw import disk

from bloodcells.preprocessing import black_tophat, normalize, purple_channel
from bloodcells.scale import opening_by_disk


def test_purple_channel_ranks_purple_above_pink_and_background():
    img = np.zeros((3, 1, 3), np.uint8)
    img[0, 0] = (110, 60, 170)  # purple nucleus
    img[1, 0] = (210, 150, 160)  # pink RBC
    img[2, 0] = (225, 225, 220)  # background
    for kind in ("lab_b", "hsv_s"):
        p = purple_channel(img, kind)[:, 0]
        assert p[0] > p[1] and p[0] > p[2]


def test_black_tophat_flattens_uneven_background():
    h, w = 240, 320
    background = np.linspace(0.6, 0.95, w)[None, :].repeat(h, axis=0)  # strong lighting gradient
    img = background.copy()
    rr, cc = disk((120, 80), 20)
    img[rr, cc] -= 0.3  # dark cell on the dark side
    rr2, cc2 = disk((120, 240), 20)
    img[rr2, cc2] -= 0.3  # same cell on the bright side

    th = black_tophat(img, radius=50, downsample=2)
    cell_values = th[120, 80], th[120, 240]
    corners = th[:20, :20], th[:20, -20:]
    assert all(abs(v - 0.3) < 0.05 for v in cell_values)  # same response on both sides
    assert all(c.mean() < 0.03 for c in corners)  # background ~0 everywhere


def test_relative_tophat_corrects_multiplicative_vignetting():
    # Transmitted light: image = illumination x transmittance. The same cell (transmittance
    # 0.6) under bright and dim light must get the same relative top-hat (1 - 0.6 = 0.4),
    # while the plain top-hat follows the illumination.
    h, w = 240, 320
    light = np.linspace(0.5, 1.0, w)[None, :].repeat(h, axis=0)
    transmittance = np.ones((h, w))
    for center in ((120, 80), (120, 240)):
        rr, cc = disk(center, 20)
        transmittance[rr, cc] = 0.6
    img = light * transmittance

    rel = black_tophat(img, radius=50, downsample=2, relative=True)
    plain = black_tophat(img, radius=50, downsample=2)
    assert abs(rel[120, 80] - 0.4) < 0.03 and abs(rel[120, 240] - 0.4) < 0.03
    assert plain[120, 240] > 1.3 * plain[120, 80]  # light ratio 0.88 / 0.63 = 1.4


def test_normalize_range_and_constant_input():
    x = np.random.default_rng(0).normal(size=(50, 50))
    n = normalize(x)
    assert n.min() == 0 and n.max() == 1
    assert not normalize(np.ones((5, 5))).any()


def test_opening_by_disk_matches_skimage():
    rng = np.random.default_rng(1)
    mask = np.zeros((120, 160), bool)
    for _ in range(12):
        rr, cc = disk(tuple(rng.integers(10, 110, 2)), rng.integers(4, 18), shape=mask.shape)
        mask[rr, cc] = True
    for r in (2, 5, 9):
        fast = opening_by_disk(mask, r)
        ref = morphology.opening(mask, morphology.disk(r))
        assert not (fast & ~mask).any()  # an opening never adds pixels
        assert (fast ^ ref).sum() <= 0.01 * max(ref.sum(), 1)
