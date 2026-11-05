import numpy as np
from skimage.draw import disk

from bloodcells.scale import estimate_rbc_radius, pattern_spectrum, smooth_spectrum


def discs(shape, radius, n, rng, big=None):
    """Mask with ``n`` non-overlapping discs of ``radius`` (+ optional big discs as (radius, count))."""
    mask = np.zeros(shape, bool)
    sizes = [radius] * n + ([big[0]] * big[1] if big else [])
    placed = []
    for r in sorted(sizes, reverse=True):
        for _ in range(500):
            c = rng.integers(r + 1, np.array(shape) - r - 1)
            if all(np.hypot(*(c - p)) > r + q + 2 for p, q in placed):
                placed.append((c, r))
                rr, cc = disk(tuple(c), r, shape=shape)
                mask[rr, cc] = True
                break
    return mask


def test_pattern_spectrum_single_disc():
    mask = np.zeros((100, 100), bool)
    rr, cc = disk((50, 50), 20)
    mask[rr, cc] = True
    radii, spectrum = pattern_spectrum(mask, 30)
    assert spectrum.sum() == mask.sum()  # everything is removed by r_max
    assert abs(int(np.argmax(spectrum)) + 0.5 - 20) <= 1  # removed between radius 19 and 20


def test_radius_of_discs_and_calibration():
    rng = np.random.default_rng(0)
    mask = discs((300, 400), 15, 25, rng)
    est = estimate_rbc_radius(mask, r_max=60, r_default=50)
    assert est.ok and abs(est.r - 15) <= 1
    calibrated = estimate_rbc_radius(mask, r_max=60, r_default=50, calibration=1.2)
    assert np.isclose(calibrated.r, 1.2 * est.r)


def test_radius_scales_with_magnification():
    rng = np.random.default_rng(1)
    small = estimate_rbc_radius(discs((200, 260), 10, 20, rng), r_max=50, r_default=50).r
    large = estimate_rbc_radius(discs((400, 520), 20, 20, rng), r_max=100, r_default=50).r
    assert abs(large / small - 2) < 0.15


def test_one_big_cell_does_not_win_over_many_small_ones():
    # A WBC (radius 1.9 r) has ~3.6x the area of an RBC; with 10 RBCs the area-weighted
    # spectrum is already close, with the weighting the RBC peak must win clearly.
    rng = np.random.default_rng(2)
    mask = discs((300, 400), 15, 10, rng, big=(28, 1))
    assert abs(estimate_rbc_radius(mask, r_max=60, r_default=50).r - 15) <= 1.5


def test_empty_mask_falls_back_to_default():
    est = estimate_rbc_radius(np.zeros((50, 50), bool), r_max=20, r_default=42)
    assert not est.ok and est.r == 42


def test_smooth_spectrum_is_scale_free():
    # One object at rho = 19.5 and the same object magnified (area x (39.5 / 19.5)^2):
    # with weight 2 (object count) both peaks must have the same height and width.
    s = np.zeros(200)
    s[19], s[39] = 1000.0, 1000.0 * (39.5 / 19.5) ** 2
    grid, density = smooth_spectrum(s, weight=2, n_grid=4000)
    peaks = []
    for part in (grid < 28, grid >= 28):
        g, d = grid[part], density[part]
        half = g[d > d.max() / 2]
        peaks.append((d.max(), (half[-1] - half[0]) / g[d.argmax()]))  # height, relative width
    (h1, w1), (h2, w2) = peaks
    assert np.isclose(h1, h2, rtol=0.01) and np.isclose(w1, w2, rtol=0.05)
