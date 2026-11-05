import numpy as np
import pytest
from skimage.draw import disk

from bloodcells.config import load_config, with_overrides
from bloodcells.separation import merge_small_fragments, separate
from bloodcells.visualization import label_contours

CFG = load_config()
R = 30


def discs(shape, centres, radius=R):
    mask = np.zeros(shape, bool)
    for c in centres:
        mask[disk(c, radius, shape=shape)] = True
    return mask


def n_labels(labels):
    return len(np.unique(labels[labels > 0]))


def test_one_disc_gives_one_label():
    assert n_labels(separate(discs((120, 120), [(60, 60)]), R, CFG).labels) == 1


@pytest.mark.parametrize("method", ["hmaxima", "peaks"])
def test_two_discs_1_5_r_apart_give_two_labels(method):
    mask = discs((120, 200), [(60, 60), (60, 60 + int(1.5 * R))])
    sep = separate(mask, R, CFG, method=method)
    assert n_labels(sep.labels) == 2
    # The cut is near the middle: both halves have about the same area.
    areas = np.bincount(sep.labels.ravel())[1:]
    assert areas.min() / areas.max() > 0.8


def test_empty_mask_gives_no_label():
    sep = separate(np.zeros((50, 50), bool), R, CFG)
    assert sep.labels.max() == 0


def test_component_without_marker_keeps_a_label():
    # Tiny disc (radius 4 < h = 0.2 r = 6): no h-maximum, but it must not be dropped.
    mask = discs((120, 120), [(60, 60)], radius=4)
    cfg = with_overrides(CFG, {("separation", "min_fragment_area"): 0.0})
    sep = separate(mask, R, cfg)
    assert n_labels(sep.labels) == 1
    assert sep.n_orphans == 1


def test_without_watershed_touching_discs_stay_merged():
    mask = discs((120, 200), [(60, 60), (60, 60 + int(1.5 * R))])
    assert n_labels(separate(mask, R, CFG, watershed=False).labels) == 1


def test_small_fragment_is_merged_into_its_neighbour():
    labels = np.zeros((20, 20), np.int32)
    labels[:, :15] = 1
    labels[:, 15:] = 2  # 20 x 5 = 100 px
    merged, n = merge_small_fragments(labels, min_area=150)
    assert n == 1 and n_labels(merged) == 1


def test_label_contours_draw_the_line_between_touching_labels():
    labels = np.zeros((20, 20), np.int32)
    labels[5:15, 2:10] = 1
    labels[5:15, 10:18] = 2
    edge = label_contours(labels)
    assert edge[10, 9] or edge[10, 10]  # the split line inside the merged blob
    assert not edge[0, 0]  # background far from any object


def test_skiz_and_ultimate_erosion():
    from bloodcells.separation import skiz, ultimate_erosion_count

    markers = np.zeros((40, 80), np.int32)
    markers[20, 20], markers[20, 60] = 1, 2
    zones, lines = skiz(markers)
    assert set(np.unique(zones)) == {1, 2}
    assert lines[:, 39:41].any() and not lines[:, :30].any()  # boundary halfway between markers
    clump = discs((120, 260), [(60, 60), (60, 60 + int(1.5 * R)), (60, 60 + 3 * R)])
    assert ultimate_erosion_count(clump, R) == 3
