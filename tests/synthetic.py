"""Synthetic smear images shared by the tests (drawn discs, known ground truth)."""

from __future__ import annotations

import numpy as np
from skimage.draw import disk

BACKGROUND = (228, 226, 215)
RBC = (205, 120, 118)
RBC_CENTRE = (218, 160, 155)
CYTOPLASM = (200, 186, 206)  # pale lilac: clearly less stained than the nucleus
NUCLEUS = (105, 55, 165)
PLATELET = (120, 70, 175)


def smear(
    shape=(360, 480),
    rbcs=((90, 90), (90, 300), (250, 120), (250, 380)),
    r: int = 30,
    wbc=(180, 240),
    platelets=((40, 430), (320, 40)),
    touching=((300, 250),),
    seed: int = 0,
) -> np.ndarray:
    """RGB uint8 image: RBC discs with pale centres, one WBC, platelets, pairs of touching RBCs.

    Each centre in ``touching`` gets two RBCs 1.5 r apart (horizontal).
    """
    img = np.zeros(shape + (3,), float)
    img[:] = BACKGROUND
    centres = list(rbcs)
    for cy, cx in touching:
        centres += [(cy, cx - int(0.75 * r)), (cy, cx + int(0.75 * r))]
    for c in centres:
        img[disk(c, r, shape=shape)] = RBC
        img[disk(c, int(0.35 * r), shape=shape)] = RBC_CENTRE
    if wbc is not None:
        img[disk(wbc, int(1.6 * r), shape=shape)] = CYTOPLASM
        img[disk(wbc, int(1.0 * r), shape=shape)] = NUCLEUS
    for c in platelets:
        img[disk(c, max(3, int(0.2 * r)), shape=shape)] = PLATELET
    img += np.random.default_rng(seed).normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
