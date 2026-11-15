#!/usr/bin/env python3
"""
palm_embed.py - fixed-length palm vector for 1:N search (shortlist, then LightGlue re-ranking) and for cancellable
template protection (IoM hashing, palm_protect.py). palm_match.py is not changed by this file.

Two vectors, both from the aligned 512 x 512 palm crop that every matcher already computes:

  hol    Histogram of Oriented Lines (after Jia et al., IEEE T-SMC 2014): dominant line orientation and line strength
         per pixel from a 12-orientation Gabor bank on the 128 x 128 crop, histograms over overlapping 16 x 16 blocks,
         block-normalised. Needs no training at all.
  disk   DISK descriptors of the v3c template itself (no extra network run), aggregated with VLAD on a 2 x 2 spatial
         grid, power- and L2-normalised, projected to 256 dimensions. The codebook and the projection are fitted on DEV
         photos without identity labels (models/palm_embed_disk.npz, written by benchmarks/embed_eval.py --fit).

Similarity = cosine (dot product of unit vectors).
"""
import os

import cv2
import numpy as np

import palm_match as pm

HERE = os.path.dirname(os.path.abspath(__file__))
DISK_MODEL = os.path.join(HERE, "models", "palm_embed_disk.npz")

# ----------------------------------------------------------------------------- HOL
HOL_SIZE, HOL_ORI, HOL_BLOCK, HOL_STRIDE = 128, 12, 16, 8


def _hol_bank():
    bank = []
    for k in range(HOL_ORI):
        kern = cv2.getGaborKernel((17, 17), sigma=2.8, theta=np.pi * k / HOL_ORI, lambd=9.0, gamma=0.5, psi=0,
                                  ktype=cv2.CV_32F)
        bank.append(kern - kern.mean())
    return bank


_HOL_BANK = _hol_bank()


def hol(gray):
    """Histogram of Oriented Lines of an aligned palm crop -> unit vector (15 x 15 blocks x 12 orientations)."""
    g = cv2.resize(gray, (HOL_SIZE, HOL_SIZE), interpolation=cv2.INTER_AREA)
    x = pm._normalise(g)                                               # removes uneven lighting
    resp = np.stack([cv2.filter2D(x, cv2.CV_32F, k) for k in _HOL_BANK])
    ori = resp.argmin(0)                                               # palm lines are dark: strongest negative
    energy = np.maximum(-resp.min(0), 0)
    onehot = np.zeros((HOL_ORI, HOL_SIZE, HOL_SIZE), np.float32)
    np.put_along_axis(onehot, ori[None], energy[None], 0)
    # block sums via an integral image: blocks of HOL_BLOCK at stride HOL_STRIDE
    ii = np.pad(onehot.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    s = np.arange(0, HOL_SIZE - HOL_BLOCK + 1, HOL_STRIDE)
    e = s + HOL_BLOCK
    h = ii[:, e][:, :, e] - ii[:, s][:, :, e] - ii[:, e][:, :, s] + ii[:, s][:, :, s]   # (ori, by, bx)
    h = np.sqrt(np.maximum(h.transpose(1, 2, 0), 0))                   # Hellinger-style compression (float error < 0)
    h /= np.linalg.norm(h, axis=2, keepdims=True) + 1e-6               # per-block normalisation (lighting)
    v = h.ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


# ----------------------------------------------------------------------------- DISK-VLAD
VLAD_K, VLAD_GRID, VLAD_DIM = 32, 2, 256
_DISK = None


def kmeans(x, k, iters=25, seed=0):
    """Spherical k-means (unit vectors) in numpy."""
    rng = np.random.RandomState(seed)
    c = x[rng.choice(len(x), k, replace=False)].copy()
    for _ in range(iters):
        a = (x @ c.T).argmax(1)
        for j in range(k):
            m = x[a == j]
            c[j] = m.mean(0) if len(m) else x[rng.randint(len(x))]
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
    return c.astype(np.float32)


def _spatial_weights(kp):
    """Bilinear weights of each keypoint to the VLAD_GRID x VLAD_GRID cells (tolerates alignment shifts)."""
    g = VLAD_GRID
    u = np.clip(kp / (pm.BIG / g) - 0.5, 0, g - 1)                       # position in cell-centre units
    i0 = np.floor(u).astype(int)
    f = u - i0
    i1 = np.minimum(i0 + 1, g - 1)
    w = np.zeros((len(kp), g * g), np.float32)
    for (iy, fy) in ((i0[:, 1], 1 - f[:, 1]), (i1[:, 1], f[:, 1])):
        for (ix, fx) in ((i0[:, 0], 1 - f[:, 0]), (i1[:, 0], f[:, 0])):
            np.add.at(w, (np.arange(len(kp)), iy * g + ix), fy * fx)
    return w


def vlad_raw(kp, des, codebook):
    """Unprojected spatial VLAD of one template (float positions in the 512 crop, unit descriptors)."""
    a = (des @ codebook.T).argmax(1)
    res = des - codebook[a]
    w = _spatial_weights(kp)
    v = np.zeros((VLAD_GRID * VLAD_GRID, len(codebook), des.shape[1]), np.float32)
    for c in range(VLAD_GRID * VLAD_GRID):
        np.add.at(v[c], a, res * w[:, c:c + 1])
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-6               # intra-normalisation
    v = v.ravel()
    v = np.sign(v) * np.sqrt(np.abs(v))                                # power normalisation
    return v / (np.linalg.norm(v) + 1e-12)


def fit_disk(templates, path=DISK_MODEL, seed=0):
    """Fit codebook + PCA projection on unlabelled templates [(kp float, des unit)], save to path."""
    rng = np.random.RandomState(seed)
    allx = np.concatenate([d[rng.choice(len(d), min(len(d), 200), replace=False)] for _, d in templates])
    codebook = kmeans(allx, VLAD_K, seed=seed)
    raw = np.stack([vlad_raw(k, d, codebook) for k, d in templates])
    mean = raw.mean(0)
    _, _, vt = np.linalg.svd(raw - mean, full_matrices=False)
    proj = vt[:VLAD_DIM].astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, codebook=codebook, mean=mean.astype(np.float32), proj=proj, n_fit=len(templates))
    global _DISK
    _DISK = None
    return path


def disk_model(path=DISK_MODEL):
    global _DISK
    if _DISK is None or _DISK[0] != path:
        z = np.load(path)
        _DISK = (path, z["codebook"], z["mean"], z["proj"])
    return _DISK[1:]


def disk_vlad(template, path=DISK_MODEL):
    """Unit vector from a v3c template (compact or unpacked)."""
    kp, des = template
    if des.dtype == np.uint8:
        import palm_match_compact as pmc
        kp, des = pmc.unpack(template)
    codebook, mean, proj = disk_model(path)
    v = proj @ (vlad_raw(kp, des, codebook) - mean)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def similarity(a, b):
    return float(np.dot(a, b))
