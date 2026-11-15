#!/usr/bin/env python3
"""
palm_protect.py - template protection for the palm matcher.

1. CANCELLABLE SEARCH TEMPLATE (ISO/IEC 24745 properties: renewable, unlinkable)
   The fixed-length palm vector (palm_embed.py) is hashed with Index-of-Max (IoM) hashing: K random projection sets
   of M directions, keyed by a secret; each hash value is the index of the largest projection (common/crypto/iom_urp.py).
   Only these indices are stored. A new key gives a new template that does not match the old one (renewable), and
   templates of the same palm under different keys look unrelated (unlinkable, measured as D_sys by
   benchmarks/embed_eval.py). Similarity = share of equal indices.
   Limits: IoM hashing discards magnitudes, but published attacks can approximate a palm vector from hashes when the
   key is also stolen; keep keys in a hardware keystore.

2. ENCRYPTED VERIFICATION TEMPLATE
   The full keypoint template (v3c, ~97 KB) needed for accurate 1:1 verification is encrypted at rest:
   Windows DPAPI (key managed by the operating system and bound to the Windows account) or, where the `cryptography`
   package is installed, AES-256-GCM with a key from the environment variable PALM_TEMPLATE_KEY (64 hex characters).
   It is decrypted only in memory, at match time.
"""
import hashlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.crypto.iom_urp import IndexOfMaxHasher  # noqa: E402

IOM_K = 512        # hashes per template
IOM_M = 16         # projections per hash; template = 512 indices, 1 byte each = 512 bytes
_HASHERS = {}


def key_seed(key):
    """32-bit seed from a secret key (bytes or str). Different key = different, unlinkable templates."""
    k = key.encode() if isinstance(key, str) else key
    return int.from_bytes(hashlib.sha256(b"palm-iom-v1" + k).digest()[:4], "little")


def _hasher(dim, key):
    s = key_seed(key)
    if (dim, s) not in _HASHERS:
        if len(_HASHERS) > 8:
            _HASHERS.clear()
        _HASHERS[dim, s] = IndexOfMaxHasher(embedding_dim=dim, num_hashes_k=IOM_K, subset_size_m=IOM_M, seed=s)
    return _HASHERS[dim, s]


def protect(vec, key):
    """Palm vector (unit, any length) -> cancellable template: IOM_K indices in 0..IOM_M-1 (uint8)."""
    return _hasher(len(vec), key).hash(np.asarray(vec, np.float32)).astype(np.uint8)


def protect_many(vecs, key):
    vecs = np.asarray(vecs, np.float32)
    return _hasher(vecs.shape[1], key).hash(vecs).astype(np.uint8)


def similarity(a, b):
    """Share of equal hash indices (0..1). Unrelated palms give about 1/IOM_M."""
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def similarity_matrix(probes, gallery):
    """(P x K) vs (G x K) uint8 -> P x G share of equal indices."""
    p, g = np.asarray(probes), np.asarray(gallery)
    return (p[:, None, :] == g[None, :, :]).mean(2)


# ----------------------------------------------------------------------------- encryption at rest
MAGIC_DPAPI, MAGIC_AES = b"PALMDP1", b"PALMAG1"


def _dpapi(data, protect_):
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    src, dst = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if protect_ else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None if not protect_ else "palm template", None, None, None, 0x1, ctypes.byref(dst))
    if not ok:
        raise OSError("DPAPI failed: the template was encrypted by another Windows account or is damaged")
    try:
        return ctypes.string_at(dst.pbData, dst.cbData)
    finally:
        kernel32.LocalFree(dst.pbData)


def _aes_key():
    k = os.environ.get("PALM_TEMPLATE_KEY", "")
    if len(k) != 64:
        raise RuntimeError("set PALM_TEMPLATE_KEY to 64 hex characters (a 256-bit key from your key store)")
    return bytes.fromhex(k)


def encrypt(data):
    """bytes -> encrypted bytes (DPAPI on Windows, else AES-256-GCM)."""
    if sys.platform == "win32":
        return MAGIC_DPAPI + _dpapi(data, True)
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM   # pip install cryptography
    nonce = os.urandom(12)
    return MAGIC_AES + nonce + AESGCM(_aes_key()).encrypt(nonce, data, MAGIC_AES)


def decrypt(blob):
    if blob.startswith(MAGIC_DPAPI):
        return _dpapi(blob[len(MAGIC_DPAPI):], False)
    if blob.startswith(MAGIC_AES):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        b = blob[len(MAGIC_AES):]
        return AESGCM(_aes_key()).decrypt(b[:12], b[12:], MAGIC_AES)
    raise ValueError("not an encrypted palm template")


def save_encrypted(path, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(encrypt(buf.getvalue()))
    os.replace(tmp, path)


def load_encrypted(path):
    with open(path, "rb") as f:
        data = decrypt(f.read())
    with np.load(io.BytesIO(data)) as z:
        return {k: z[k] for k in z.files}
