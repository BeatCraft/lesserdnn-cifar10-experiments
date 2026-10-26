"""Codebooks and exact uniform initialization; no Metal or dataset required."""
import hashlib
import numpy as np

CODEBOOKS = {
    8: np.asarray([-1,-.5,-.25,-.125,.125,.25,.5,1], dtype=np.float32),
    9: np.asarray([-1,-.5,-.25,-.125,0,.125,.25,.5,1], dtype=np.float32),
    10: np.asarray([-1,-.5,-.25,-.125,-.0625,.0625,.125,.25,.5,1], dtype=np.float32),
    11: np.asarray([-1,-.5,-.25,-.125,-.0625,0,.0625,.125,.25,.5,1], dtype=np.float32),
}
SHAPES = {1:(32,27), 3:(64,288), 5:(128,576), 6:(10,8192)}


def initial_indexes(seed, quant_set):
    levels = CODEBOOKS[quant_set]
    allowed = np.flatnonzero(np.abs(levels) <= .125)
    generator = np.random.default_rng(seed)
    return {key:allowed[generator.integers(0,len(allowed),size=shape)].astype(np.uint8)
            for key,shape in SHAPES.items()}


def digest(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(str((key,a.shape,str(a.dtype))).encode())
        h.update(a.tobytes())
    return h.hexdigest()
