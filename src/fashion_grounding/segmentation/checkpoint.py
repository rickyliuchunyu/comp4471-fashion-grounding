"""Read the checksum-pinned Model Zoo numpy pickle with a narrow global allowlist."""
from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import pickle

import numpy as np
from numpy.core.multiarray import _reconstruct


class _NumpyCheckpointReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('collections', 'OrderedDict'): OrderedDict,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): _reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): _reconstruct,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported checkpoint global: {module}.{name}')
        return allowed[(module, name)]


def read_coco_checkpoint(path, *, expected_sha256, expected_size, md5_prefix):
    """Verify the exact bytes before parsing; this is not a general pickle loader."""
    path = Path(path)
    if not 0 < expected_size <= 256 * 1024**2 or path.stat().st_size != expected_size:
        raise ValueError('Checkpoint size mismatch or exceeds limit')
    # Parse the same immutable bytes that were hashed (no second path read).
    payload = path.read_bytes()
    digest, md5 = hashlib.sha256(payload).hexdigest(), hashlib.md5(payload).hexdigest()
    if digest != expected_sha256 or not md5.startswith(md5_prefix):
        raise ValueError('Checkpoint checksum mismatch; refusing deserialization')
    stream = io.BytesIO(payload)
    checkpoint = _NumpyCheckpointReader(stream).load()
    if stream.read(1) or type(checkpoint) is not dict or set(checkpoint) != {'model', '__author__'}:
        raise ValueError('Unexpected checkpoint structure or trailing data')
    if checkpoint['__author__'] != 'MaskFormer Model Zoo':
        raise ValueError('Unexpected checkpoint author field')
    state = checkpoint['model']
    if not isinstance(state, (dict, OrderedDict)) or not state:
        raise ValueError('Expected nonempty numpy model state')
    for key, value in state.items():
        if not isinstance(key, str) or type(value) is not np.ndarray or value.dtype != np.float32:
            raise ValueError('Expected float32 numpy parameter arrays')
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite checkpoint parameter: {key}')
    return state, dict(sha256=digest, md5=md5, bytes=len(payload),
                       author=checkpoint['__author__'], tensor_count=len(state),
                       verified_before_deserialization=True)
