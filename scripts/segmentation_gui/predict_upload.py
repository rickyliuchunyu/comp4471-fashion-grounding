"""Single upload request. stdin contains normalized RGB PNG, stdout contains predictions."""
import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api

from with_gpu_lock import GpuOccupied, idle_gpu

from fashion_grounding.data.schema import sha256
from fashion_grounding.segmentation.export_candidates import validate_checkpoint_metadata
from fashion_grounding.segmentation.model import build_model
from fashion_grounding.segmentation.predict import predict_instances


class GpuUnavailable(RuntimeError):
    pass


def select_device(mode, coordination, lease):
    if mode == 'cpu':
        return 'cpu', ''
    if mode not in {'auto', 'cuda'}:
        raise ValueError('Invalid device preference')
    try:
        lease.enter_context(idle_gpu(coordination))
        if torch.cuda.is_available():
            return 'cuda', ''
        reason = '服务器 CUDA 不可用'
    except (BlockingIOError, GpuOccupied):
        reason = '共享 GPU 正被占用'
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError):
        reason = '无法确认 GPU 可用'
    lease.close()
    if mode == 'cuda':
        raise GpuUnavailable(reason)
    return 'cpu', reason


def infer(release, device):
    root = Path.cwd()
    model_info = release['model']
    config_path = root / release['export_config']['path']
    if sha256(config_path) != release['export_config']['sha256']:
        raise ValueError('Export configuration digest mismatch')
    config = json.loads(config_path.read_text())
    checkpoint = Path(release['checkpoint_path'])
    if (config['checkpoint_sha256'] != model_info['checkpoint'] or sha256(checkpoint) != model_info['checkpoint']
            or config['completed_updates'] != model_info['completed_updates']
            or model_info['threshold'] not in config['threshold_options']):
        raise ValueError('Release and checkpoint differ')
    with Image.open(io.BytesIO(sys.stdin.buffer.read())) as image:
        rgb = np.array(image.convert('RGB'))
    torch.set_num_threads(2)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(state, config)
    model, _ = build_model(root.parents[1] / 'vendor/segmentation/Mask2Former', device='cpu')
    model.load_state_dict(state['model'], strict=True)
    del state
    model.to(device).eval()
    with torch.inference_mode():
        predictions = predict_instances(rgb, model, dict(config['decoder'], image_id=1, score_threshold=model_info['threshold']))
    instances = []
    for pred in predictions:
        mask = mask_api.encode(np.asfortranarray(pred.mask, dtype=np.uint8))
        mask['counts'] = mask['counts'].decode('ascii')
        instances.append(dict(category_id=pred.category_id, bbox_xyxy=pred.bbox_xyxy,
                              decoder_query_id=pred.decoder_query_id, score=pred.score, rle=mask))
    actual = model.device
    execution = dict(device=actual.type, name=torch.cuda.get_device_name(actual) if actual.type == 'cuda' else 'CPU')
    return dict(model=model_info, execution=execution, size=[rgb.shape[1], rgb.shape[0]], instances=instances)


if __name__ == '__main__':
    release = json.loads(sys.argv[1])
    mode = release['device']
    with contextlib.ExitStack() as lease, contextlib.redirect_stdout(sys.stderr):
        try:
            device, reason = select_device(mode, Path.cwd().parents[1] / 'coordination', lease)
        except GpuUnavailable as exc:
            result = dict(error=str(exc))
        else:
            if device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
            result = infer(release, device)
            result['execution']['reason'] = reason
    json.dump(result, sys.stdout)
