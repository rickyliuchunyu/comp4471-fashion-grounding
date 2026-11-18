"""Single CPU request. stdin contains normalized RGB PNG, stdout contains predictions."""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_api

from fashion_grounding.data.schema import sha256
from fashion_grounding.segmentation.export_candidates import validate_checkpoint_metadata
from fashion_grounding.segmentation.model import build_model
from fashion_grounding.segmentation.predict import predict_instances


def infer():
    root = Path.cwd()
    release = json.loads(sys.argv[1])
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
    model.eval()
    with torch.inference_mode():
        predictions = predict_instances(rgb, model, dict(config['decoder'], image_id=1, score_threshold=model_info['threshold']))
    instances = []
    for pred in predictions:
        mask = mask_api.encode(np.asfortranarray(pred.mask, dtype=np.uint8))
        mask['counts'] = mask['counts'].decode('ascii')
        instances.append(dict(category_id=pred.category_id, bbox_xyxy=pred.bbox_xyxy,
                              decoder_query_id=pred.decoder_query_id, score=pred.score, rle=mask))
    return dict(model=model_info, size=[rgb.shape[1], rgb.shape[0]], instances=instances)


if __name__ == '__main__':
    with contextlib.redirect_stdout(sys.stderr):
        result = infer()
    json.dump(result, sys.stdout)
