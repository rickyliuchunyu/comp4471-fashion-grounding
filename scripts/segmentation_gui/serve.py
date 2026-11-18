"""Serve the frozen dev gallery on loopback and process one uploaded picture at a time."""
import argparse
import base64
import functools
import hashlib
import io
import json
import secrets
import shlex
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from pycocotools import mask as mask_api

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = REPO / 'outputs/segmentation_gui'
MAX_BYTES = 20 * 1024 * 1024
RELEASE = json.loads((REPO / 'configs/artifacts/segmentation_dev_candidates.json').read_text())
GALLERY = LOCAL / 'releases' / RELEASE['release_id']
MODEL = dict(release_id=RELEASE['release_id'], checkpoint=RELEASE['checkpoint']['sha256'],
             completed_updates=RELEASE['checkpoint']['completed_updates'], threshold=RELEASE['selected_threshold'])
CATEGORIES = json.loads((REPO / 'configs/categories.json').read_text())['categories']
UPLOAD = threading.Lock()
TOKEN = secrets.token_urlsafe(32)


def verified(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('文件校验失败')
    return data


def png_url(data):
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def display_instance(candidate, rle):
    x0, y0, x1, y1 = candidate['bbox_xyxy']
    crop = mask_api.decode(rle)[y0:y1, x0:x1]
    image = Image.fromarray(crop)
    image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG', transparency=0, bits=1)
    return dict(query=candidate['decoder_query_id'], category=candidate['category_id'], score=candidate['score'],
                box=candidate['bbox_xyxy'], mask=png_url(buffer.getvalue()))


class Gallery:
    def __init__(self):
        package = json.loads(verified(GALLERY / 'candidates_predicted.json', RELEASE['candidate_manifest']['sha256']))
        manifest = json.loads(verified(GALLERY / 'image_manifest.json', package['image_manifest']['sha256']))
        self.images = {row['image_id']: row for row in manifest['images']}
        self.results = {row['image_id']: row for row in package['images']}
        provenance = package['provenance']
        if (self.images.keys() != self.results.keys() or len(self.images) != RELEASE['dev']['images']
                or package['candidate_set_id'] != RELEASE['candidate_set_id']
                or provenance['source_cache_sha256'] != RELEASE['cache']['sha256']
                or provenance['score_threshold'] != MODEL['threshold']
                or provenance['checkpoint_sha256'] != MODEL['checkpoint']):
            raise ValueError('候选包与图片或模型版本不符')

    @functools.lru_cache(maxsize=24)
    def example(self, iid):
        row, result = self.images[iid], self.results[iid]
        if result['status'] == 'inference_error':
            raise ValueError('这张图片的原始推理记录有错误')
        instances = [display_instance(c, json.loads(verified(GALLERY / c['mask_reference'], c['mask_sha256'])))
                     for c in result['candidates']]
        return dict(model=MODEL, image='/photo/' + str(iid), size=[row['width'], row['height']], instances=instances)


def normalize(raw):
    with Image.open(io.BytesIO(raw)) as original:
        if original.format not in {'JPEG', 'PNG', 'WEBP'} or original.width * original.height > 60_000_000:
            raise ValueError('请选择 6000 万像素以内的 JPG、PNG 或 WebP 图片。')
        image = ImageOps.exif_transpose(original).convert('RGB')
        image.thumbnail((1440, 1440))
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        return buf.getvalue(), list(image.size)


def predict(data, size):
    settings = json.loads((LOCAL / 'ssh.json').read_text())
    server_root = Path(settings['root'])
    checkout = server_root / 'workspaces/segmentation'
    worker = (HERE / 'cpu_predict.py').read_text()
    command = ['timeout', '-k', '5', '120', 'nice', '-n', '10', 'env', 'CUDA_VISIBLE_DEVICES=',
               'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
               'PYTHONPATH=' + str(checkout / 'src'), str(server_root / 'envs/segmentation-v1/bin/python'),
               '-c', worker, json.dumps(dict(model=MODEL, checkpoint_path=RELEASE['checkpoint']['path'],
                                             export_config=RELEASE['export_config']))]
    remote = 'cd ' + shlex.quote(str(checkout)) + ' && ' + shlex.join(command)
    result = subprocess.run(settings['ssh'] + [remote], input=data, capture_output=True, timeout=145)
    if result.returncode:
        print(result.stderr.decode(errors='replace')[-1200:], file=sys.stderr, flush=True)
        raise RuntimeError('项目服务器 CPU 推理未完成，请检查 SSH 连接后重试。')
    response = json.loads(result.stdout)
    if response['model'] != MODEL or response['size'] != size:
        raise RuntimeError('返回的模型版本或图片尺寸不一致。')
    return dict(model=MODEL, image=png_url(data), size=size, instances=[display_instance(c, c['rle']) for c in response['instances']])


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, data, mime):
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local(self):
        return self.headers.get('Host') == self.server.authority

    def respond_json(self, status, obj):
        self.reply(status, json.dumps(obj, ensure_ascii=False).encode(), 'application/json; charset=utf-8')

    def do_GET(self):
        if not self.local():
            return self.send_error(403)
        static = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                  '/style.css': ('style.css', 'text/css; charset=utf-8')}
        if self.path in static:
            name, mime = static[self.path]
            return self.reply(200, (HERE / name).read_bytes(), mime)
        gallery = self.server.gallery
        if self.path == '/api/gallery':
            return self.respond_json(200, dict(ids=sorted(gallery.images), categories=CATEGORIES, token=TOKEN, model=MODEL))
        route, _, arg = self.path.rpartition('/')
        if route not in ('/api/example', '/photo') or not arg.isdigit() or int(arg) not in gallery.images:
            return self.send_error(404)
        iid = int(arg)
        try:
            if route == '/photo':
                im = gallery.images[iid]
                return self.reply(200, verified(GALLERY / im['file_name'], im['sha256']), 'image/jpeg')
            self.respond_json(200, gallery.example(iid))
        except (OSError, ValueError) as exc:
            self.respond_json(500, dict(error=str(exc)))

    def do_POST(self):
        if self.path != '/api/predict':
            return self.send_error(404)
        if not self.local() or self.headers.get('Origin') != 'http://' + self.server.authority or self.headers.get('X-Viewer-Token') != TOKEN:
            return self.send_error(403)
        length = self.headers.get('Content-Length', '')
        if not length.isdigit() or not 0 < int(length) <= MAX_BYTES:
            return self.respond_json(413, dict(error='请选择 20 MB 以内的图片。'))
        if not UPLOAD.acquire(blocking=False):
            return self.respond_json(409, dict(error='另一张图片正在分割，请稍后重试。'))
        try:
            data, size = normalize(self.rfile.read(int(length)))
            self.respond_json(200, predict(data, size))
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError):
            self.respond_json(400, dict(error='图片无法读取，请试用 6000 万像素以内的 JPG、PNG 或 WebP。'))
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            self.respond_json(503, dict(error=str(exc) if isinstance(exc, RuntimeError) else '暂时无法完成 CPU 推理，请检查服务器连接后重试。'))
        finally:
            UPLOAD.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    gallery = Gallery()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.gallery = gallery
    server.authority = '127.0.0.1:' + str(server.server_port)
    url = 'http://' + server.authority
    print(url, flush=True)
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == '__main__':
    main()
