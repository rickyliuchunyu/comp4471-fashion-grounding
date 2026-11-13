"""Download the pinned COCO R50 artifact with TLS and both published integrity checks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from urllib.request import urlopen


def verify_file(path, expected_sha256, expected_size, md5_prefix):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
            md5.update(block)
    result = dict(bytes=Path(path).stat().st_size, sha256=sha.hexdigest(), md5=md5.hexdigest())
    if result['bytes'] != expected_size or result['sha256'] != expected_sha256:
        raise ValueError('Size/SHA256 mismatch; artifact must not be loaded')
    if not result['md5'].startswith(md5_prefix):
        raise ValueError('Official Model Zoo filename MD5 prefix mismatch')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', default='envs/segmentation/upstream.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--record', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.provenance).read_text(encoding='utf-8'))
    mirror = config['checkpoint_mirror']
    output, record = Path(args.output), Path(args.record)
    if record.exists():
        raise FileExistsError(record)
    expected_size = mirror['bytes']
    if expected_size > 256 * 1024 * 1024 or not mirror['url'].startswith('https://'):
        raise ValueError('Download exceeds bounded artifact scope or is not HTTPS')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reused = output.exists()
    if not reused:
        if shutil.disk_usage(output.parent).free < expected_size + 1024**3:
            raise OSError('Insufficient free space with 1 GiB reserve')
        partial = output.with_name(output.name + f'.{os.getpid()}.part')
        try:
            with urlopen(mirror['url'], timeout=30) as response, partial.open('xb') as stream:
                if not response.url.startswith('https://'):
                    raise ValueError('Refusing a non-HTTPS redirect')
                if int(response.headers.get('Content-Length', expected_size)) != expected_size:
                    raise ValueError('Unexpected Content-Length')
                size = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    size += len(block)
                    if size > expected_size or time.monotonic()-started > 180:
                        raise ValueError('Download exceeded byte/time budget')
                    stream.write(block)
            result = verify_file(partial, mirror['sha256'], expected_size, config['checkpoint_md5_prefix'])
            os.link(partial, output)  # Atomic publish; never overwrite an existing file.
        finally:
            partial.unlink(missing_ok=True)
    result = verify_file(output, mirror['sha256'], expected_size, config['checkpoint_md5_prefix'])
    result.update(status='verified', reused=reused, official_url=config['checkpoint_url'],
                  download_url=mirror['url'], mirror_revision=mirror['revision'],
                  official_endpoint_status='HTTPS certificate expired in both local and server checks',
                  tls_verification=True, checked_at=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.monotonic()-started,
                  integrity_scope='Full SHA256 matches pinned mirror LFS object; MD5 prefix matches official Model Zoo filename. No publisher-signed SHA256 was available.')
    record.parent.mkdir(parents=True, exist_ok=True)
    with record.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
