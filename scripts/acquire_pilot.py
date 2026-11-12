"""Download a deterministic training-only sample, using HTTP ranges for ZIP members."""
import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import random
import sys
import zipfile

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.schema import decode_mask, read_json, require, sha256, write_json

ANNOTATIONS_URL = "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
IMAGES_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        r = self.session.head(url, timeout=60)
        r.raise_for_status()
        self.length = int(r.headers["Content-Length"])
        self.etag = r.headers["ETag"]
        self.pos = 0
        self.cache_start, self.cache = 0, b""
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.length + offset
        require(self.pos >= 0, "negative seek")
        return self.pos

    def read(self, size=-1):
        size = self.length - self.pos if size < 0 else min(size, self.length - self.pos)
        if size <= 0:
            return b""
        if not (self.cache_start <= self.pos and self.pos + size <= self.cache_start + len(self.cache)):
            end = min(self.length - 1, self.pos + max(size, 65536) - 1)
            r = self.session.get(self.url, headers={"Range": f"bytes={self.pos}-{end}", "If-Match": self.etag}, timeout=60, stream=True)
            with r:
                require(r.status_code == 206, "server must honor byte ranges")
                require(r.headers["Content-Range"] == f"bytes {self.pos}-{end}/{self.length}", "unexpected byte range")
                self.cache_start, self.cache = self.pos, r.content
            require(len(self.cache) == end - self.pos + 1, "truncated HTTP range")
            self.transferred += len(self.cache)
        offset = self.pos - self.cache_start
        data = self.cache[offset:offset + size]
        self.pos += len(data)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    require(1 <= args.count <= 200, "pilot count must be 1..200")
    repo = Path(__file__).resolve().parents[1]
    raw = repo / "data/raw/fashionpedia"
    raw.mkdir(parents=True, exist_ok=True)
    source = raw / "instances_attributes_train2020.json"
    if not source.exists() or not source.stat().st_size:
        with requests.get(ANNOTATIONS_URL, timeout=90, stream=True) as r:
            r.raise_for_status()
            partial = source.with_suffix(".partial")
            with partial.open("wb") as f:
                for block in r.iter_content(1024 * 1024):
                    f.write(block)
            read_json(partial)
            partial.replace(source)
    data = read_json(source)
    mapping = {r["source_id"]: r for r in read_json(repo / "configs/source_category_mapping.json")["fashionpedia"]}
    grouped = defaultdict(list)
    for ann in data["annotations"]:
        grouped[ann["image_id"]].append(ann)
    images = sorted(data["images"], key=lambda im: im["id"])
    random.Random(args.seed).shuffle(images)
    pool, rejected = [], Counter()
    # A small diversity-oriented interface sample, not a representative benchmark.
    for im in images:
        anns = grouped[im["id"]]
        if any(mapping.get(a["category_id"], {}).get("action") == "exclude_image" for a in anns):
            rejected["jumpsuit_image"] += 1
            continue
        whole = [a for a in anns if mapping.get(a["category_id"], {}).get("action") == "map"]
        if not 2 <= len(whole) <= 15:
            rejected["pilot_instance_count_preference"] += 1
            continue
        try:
            for a in whole:
                decode_mask(a["segmentation"], im["height"], im["width"])
        except (ValueError, TypeError, KeyError) as exc:
            rejected["invalid_mask_image"] += 1
            continue
        counts = Counter(mapping[a["category_id"]]["project_category_id"] for a in whole if not a["iscrowd"])
        pool.append((im, counts))
        if len(pool) == 500:
            break
    selected, coverage = [], Counter()
    while pool and len(selected) < args.count:
        best = max(range(len(pool)), key=lambda i: sum(1 / (1 + coverage[c]) for c in pool[i][1]) + 0.1 * any(n >= 2 for n in pool[i][1].values()))
        im, counts = pool.pop(best)
        selected.append(im)
        coverage.update(counts.keys())
    require(len(selected) == args.count, "insufficient eligible images")
    selection_file = raw / "pilot_selection.json"
    selection = {"seed": args.seed, "count": args.count, "source_split": "official_train2020", "image_ids": sorted(im["id"] for im in selected), "source_annotation_sha256": sha256(source), "selection_policy": "seeded pool of 500 mask-valid, non-jumpsuit images with 2..15 whole instances; greedy eight-class image coverage", "scanned_rejections": dict(rejected), "class_image_counts": dict(coverage)}
    if selection_file.exists():
        require(read_json(selection_file) == selection, "existing selection differs; use a new raw directory/version")
    else:
        write_json(selection_file, selection)
    remote = RemoteZip(IMAGES_URL)
    with zipfile.ZipFile(remote) as archive:
        members = {Path(info.filename).name: info for info in archive.infolist() if not info.is_dir()}
        print("ZIP index loaded", len(members), "members", flush=True)
        for index, im in enumerate(selected, 1):
            name = Path(im["file_name"]).name
            destination = raw / "images" / name
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists():
                content = archive.read(members[name])  # zipfile checks member CRC
                destination.write_bytes(content)
            print(f"{index}/{len(selected)} {im['id']} {name}", flush=True)
    write_json(raw / "download_provenance.json", {"annotation_url": ANNOTATIONS_URL, "annotation_sha256": sha256(source), "image_archive_url": IMAGES_URL, "archive_etag": remote.etag, "archive_bytes": remote.length, "range_bytes_transferred_this_run": remote.transferred, "images": [{"file_name": im["file_name"], "sha256": sha256(raw / "images" / Path(im["file_name"]).name)} for im in selected]})


if __name__ == "__main__":
    main()
