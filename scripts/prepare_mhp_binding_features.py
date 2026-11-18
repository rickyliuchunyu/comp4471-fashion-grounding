"""Prepare pinned train and development features for the MHP binding comparison."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from fashion_grounding.data.schema import read_json, require, sha256, write_json
from fashion_grounding.grounding.cache import FeatureCache
from fashion_grounding.grounding.controlled import FeatureBuilder
from scripts.migrate_identical_grounding_features import run as migrate


def builder_for(config, output, sources):
    builder = FeatureBuilder(config, output)
    def cache_chain():
        caches = []
        for source in sources:
            root = Path(source) / "features"
            require(root.is_dir(), "missing reference cache: " + str(root))
            caches.append(FeatureCache(root))
        return caches

    class ReadOnlyChain:
        def __init__(self):
            self.caches = cache_chain()

        def get(self, spec, n, dimension):
            for cache in self.caches:
                result = cache.get(spec, n, dimension)
                if result is not None:
                    return result
            return None

    builder.readonly = ReadOnlyChain()
    return builder


def build_inputs(builder, config, names):
    data = {}
    for name in names:
        spec = config["datasets"][name]
        data[name] = builder.build(name, Path(spec["candidates"]), Path(spec["queries"]), Path(spec["annotations"]),
                                  training=name == "train", draft_dev=name.startswith("mhp"))
        require(len(data[name]["queries"]) == spec["count"], "wrong query denominator: " + name)
    return data


def verify(config):
    for file, digest in config["pins"].items():
        require(sha256(file) == digest, "changed input: " + file)


def run(config_path, out, phase):
    config = read_json(config_path)
    verify(config)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    builder = builder_for(config, out, config["cache_sources"])
    if phase == "train":
        migrate(SimpleNamespace(
            source=config["old_candidates"], destination=config["datasets"]["train"]["candidates"],
            encoder_metadata=Path(config["previous_inference"]) / "run_metadata.json", cache=out / "features",
            source_cache=Path(config["cache_sources"][0]) / "features", output=out / "cache_migration.json"))
        names = ("train", "fp_gt", "fp_pred")
    else:
        names = ("mhp_gt", "mhp_pred")
    datasets = build_inputs(builder, config, names)
    write_json(out / "summary.json", {"hashes": {k: ds["hashes"] for k, ds in datasets.items()},
                                    "encoded_images": builder.encoded_images, "encoded_texts": builder.encoded_texts,
                                    "encoder": builder.signature, "phase": phase})
    print(json.dumps({"phase":phase,"encoded_images":builder.encoded_images,"encoded_texts":builder.encoded_texts}),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("train", "dev"), required=True)
    a = p.parse_args()
    run(a.config, a.output, a.phase)
