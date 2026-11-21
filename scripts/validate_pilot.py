import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_grounding.data.schema import validate_package, write_json

parser = argparse.ArgumentParser()
parser.add_argument("package", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
report = validate_package(args.package)
if args.report:
    write_json(args.report, report)
print(json.dumps(report, indent=2))
