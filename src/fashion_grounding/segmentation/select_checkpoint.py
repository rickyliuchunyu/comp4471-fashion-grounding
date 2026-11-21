"""Select from complete, comparable dev AP reports without copying model weights."""
import argparse
from pathlib import Path

from ..data.schema import read_json, sha256, write_json, require
from .evaluation import choose_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports, references = [], []
    for value in args.reports:
        path = Path(value).resolve()
        digest = sha256(path)
        require(digest == path.with_suffix('.sha256').read_text().strip(), 'Report hash mismatch')
        report = read_json(path)
        require(sha256(path.parent/'predictions.json') == report['predictions_sha256'], 'Prediction hash mismatch')
        checkpoint = Path(report['checkpoint_path'])
        require(sha256(checkpoint) == report['checkpoint_sha256'], 'Selected checkpoint hash mismatch')
        reports.append(report)
        references.append(dict(path=str(path), sha256=digest))
    selected = choose_checkpoint(reports)
    selected['evaluation_reports'] = references
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'selection.json', selected)
    print(selected)


if __name__ == '__main__':
    main()
