import argparse
import json
from pathlib import Path

import numpy as np


def merge(paths):
    payloads = [np.load(path, allow_pickle=True) for path in paths]
    keys = payloads[0].files
    arrays = {}
    for key in keys:
        if all(key in payload.files for payload in payloads):
            arrays[key] = np.concatenate([payload[key] for payload in payloads], axis=0)
    metadata = {
        "sources": [str(path) for path in paths],
        "samples": int(len(arrays["x"])),
        "keys": sorted(arrays),
    }
    return arrays, metadata


def main():
    parser = argparse.ArgumentParser(description="Merge compatible npz value datasets.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    inputs = [Path(path) for path in args.inputs]
    arrays, metadata = merge(inputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    with out.with_suffix(out.suffix + ".json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    print(f"wrote {out}: samples={metadata['samples']} sources={len(inputs)}")


if __name__ == "__main__":
    main()
