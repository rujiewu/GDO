#!/usr/bin/env python
"""Uniformly resample frame paths inside subset jsonl files."""
import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target-frames", type=int, default=32)
    return p.parse_args()


def resample_paths(paths, target):
    if not isinstance(paths, list):
        return []
    n = len(paths)
    if n <= target:
        return paths
    idx = [int(round(i * (n - 1) / (target - 1))) for i in range(target)]
    return [paths[i] for i in idx]


def main():
    args = parse_args()
    total = 0
    changed = 0
    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            total += 1
            x = json.loads(line)
            images = x.get("images")
            if isinstance(images, list) and len(images) > 1:
                orig = len(images)
                images = resample_paths(images, args.target_frames)
                if len(images) != orig:
                    changed += 1
                x["images"] = images
                dqm = x.get("data_quality_metrics")
                if isinstance(dqm, dict):
                    dqm["n_frames"] = len(images)
                    x["data_quality_metrics"] = dqm
            fout.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"total={total}, changed={changed}, output={args.output}")


if __name__ == "__main__":
    main()
