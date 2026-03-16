#!/usr/bin/env python
"""Merge distributed six-signal shards into one metric table.

The merged jsonl is directly consumable by build_pair.py via --metrics-jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge six-signal GDO metric shards")
    p.add_argument("--input-dir", required=True, help="dir containing sixd_metrics_rank*.jsonl")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--cluster-k", type=int, default=4096)
    p.add_argument("--cluster-batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="debug cap")
    p.add_argument(
        "--stats-sample-size",
        type=int,
        default=800000,
        help="reservoir sample size for robust statistics (median/IQR approximation)",
    )
    p.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bars")
    return p.parse_args()


def iter_shard_lines(path: str, max_retries: int = 8, retry_sleep: float = 2.0) -> Iterable[str]:
    line_no = 0
    retries = 0
    while True:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for _ in range(line_no):
                    skipped = f.readline()
                    if skipped == "":
                        return
                while True:
                    line = f.readline()
                    if line == "":
                        return
                    line_no += 1
                    yield line
            return
        except OSError as e:
            if getattr(e, "errno", None) != 5 or retries >= max_retries:
                raise
            retries += 1
            print(
                f"[warn] transient I/O error while reading {path} at line={line_no}; "
                f"retry {retries}/{max_retries} after {retry_sleep:.1f}s",
                flush=True,
            )
            time.sleep(retry_sleep)


def iter_shards(paths: List[str], max_rows: int = 0) -> Iterable[Dict[str, object]]:
    seen = 0
    for p in paths:
        for line in iter_shard_lines(p):
            if max_rows > 0 and seen >= max_rows:
                return
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen += 1
            yield rec


def safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def as_feat(rec: Dict[str, object]) -> np.ndarray:
    m = rec.get("data_quality_metrics", {})
    if not isinstance(m, dict):
        m = {}
    txt = m.get("text_feat", [])
    vis = m.get("vis_feat", [])
    if not isinstance(txt, list):
        txt = []
    if not isinstance(vis, list):
        vis = []
    base = [safe_float(x, 0.0) for x in txt] + [safe_float(x, 0.0) for x in vis]
    base += [
        safe_float(m.get("tnc_score"), 0.0),
        safe_float(m.get("amm"), 0.0),
        safe_float(m.get("frame_diversity"), 0.0),
        safe_float(m.get("ppl"), 0.0),
        safe_float(m.get("vds"), 0.0),
    ]
    return np.asarray(base, dtype=np.float32)


def robust_center_scale(vals: np.ndarray) -> Tuple[float, float]:
    if vals.size == 0:
        return 0.0, 1.0
    q50 = float(np.quantile(vals, 0.5))
    q25 = float(np.quantile(vals, 0.25))
    q75 = float(np.quantile(vals, 0.75))
    iqr = max(q75 - q25, 1e-6)
    return q50, iqr / 1.349


def zscore(v: float, center: float, scale: float) -> float:
    return float((v - center) / max(scale, 1e-6))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class ReservoirSampler:
    def __init__(self, capacity: int, seed: int):
        self.capacity = max(0, int(capacity))
        self.buf: List[float] = []
        self.total_seen = 0
        self.rng = random.Random(seed)

    def add(self, value: float) -> None:
        self.total_seen += 1
        if self.capacity <= 0:
            return
        if len(self.buf) < self.capacity:
            self.buf.append(value)
            return
        j = self.rng.randint(1, self.total_seen)
        if j <= self.capacity:
            self.buf[j - 1] = value

    def to_array(self) -> np.ndarray:
        if not self.buf:
            return np.asarray([], dtype=np.float32)
        return np.asarray(self.buf, dtype=np.float32)


def fit_kmeans_streaming(
    paths: List[str],
    k: int,
    batch_size: int,
    seed: int,
    max_rows: int,
    total_rows: int,
    enable_tqdm: bool,
) -> Tuple[MiniBatchKMeans, int]:
    if k < 2:
        raise RuntimeError(f"invalid k for kmeans: {k}")
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=batch_size,
        n_init=3,
        reassignment_ratio=0.01,
    )

    init_buf: List[np.ndarray] = []
    batch: List[np.ndarray] = []
    fit_rows = 0
    initialized = False

    it = iter_shards(paths, max_rows=max_rows)
    if enable_tqdm and tqdm is not None:
        it = tqdm(
            it,
            total=total_rows if total_rows > 0 else None,
            desc="merge/pass2-kmeans",
            unit="row",
            dynamic_ncols=True,
            mininterval=1.0,
        )

    for rec in it:
        feat = as_feat(rec)
        fit_rows += 1

        if not initialized:
            init_buf.append(feat)
            if len(init_buf) == k:
                arr = np.stack(init_buf, axis=0).astype(np.float32)
                kmeans.partial_fit(arr)
                initialized = True
            continue

        batch.append(feat)
        if len(batch) >= batch_size:
            arr = np.stack(batch, axis=0).astype(np.float32)
            kmeans.partial_fit(arr)
            batch.clear()

    if fit_rows < 2:
        raise RuntimeError("not enough rows for clustering")
    if not initialized:
        # This can happen only when max_rows < cluster_k.
        k_adj = max(2, min(k, len(init_buf)))
        kmeans = MiniBatchKMeans(
            n_clusters=k_adj,
            random_state=seed,
            batch_size=batch_size,
            n_init=3,
            reassignment_ratio=0.01,
        )
        arr = np.stack(init_buf, axis=0).astype(np.float32)
        kmeans.fit(arr)
        return kmeans, fit_rows

    if batch:
        arr = np.stack(batch, axis=0).astype(np.float32)
        kmeans.partial_fit(arr)

    if enable_tqdm and tqdm is not None and hasattr(it, "close"):
        it.close()

    return kmeans, fit_rows


def main() -> None:
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.input_dir, "sixd_metrics_rank*.jsonl")))
    if not paths:
        raise FileNotFoundError(f"no shard files under {args.input_dir}")

    enable_tqdm = bool((not args.no_tqdm) and (tqdm is not None))

    # Pass 1: streaming robust-stat sampling and row counting.
    vds_res = ReservoirSampler(args.stats_sample_size, args.seed + 11)
    ppl_res = ReservoirSampler(args.stats_sample_size, args.seed + 13)
    amm_res = ReservoirSampler(args.stats_sample_size, args.seed + 17)
    fd_res = ReservoirSampler(args.stats_sample_size, args.seed + 19)
    sc_res = ReservoirSampler(args.stats_sample_size, args.seed + 23)
    rows_scanned = 0

    it1 = iter_shards(paths, max_rows=args.max_rows)
    if enable_tqdm:
        it1 = tqdm(
            it1,
            total=args.max_rows if args.max_rows > 0 else None,
            desc="merge/pass1-scan",
            unit="row",
            dynamic_ncols=True,
            mininterval=1.0,
        )
    for rec in it1:
        rows_scanned += 1
        m = rec.get("data_quality_metrics", {})
        if not isinstance(m, dict):
            m = {}
        vds_res.add(safe_float(m.get("vds"), 0.0))
        ppl_res.add(safe_float(m.get("ppl"), 0.0))
        amm_res.add(safe_float(m.get("amm"), 0.0))
        fd_res.add(safe_float(m.get("frame_diversity"), 0.0))
        sc = safe_float(m.get("self_consistency"), -1.0)
        if sc >= 0.0:
            sc_res.add(sc)

    if enable_tqdm and tqdm is not None and hasattr(it1, "close"):
        it1.close()

    if rows_scanned <= 0:
        raise RuntimeError("no valid metric rows loaded")

    # Pass 2: streaming kmeans fit over all rows.
    k = min(max(2, args.cluster_k), rows_scanned)
    kmeans, fit_rows = fit_kmeans_streaming(
        paths=paths,
        k=k,
        batch_size=args.cluster_batch_size,
        seed=args.seed,
        max_rows=args.max_rows,
        total_rows=rows_scanned,
        enable_tqdm=enable_tqdm,
    )
    centers = kmeans.cluster_centers_

    vds_center, vds_scale = robust_center_scale(vds_res.to_array())
    ppl_center, ppl_scale = robust_center_scale(ppl_res.to_array())
    amm_center, amm_scale = robust_center_scale(amm_res.to_array())
    fd_center, fd_scale = robust_center_scale(fd_res.to_array())
    sc_arr = sc_res.to_array()
    if sc_arr.size > 0:
        sc_center, sc_scale = robust_center_scale(sc_arr)
    else:
        sc_center, sc_scale = 0.5, 0.2

    # Pass 3: score + write merged jsonl.
    out_parent = os.path.dirname(args.output_jsonl)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    written = 0
    bad = 0
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        it3 = iter_shards(paths, max_rows=args.max_rows)
        if enable_tqdm:
            it3 = tqdm(
                it3,
                total=rows_scanned,
                desc="merge/pass3-write",
                unit="row",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        for rec in it3:
            m = rec.get("data_quality_metrics", {})
            if not isinstance(m, dict):
                m = {}

            feat = as_feat(rec)
            cid = int(kmeans.predict(feat.reshape(1, -1))[0])
            center = centers[cid]
            dist = float(np.linalg.norm(feat - center))

            vds = safe_float(m.get("vds"), 0.0)
            loss_video = m.get("loss_video")
            loss_blind = m.get("loss_blind")
            frame_div = safe_float(m.get("frame_diversity"), 0.0)
            ppl = safe_float(m.get("ppl"), 0.0)
            amm = safe_float(m.get("amm"), 0.0)
            tnc = safe_float(m.get("tnc_score"), 0.0)
            sc = safe_float(m.get("self_consistency"), -1.0)
            if sc < 0:
                # fallback consistency proxy from model confidence margins
                sc = sigmoid(0.6 * zscore(vds, vds_center, vds_scale) - 0.2 * abs(zscore(ppl, ppl_center, ppl_scale)))

            vds3 = zscore(vds, vds_center, vds_scale) + zscore(frame_div, fd_center, fd_scale)
            ppl_mid = math.exp(-abs(zscore(ppl, ppl_center, ppl_scale)))
            amm_mid = math.exp(-abs(zscore(amm, amm_center, amm_scale)))
            cluster_rep = math.exp(-dist)
            cluster_hard = 1.0 - cluster_rep

            quality = (
                0.28 * sigmoid(vds3)
                + 0.18 * sc
                + 0.14 * ppl_mid
                + 0.12 * amm_mid
                + 0.12 * tnc
                + 0.10 * cluster_rep
                + 0.06 * cluster_hard
            )

            out = {
                "qa_uid": rec.get("qa_uid", ""),
                "video_uid": rec.get("video_uid", ""),
                "loss_video": loss_video,
                "loss_blind": loss_blind,
                "frame_diversity": frame_div,
                "vds": vds,
                "quality_score": float(quality),
                "ppl": ppl,
                "amm": amm,
                "tnc_score": tnc,
                "self_consistency": sc,
                "cluster_score": cluster_rep,
                "cluster_hardness": cluster_hard,
                "data_quality_metrics": {
                    "loss_video": loss_video,
                    "loss_blind": loss_blind,
                    "frame_diversity": frame_div,
                    "vds": vds,
                    "quality_score": float(quality),
                    "ppl": ppl,
                    "amm": amm,
                    "tnc_score": tnc,
                    "self_consistency": sc,
                    "cluster_score": cluster_rep,
                    "cluster_hardness": cluster_hard,
                    "cluster_id": cid,
                    "cluster_dist": dist,
                },
            }
            if out["qa_uid"] == "":
                bad += 1
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
        if enable_tqdm and tqdm is not None and hasattr(it3, "close"):
            it3.close()

    report = {
        "input_dir": args.input_dir,
        "num_shards": len(paths),
        "rows_scanned": rows_scanned,
        "rows_fit": fit_rows,
        "rows_written": written,
        "bad_rows": bad,
        "cluster_k": int(kmeans.n_clusters),
        "cluster_k_requested": args.cluster_k,
        "stats_sample_size": args.stats_sample_size,
        "vds_center": vds_center,
        "vds_scale": vds_scale,
        "ppl_center": ppl_center,
        "ppl_scale": ppl_scale,
        "amm_center": amm_center,
        "amm_scale": amm_scale,
        "frame_div_center": fd_center,
        "frame_div_scale": fd_scale,
        "self_consistency_center": sc_center,
        "self_consistency_scale": sc_scale,
        "output_jsonl": args.output_jsonl,
    }
    report_parent = os.path.dirname(args.output_report)
    if report_parent:
        os.makedirs(report_parent, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
