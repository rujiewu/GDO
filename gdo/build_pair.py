#!/usr/bin/env python
"""Build GDO vs. Uni-10x subsets from full LLaVA-OneVision + LLaVA-Video pools.

This script is designed for large corpora (millions of rows). It uses two passes:
1) profile pass: estimate full-corpus field distributions;
2) selection pass: distribution-constrained GDO sampling + Uni-10x baseline construction.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
MEDIA_TOKEN_RE = re.compile(r"<image>|<video>|<audio>")
MCQ_RE = re.compile(r"(\bchoices?\b)|(\nA\.)|(\nA\))", re.IGNORECASE)
TEMPORAL_RE = re.compile(
    r"\b(before|after|then|while|during|first|next|finally|sequence|later|earlier|transition|happen)\b",
    re.IGNORECASE,
)


PARSE_ERROR_STATS: Counter = Counter()
PARSE_ERROR_LOG_LIMIT = 20


def log_parse_error(kind: str, line_idx: int, exc: Exception) -> None:
    PARSE_ERROR_STATS[kind] += 1
    n = PARSE_ERROR_STATS[kind]
    if n <= PARSE_ERROR_LOG_LIMIT or n % 1000 == 0:
        print(f"[skip/{kind}] line={line_idx} count={n} err={type(exc).__name__}: {str(exc)[:160]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paired GDO and Uni-10x subsets from LLaVA-OneVision and LLaVA-Video")
    p.add_argument("--onevision", required=True, help="path to llava_onevision_*.jsonl")
    p.add_argument("--video", required=True, help="path to llava_video_*.jsonl")
    p.add_argument("--target-count", type=int, default=0, help="subset size; 0 means auto")
    p.add_argument("--target-ratio", type=float, default=0.0, help="subset ratio in (0,1], ignored if target-count > 0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-frames", type=int, default=32)
    p.add_argument("--max-qa-per-video", type=int, default=8)
    p.add_argument("--oversample-factor", type=float, default=1.35)
    p.add_argument("--random-oversample-factor", type=float, default=1.35, help="extra random reservoir budget for fair dedup-constrained baseline")
    p.add_argument(
        "--random-target-multiplier",
        type=float,
        default=1.0,
        help="final random subset size multiplier relative to filtered target_count (e.g. 10.0 for random 10x)",
    )
    p.add_argument(
        "--skip-random-output",
        action="store_true",
        help="build only the filtered set and write an empty random placeholder file",
    )
    p.add_argument(
        "--video-oversample-factor",
        type=float,
        default=2.2,
        help="oversample factor for short_video strata candidate pools",
    )
    p.add_argument(
        "--temporal-oversample-factor",
        type=float,
        default=3.0,
        help="oversample factor for temporal short_video strata candidate pools",
    )
    p.add_argument("--output-random", required=True)
    p.add_argument("--output-filtered", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--profile-output", default="")
    p.add_argument("--profile-only", action="store_true", help="only run pass1 profiling and budget recommendation")
    p.add_argument("--preprofile-report", default="", help="optional report json from profile-only run to skip pass1 scan")
    p.add_argument("--onevision-max-lines", type=int, default=0, help="debug cap; 0 means all")
    p.add_argument("--video-max-lines", type=int, default=0, help="debug cap; 0 means all")
    p.add_argument("--onevision-total-lines", type=int, default=0, help="optional cached line count for tqdm")
    p.add_argument("--video-total-lines", type=int, default=0, help="optional cached line count for tqdm")
    p.add_argument("--no-tqdm", action="store_true", help="disable tqdm progress bars")
    p.add_argument("--log-every", type=int, default=200000)
    p.add_argument("--auto-min-count", type=int, default=12000)
    p.add_argument("--auto-max-count", type=int, default=80000)
    p.add_argument("--auto-min-ratio", type=float, default=0.001)
    p.add_argument("--auto-max-ratio", type=float, default=0.012)
    p.add_argument("--auto-effective-strata-mult", type=float, default=24.0)
    p.add_argument("--auto-coverage-mass", type=float, default=0.95)
    p.add_argument("--auto-tail-target", type=float, default=10.0)
    p.add_argument(
        "--auto-vds3-target-positive",
        type=int,
        default=6000,
        help="when legacy metrics are provided, ensure auto budget can cover this many VDS3-positive samples",
    )
    p.add_argument(
        "--auto-vds3-threshold",
        type=float,
        default=0.0,
        help="threshold defining VDS3-positive samples in legacy-driven auto budget adjustment",
    )
    p.add_argument(
        "--auto-vds3-max-mult",
        type=float,
        default=3.0,
        help="max multiplier on base auto budget when legacy VDS3 adjustment is enabled",
    )
    p.add_argument(
        "--min-video-ratio",
        type=float,
        default=-1.0,
        help="hard lower bound on selected short_video ratio; <0 means auto (follow full-corpus ratio)",
    )
    p.add_argument(
        "--max-video-ratio",
        type=float,
        default=-1.0,
        help="hard upper bound on selected short_video ratio; <0 disables upper bound",
    )
    p.add_argument(
        "--min-temporal-in-video-ratio",
        type=float,
        default=-1.0,
        help="hard lower bound on Temporal share within selected short_video; <0 means auto (follow full-corpus ratio)",
    )
    p.add_argument(
        "--temporal-categories",
        default="Temporal",
        help="comma-separated categories regarded as temporal for hard constraints (e.g. Temporal or Temporal,Mixed)",
    )
    p.add_argument(
        "--source-group-floor-topk",
        type=int,
        default=0,
        help="apply source-group minimum-count floors to top-k source groups; 0 disables",
    )
    p.add_argument(
        "--min-source-group-ratio",
        type=float,
        default=0.0,
        help="minimum selected ratio floor used in source-group hard constraints",
    )
    p.add_argument(
        "--source-group-floor-frac-of-expected",
        type=float,
        default=0.35,
        help="minimum fraction of expected count (under full distribution) for source-group floor",
    )
    p.add_argument(
        "--metrics-jsonl",
        "--legacy-metrics-jsonl",
        dest="legacy_metrics_jsonl",
        default="",
        help="optional merged metric jsonl with qa_uid-level signals (loss_video/loss_blind/frame_diversity/vds/quality_score)",
    )
    p.add_argument(
        "--legacy-vds3-weight",
        type=float,
        default=0.95,
        help="weight for legacy VDS3 score when legacy metrics are provided",
    )
    p.add_argument(
        "--legacy-zscore-cap",
        type=float,
        default=4.0,
        help="clip absolute z-score used in legacy metric normalization",
    )
    p.add_argument("--video-base-weight", type=float, default=0.35, help="weight of heuristic base score for short_video")
    p.add_argument("--video-legacy-quality-weight", type=float, default=0.35, help="weight of legacy quality term for short_video")
    p.add_argument("--image-base-weight", type=float, default=0.9, help="weight of heuristic base score for image_qa")
    p.add_argument("--image-legacy-quality-weight", type=float, default=0.15, help="weight of legacy quality term for image_qa")
    p.add_argument("--missing-legacy-penalty", type=float, default=0.15, help="penalty for short_video sample without legacy metrics")
    p.add_argument("--min-legacy-coverage-ratio", type=float, default=0.0, help="warn/fail when loaded legacy metrics cover too little of full pairs")
    p.add_argument("--fail-on-low-legacy-coverage", action="store_true", help="raise error when legacy coverage ratio is below --min-legacy-coverage-ratio")
    p.add_argument("--disable-score-vds", action="store_true", help="ablation: drop the legacy VDS/VDS3 contribution from the filtered score")
    p.add_argument("--disable-score-ppl", action="store_true", help="ablation: drop the heuristic QA naturalness / PPL-like term from the filtered score")
    p.add_argument("--disable-score-sc", action="store_true", help="ablation: drop the legacy quality/self-consistency contribution from the filtered score")
    p.add_argument("--disable-vds3-budget", action="store_true", help="ablation: disable the legacy VDS-aware auto-budget adjustment")
    p.add_argument(
        "--exclude-reference-jsonl",
        default="",
        help="optional comma-separated jsonl paths; samples with matching question or qa fingerprints will be skipped",
    )
    p.add_argument(
        "--exclude-reference-max-lines",
        type=int,
        default=0,
        help="optional per-file read cap for exclusion reference loading; 0 means all",
    )
    p.add_argument(
        "--exclude-mode",
        choices=["question", "qa", "both"],
        default="both",
        help="fingerprint mode for exclusion reference matching",
    )
    return p.parse_args()


def format_seconds(sec: float) -> str:
    sec_i = int(max(0, round(sec)))
    h = sec_i // 3600
    m = (sec_i % 3600) // 60
    s = sec_i % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def count_lines(path: str, cap: int = 0) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for n, _ in enumerate(f, start=1):
            if cap > 0 and n >= cap:
                break
    return n


def normalize_text(x: object) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def parse_csv_paths(raw: str) -> List[str]:
    if not raw:
        return []
    out: List[str] = []
    for part in raw.split(","):
        p = normalize_text(part)
        if p:
            out.append(p)
    return out


def text_fingerprint(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()


def qa_fingerprint(question: str, answer: str) -> str:
    q = normalize_text(question)
    a = normalize_text(answer)
    return hashlib.sha1(f"{q}\t{a}".encode("utf-8")).hexdigest()


def strip_media_tokens(text: str) -> str:
    return normalize_text(MEDIA_TOKEN_RE.sub(" ", text))


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def as_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def clip_abs(x: float, cap: float) -> float:
    if cap <= 0:
        return x
    if x > cap:
        return cap
    if x < -cap:
        return -cap
    return x


def parse_message_pairs(messages: object) -> List[Tuple[str, str, int]]:
    if not isinstance(messages, list):
        return []
    pairs: List[Tuple[str, str, int]] = []
    current_user = ""
    turn_idx = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = normalize_text(m.get("role", m.get("from", ""))).lower()
        content = normalize_text(m.get("content", m.get("value", "")))
        if role in {"user", "human"}:
            current_user = strip_media_tokens(content)
            continue
        if role in {"assistant", "gpt"} and current_user:
            answer = normalize_text(content)
            if current_user and answer:
                pairs.append((current_user, answer, turn_idx))
                turn_idx += 1
            current_user = ""
    return pairs


def extract_qa_pairs_from_item(item: dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    q = normalize_text(item.get("question", ""))
    a = normalize_text(item.get("answer", ""))
    if q:
        out.append((q, a))
    for mq, ma, _ in parse_message_pairs(item.get("messages")):
        out.append((mq, ma))
    dedup: List[Tuple[str, str]] = []
    seen = set()
    for xq, xa in out:
        key = (normalize_text(xq), normalize_text(xa))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(key)
    return dedup


def load_exclusion_reference(
    paths: Sequence[str],
    mode: str = "both",
    max_lines_per_file: int = 0,
) -> Tuple[set, set, dict]:
    q_hashes: set = set()
    qa_hashes: set = set()
    stats = {
        "enabled": bool(paths),
        "mode": mode,
        "paths": list(paths),
        "files_loaded": 0,
        "rows_seen": 0,
        "rows_bad": 0,
        "q_hashes": 0,
        "qa_hashes": 0,
    }
    if not paths:
        return q_hashes, qa_hashes, stats

    use_q = mode in ("question", "both")
    use_qa = mode in ("qa", "both")

    for p in paths:
        if not os.path.isfile(p):
            print(f"[exclude-ref] skip missing file: {p}")
            continue
        stats["files_loaded"] += 1
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                if max_lines_per_file > 0 and idx > max_lines_per_file:
                    break
                stats["rows_seen"] += 1
                try:
                    item = json.loads(line)
                except Exception:
                    stats["rows_bad"] += 1
                    continue

                if not isinstance(item, dict):
                    stats["rows_bad"] += 1
                    continue
                pairs = extract_qa_pairs_from_item(item)
                if not pairs:
                    continue
                for q, a in pairs:
                    qn = normalize_text(q)
                    an = normalize_text(a)
                    if use_q and qn:
                        q_hashes.add(text_fingerprint(qn))
                    if use_qa and qn and an:
                        qa_hashes.add(qa_fingerprint(qn, an))

    stats["q_hashes"] = len(q_hashes)
    stats["qa_hashes"] = len(qa_hashes)
    return q_hashes, qa_hashes, stats


def exclusion_reason(record: dict, mode: str, q_hashes: set, qa_hashes: set) -> str:
    q = normalize_text(record.get("question", ""))
    a = normalize_text(record.get("answer", ""))
    if mode in ("question", "both") and q and text_fingerprint(q) in q_hashes:
        return "question"
    if mode in ("qa", "both") and q and a and qa_fingerprint(q, a) in qa_hashes:
        return "qa"
    return ""


def sample_frames(frames: Sequence[str], target_frames: int) -> List[str]:
    frame_list = [str(x) for x in frames if x]
    n = len(frame_list)
    if n <= target_frames:
        return frame_list
    out = []
    for i in range(target_frames):
        idx = int(round(i * (n - 1) / (target_frames - 1)))
        out.append(frame_list[idx])
    return out


def infer_question_form(question: str) -> str:
    return "MCQ" if MCQ_RE.search(question or "") else "Open"


def infer_temporal_category(question: str) -> str:
    return "Temporal" if TEMPORAL_RE.search(question or "") else "Mixed"


def q_len_bucket(q_tokens: int) -> str:
    if q_tokens <= 6:
        return "q_tiny"
    if q_tokens <= 14:
        return "q_short"
    if q_tokens <= 30:
        return "q_medium"
    if q_tokens <= 60:
        return "q_long"
    return "q_very_long"


def a_len_bucket(a_tokens: int) -> str:
    if a_tokens <= 2:
        return "a_tiny"
    if a_tokens <= 8:
        return "a_short"
    if a_tokens <= 24:
        return "a_medium"
    if a_tokens <= 64:
        return "a_long"
    return "a_very_long"


def duration_bucket(n_frames: int) -> str:
    if n_frames <= 1:
        return "image"
    if n_frames <= 8:
        return "very_short"
    if n_frames <= 24:
        return "short"
    if n_frames <= 64:
        return "medium"
    if n_frames <= 160:
        return "long"
    return "very_long"


def source_group(raw_source: str) -> str:
    s = normalize_text(raw_source)
    if not s:
        return "unknown"
    if "(" in s:
        return s.split("(", 1)[0].strip() or "unknown"
    parts = s.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return s


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self) -> float:
        if self.count <= 1:
            return 1.0
        var = self.m2 / (self.count - 1)
        if var <= 1e-12:
            return 1.0
        return float(math.sqrt(var))


@dataclass
class LegacyMetric:
    loss_video: Optional[float] = None
    loss_blind: Optional[float] = None
    frame_diversity: Optional[float] = None
    vds: Optional[float] = None
    quality_score: Optional[float] = None


@dataclass
class Candidate:
    score: float
    base_score: float
    legacy_vds3: float
    legacy_quality_score: float
    rand_tie: float
    qa_uid: str
    video_uid: str
    bucket: str
    text_hash: str
    stratum: str
    record: dict
    fields: Dict[str, str]


def iter_onevision_samples(
    path: str,
    target_frames: int,
    max_lines: int,
    log_every: int,
    enable_tqdm: bool = False,
    total_lines: int = 0,
    tqdm_desc: str = "build/onevision",
) -> Iterator[Tuple[dict, Dict[str, str], str]]:
    del target_frames  # kept for a symmetric signature with video iterator
    start_ts = time.time()
    total_for_eta = total_lines if total_lines > 0 else 0
    if max_lines > 0:
        total_for_eta = min(total_for_eta, max_lines) if total_for_eta > 0 else max_lines

    with open(path, "r", encoding="utf-8") as f:
        it = f
        if enable_tqdm and tqdm is not None:
            total = total_for_eta if total_for_eta > 0 else None
            it = tqdm(
                f,
                total=total,
                desc=tqdm_desc,
                unit="line",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )

        for line_idx, line in enumerate(it, start=1):
            if max_lines > 0 and line_idx > max_lines:
                break
            if log_every > 0 and line_idx % log_every == 0:
                elapsed = max(1e-9, time.time() - start_ts)
                rate = line_idx / elapsed
                if total_for_eta > 0:
                    rem = max(0, total_for_eta - line_idx)
                    eta = rem / max(rate, 1e-9)
                    pct = 100.0 * line_idx / max(total_for_eta, 1)
                    print(
                        f"[onevision-progress] lines={line_idx}/{total_for_eta} ({pct:.2f}%) "
                        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} rate={rate:.2f} line/s"
                    )
                else:
                    print(
                        f"[onevision-progress] lines={line_idx} "
                        f"elapsed={format_seconds(elapsed)} rate={rate:.2f} line/s"
                    )
            try:
                item = json.loads(line)
            except Exception as e:
                log_parse_error("onevision_json", line_idx, e)
                continue

            try:
                item_id = normalize_text(item.get("id", f"onevision-{line_idx}"))
                raw_source = normalize_text(item.get("data_source", "onevision"))
                src_group = source_group(raw_source)
                images = item.get("images") if isinstance(item.get("images"), list) else []
                if not images:
                    continue

                pairs = parse_message_pairs(item.get("messages"))
                if not pairs:
                    continue

                for q, a, turn_idx in pairs:
                    if not q or not a:
                        continue
                    qa_uid = f"ov::{item_id}::t{turn_idx}"
                    q_tok = len(tokenize(q))
                    a_tok = len(tokenize(a))
                    q_form = infer_question_form(q)
                    fields = {
                        "bucket": "image_qa",
                        "duration_bucket": "image",
                        "temporal_category": "Static",
                        "question_form": q_form,
                        "q_len_bucket": q_len_bucket(q_tok),
                        "a_len_bucket": a_len_bucket(a_tok),
                        "source_group": src_group,
                    }

                    record = {
                        "qa_uid": qa_uid,
                        "video_uid": "",
                        "question_type": q_form,
                        "question": q,
                        "answer": a,
                        "messages": [
                            {"role": "user", "content": f"<image>\n{q}"},
                            {"role": "assistant", "content": a},
                        ],
                        "images": [str(images[0])],
                        "data_quality_metrics": {"n_frames": 1, "temporal_category": "Static"},
                        "meta": {
                            "bucket": "image_qa",
                            "source_name": "llava_onevision_full",
                            "source_domain": "image",
                            "llava_source": raw_source,
                        },
                    }
                    yield record, fields, raw_source
            except Exception as e:
                log_parse_error("onevision_sample", line_idx, e)
                continue
        if enable_tqdm and tqdm is not None and hasattr(it, "close"):
            it.close()


def iter_video_samples(
    path: str,
    target_frames: int,
    max_lines: int,
    log_every: int,
    enable_tqdm: bool = False,
    total_lines: int = 0,
    tqdm_desc: str = "build/video",
) -> Iterator[Tuple[dict, Dict[str, str], str]]:
    start_ts = time.time()
    total_for_eta = total_lines if total_lines > 0 else 0
    if max_lines > 0:
        total_for_eta = min(total_for_eta, max_lines) if total_for_eta > 0 else max_lines

    with open(path, "r", encoding="utf-8") as f:
        it = f
        if enable_tqdm and tqdm is not None:
            total = total_for_eta if total_for_eta > 0 else None
            it = tqdm(
                f,
                total=total,
                desc=tqdm_desc,
                unit="line",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )

        for line_idx, line in enumerate(it, start=1):
            if max_lines > 0 and line_idx > max_lines:
                break
            if log_every > 0 and line_idx % log_every == 0:
                elapsed = max(1e-9, time.time() - start_ts)
                rate = line_idx / elapsed
                if total_for_eta > 0:
                    rem = max(0, total_for_eta - line_idx)
                    eta = rem / max(rate, 1e-9)
                    pct = 100.0 * line_idx / max(total_for_eta, 1)
                    print(
                        f"[video-progress] lines={line_idx}/{total_for_eta} ({pct:.2f}%) "
                        f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} rate={rate:.2f} line/s"
                    )
                else:
                    print(
                        f"[video-progress] lines={line_idx} "
                        f"elapsed={format_seconds(elapsed)} rate={rate:.2f} line/s"
                    )
            try:
                item = json.loads(line)
            except Exception as e:
                log_parse_error("video_json", line_idx, e)
                continue

            try:
                item_id = normalize_text(item.get("id", f"video-{line_idx}"))
                video_id = normalize_text(item.get("video_id", item_id))
                raw_source = normalize_text(item.get("data_source", "llava_video"))
                src_group = source_group(raw_source)
                frames = item.get("images") if isinstance(item.get("images"), list) else []
                if not frames:
                    continue
                frames_32 = sample_frames([str(x) for x in frames if x], target_frames)
                if not frames_32:
                    continue

                n_frames = len(frames_32)
                dur_bucket = duration_bucket(n_frames)
                pairs = parse_message_pairs(item.get("messages"))
                if not pairs:
                    continue

                for q, a, turn_idx in pairs:
                    if not q or not a:
                        continue
                    qa_uid = f"vv::{item_id}::t{turn_idx}"
                    q_tok = len(tokenize(q))
                    a_tok = len(tokenize(a))
                    q_form = infer_question_form(q)
                    temporal = infer_temporal_category(q)

                    fields = {
                        "bucket": "short_video",
                        "duration_bucket": dur_bucket,
                        "temporal_category": temporal,
                        "question_form": q_form,
                        "q_len_bucket": q_len_bucket(q_tok),
                        "a_len_bucket": a_len_bucket(a_tok),
                        "source_group": src_group,
                    }

                    record = {
                        "qa_uid": qa_uid,
                        "video_uid": video_id,
                        "question_type": q_form,
                        "question": q,
                        "answer": a,
                        "messages": [
                            {"role": "user", "content": f"{'<image>' * n_frames}\n{q}"},
                            {"role": "assistant", "content": a},
                        ],
                        "images": list(frames_32),
                        "data_quality_metrics": {"n_frames": n_frames, "temporal_category": temporal},
                        "meta": {
                            "bucket": "short_video",
                            "source_name": "llava_video_full",
                            "source_domain": "video_short",
                            "llava_source": raw_source,
                        },
                    }
                    yield record, fields, raw_source
            except Exception as e:
                log_parse_error("video_sample", line_idx, e)
                continue
        if enable_tqdm and tqdm is not None and hasattr(it, "close"):
            it.close()


def iter_all_samples(
    onevision_path: str,
    video_path: str,
    target_frames: int,
    onevision_max_lines: int,
    video_max_lines: int,
    log_every: int,
    enable_tqdm: bool = False,
    onevision_total_lines: int = 0,
    video_total_lines: int = 0,
    tqdm_desc_prefix: str = "build",
) -> Iterator[Tuple[dict, Dict[str, str], str]]:
    yield from iter_onevision_samples(
        onevision_path,
        target_frames,
        onevision_max_lines,
        log_every,
        enable_tqdm=enable_tqdm,
        total_lines=onevision_total_lines,
        tqdm_desc=f"{tqdm_desc_prefix}/onevision",
    )
    yield from iter_video_samples(
        video_path,
        target_frames,
        video_max_lines,
        log_every,
        enable_tqdm=enable_tqdm,
        total_lines=video_total_lines,
        tqdm_desc=f"{tqdm_desc_prefix}/video",
    )


def make_stratum(fields: Dict[str, str]) -> str:
    return "|".join(
        [
            fields.get("bucket", "unknown"),
            fields.get("duration_bucket", "unknown"),
            fields.get("temporal_category", "unknown"),
            fields.get("question_form", "unknown"),
            fields.get("q_len_bucket", "unknown"),
            fields.get("a_len_bucket", "unknown"),
            fields.get("source_group", "unknown"),
        ]
    )


def load_legacy_metrics(path: str) -> Tuple[Dict[str, LegacyMetric], Dict[str, Tuple[float, float]], int]:
    if not path:
        return {}, {}, 0
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    stats = {
        "loss_video": RunningStats(),
        "loss_blind": RunningStats(),
        "frame_diversity": RunningStats(),
        "vds": RunningStats(),
        "quality_score": RunningStats(),
    }
    out: Dict[str, LegacyMetric] = {}
    bad = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            qa_uid = normalize_text(item.get("qa_uid", item.get("id", "")))
            if not qa_uid:
                bad += 1
                continue

            m = item.get("data_quality_metrics")
            if not isinstance(m, dict):
                m = item

            lv = as_float(m.get("loss_video", item.get("loss_video")))
            lb = as_float(m.get("loss_blind", item.get("loss_blind")))
            fd = as_float(m.get("frame_diversity", item.get("frame_diversity")))
            vds = as_float(m.get("vds", item.get("vds")))
            qs = as_float(m.get("quality_score", item.get("quality_score")))

            out[qa_uid] = LegacyMetric(
                loss_video=lv,
                loss_blind=lb,
                frame_diversity=fd,
                vds=vds,
                quality_score=qs,
            )
            if lv is not None:
                stats["loss_video"].update(lv)
            if lb is not None:
                stats["loss_blind"].update(lb)
            if fd is not None:
                stats["frame_diversity"].update(fd)
            if vds is not None:
                stats["vds"].update(vds)
            if qs is not None:
                stats["quality_score"].update(qs)

    norm_stats = {
        k: (v.mean, v.std())
        for k, v in stats.items()
        if v.count > 0
    }
    return out, norm_stats, bad


def recommend_target_count(
    total: int,
    stratum_counts: Counter,
    min_count: int,
    max_count: int,
    min_ratio: float,
    max_ratio: float,
    effective_strata_mult: float,
    coverage_mass: float,
    tail_target: float,
) -> Tuple[int, dict]:
    if total <= 0:
        return 0, {}
    if not stratum_counts:
        fallback = min(total, max(min_count, int(total * max(min_ratio, 0.001))))
        return fallback, {"mode": "fallback_no_strata"}

    probs = [c / float(total) for c in stratum_counts.values() if c > 0]
    if not probs:
        fallback = min(total, max(min_count, int(total * max(min_ratio, 0.001))))
        return fallback, {"mode": "fallback_no_probs"}

    entropy = 0.0
    for p in probs:
        entropy -= p * math.log(max(p, 1e-12))
    effective_strata = math.exp(entropy)

    sorted_probs = sorted(probs, reverse=True)
    cov = 0.0
    p_tail = sorted_probs[-1]
    covered = 0
    for p in sorted_probs:
        cov += p
        covered += 1
        p_tail = p
        if cov >= coverage_mass:
            break

    n_eff = int(math.ceil(max(1.0, effective_strata * max(1.0, effective_strata_mult))))
    n_tail = int(math.ceil(max(1.0, tail_target) / max(p_tail, 1e-9)))

    low_ratio_count = int(math.ceil(total * max(min_ratio, 0.0)))
    high_ratio_count = int(math.floor(total * max(max_ratio, 0.0)))
    if high_ratio_count <= 0:
        high_ratio_count = total

    n = max(min_count, low_ratio_count, n_eff, n_tail)
    n = min(n, max_count, high_ratio_count, total)
    n = max(1, min(n, total))

    info = {
        "mode": "distribution_auto",
        "effective_strata": effective_strata,
        "coverage_mass": coverage_mass,
        "covered_strata": covered,
        "tail_prob_at_coverage": p_tail,
        "n_from_effective_strata": n_eff,
        "n_from_tail_coverage": n_tail,
        "n_from_min_ratio": low_ratio_count,
        "n_from_max_ratio": high_ratio_count,
        "recommended_count": n,
        "recommended_ratio": n / float(total),
    }
    return n, info


def choose_target_count(
    total: int,
    target_count: int,
    target_ratio: float,
    stratum_counts: Counter,
    args: argparse.Namespace,
) -> Tuple[int, dict]:
    if target_count > 0:
        return min(target_count, total), {"mode": "fixed_count", "requested_count": target_count}
    if target_ratio > 0:
        n = min(max(1, int(total * target_ratio)), total)
        return n, {"mode": "fixed_ratio", "requested_ratio": target_ratio}
    return recommend_target_count(
        total=total,
        stratum_counts=stratum_counts,
        min_count=args.auto_min_count,
        max_count=args.auto_max_count,
        min_ratio=args.auto_min_ratio,
        max_ratio=args.auto_max_ratio,
        effective_strata_mult=args.auto_effective_strata_mult,
        coverage_mass=args.auto_coverage_mass,
        tail_target=args.auto_tail_target,
    )


def apply_legacy_vds3_budget(
    base_target: int,
    total: int,
    metrics_map: Dict[str, LegacyMetric],
    stats: Dict[str, Tuple[float, float]],
    threshold: float,
    target_positive: int,
    zcap: float,
    max_mult: float,
    args: argparse.Namespace,
) -> Tuple[int, dict]:
    if base_target <= 0 or total <= 0:
        return base_target, {"enabled": False, "reason": "empty_target_or_total"}
    if not metrics_map:
        return base_target, {"enabled": False, "reason": "no_legacy_metrics"}

    total_valid = 0
    pos = 0
    for m in metrics_map.values():
        vds3, _ = legacy_vds3_score(m, stats, zcap)
        total_valid += 1
        if vds3 > threshold:
            pos += 1

    if total_valid <= 0:
        return base_target, {"enabled": False, "reason": "no_valid_vds3"}

    pos_ratio = pos / float(total_valid)
    if pos_ratio <= 0:
        # no positive items, keep base target and report.
        info = {
            "enabled": True,
            "reason": "zero_positive_ratio",
            "vds3_positive_ratio": 0.0,
            "vds3_positive_count": pos,
            "vds3_total_count": total_valid,
            "n_from_vds3_positive": total,
            "adjusted_target": base_target,
        }
        return base_target, info

    n_from_vds3 = int(math.ceil(max(1, target_positive) / pos_ratio))
    n_cap_mult = int(math.ceil(base_target * max(1.0, max_mult)))
    n_cap_ratio = int(math.floor(total * max(args.auto_max_ratio, 0.0)))
    if n_cap_ratio <= 0:
        n_cap_ratio = total
    n_cap = min(
        total,
        max(1, args.auto_max_count),
        max(1, n_cap_mult),
        max(1, n_cap_ratio),
    )
    adjusted = min(max(base_target, n_from_vds3), n_cap)
    adjusted = max(1, min(adjusted, total))

    info = {
        "enabled": True,
        "vds3_positive_ratio": pos_ratio,
        "vds3_positive_count": pos,
        "vds3_total_count": total_valid,
        "vds3_threshold": threshold,
        "vds3_target_positive": target_positive,
        "n_from_vds3_positive": n_from_vds3,
        "vds3_adjust_cap": n_cap,
        "adjusted_target": adjusted,
    }
    return adjusted, info


def stable_profile(field_counts: Dict[str, Counter]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for field, cnt in field_counts.items():
        s = sum(cnt.values())
        if s <= 0:
            out[field] = {}
            continue
        out[field] = {k: v / s for k, v in cnt.items()}
    return out


def quality_score(question: str, answer: str) -> float:
    q_tok = tokenize(question)
    a_tok = tokenize(answer)
    qn = max(1, len(q_tok))
    an = max(1, len(a_tok))
    q_term = -abs(math.log(qn) - math.log(14.0))
    a_term = -abs(math.log(an) - math.log(16.0))
    diversity = len(set(a_tok)) / float(an)
    repeat_pen = max(0.0, 0.55 - diversity)
    return float(q_term + 1.25 * a_term - 1.2 * repeat_pen)


def difficulty_score(question: str, answer: str) -> float:
    qn = len(tokenize(question))
    an = len(tokenize(answer))
    x = max(1.0, qn + 0.5 * an)
    z = abs(math.log(x) - math.log(22.0))
    return float(math.exp(-z * z / (2.0 * 0.85 * 0.85)))


def alignment_score(fields: Dict[str, str], profile: Dict[str, Dict[str, float]]) -> float:
    eps = 1e-9
    vals = []
    for key in ["duration_bucket", "temporal_category", "question_form", "q_len_bucket", "a_len_bucket", "source_group"]:
        p = profile.get(key, {}).get(fields.get(key, "unknown"), eps)
        vals.append(math.log(max(p, eps)))
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def temporal_bonus(record: dict) -> float:
    tag = normalize_text(record.get("data_quality_metrics", {}).get("temporal_category", "Mixed"))
    if tag == "Temporal":
        return 1.0
    if tag == "Mixed":
        return 0.2
    return -0.5


def source_prior(raw_source: str, source_counts: Counter, total: int) -> float:
    c = source_counts.get(raw_source, 1)
    return float(math.log((total + 1.0) / (c + 1.0)))


def z_value(x: Optional[float], stats: Dict[str, Tuple[float, float]], key: str, cap: float) -> Optional[float]:
    if x is None:
        return None
    if key not in stats:
        return None
    mean, std = stats[key]
    if std <= 0:
        return 0.0
    return clip_abs((x - mean) / std, cap)


def legacy_vds3_score(metric: Optional[LegacyMetric], stats: Dict[str, Tuple[float, float]], cap: float) -> Tuple[float, float]:
    if metric is None:
        return 0.0, 0.0

    z_lv = z_value(metric.loss_video, stats, "loss_video", cap)
    z_lb = z_value(metric.loss_blind, stats, "loss_blind", cap)
    z_fd = z_value(metric.frame_diversity, stats, "frame_diversity", cap)
    z_vds = z_value(metric.vds, stats, "vds", cap)
    z_qs = z_value(metric.quality_score, stats, "quality_score", cap)

    if z_lv is not None and z_lb is not None and z_fd is not None:
        vds3 = -z_lv + z_lb + z_fd
    elif z_vds is not None and z_fd is not None:
        vds3 = z_vds + 0.5 * z_fd
    elif z_vds is not None:
        vds3 = z_vds
    else:
        vds3 = 0.0

    quality = z_qs if z_qs is not None else 0.0
    return float(vds3), float(quality)


def candidate_score(
    record: dict,
    fields: Dict[str, str],
    raw_source: str,
    profile: Dict[str, Dict[str, float]],
    source_counts: Counter,
    total: int,
    legacy_metric: Optional[LegacyMetric],
    legacy_stats: Dict[str, Tuple[float, float]],
    legacy_vds3_weight: float,
    legacy_zscore_cap: float,
    video_base_weight: float,
    video_legacy_quality_weight: float,
    image_base_weight: float,
    image_legacy_quality_weight: float,
    missing_legacy_penalty: float,
    disable_score_vds: bool,
    disable_score_ppl: bool,
    disable_score_sc: bool,
) -> Tuple[float, float, float, float]:
    q = normalize_text(record.get("question", ""))
    a = normalize_text(record.get("answer", ""))
    qlty = quality_score(q, a)
    diff = difficulty_score(q, a)
    align = alignment_score(fields, profile)
    spr = source_prior(raw_source, source_counts, total)
    legacy_vds3, legacy_quality = legacy_vds3_score(legacy_metric, legacy_stats, legacy_zscore_cap)

    ppl_like_term = 0.0 if disable_score_ppl else qlty

    if fields.get("bucket") == "short_video":
        base = 1.0 * ppl_like_term + 0.85 * diff + 0.9 * align + 0.55 * temporal_bonus(record) + 0.15 * spr
    else:
        base = 1.1 * ppl_like_term + 0.85 * diff + 0.9 * align + 0.15 * spr

    norm_denom = max(legacy_zscore_cap, 1e-6)
    base_norm = math.tanh(base / 3.0)
    legacy_vds3_norm = clip_abs(legacy_vds3, legacy_zscore_cap) / norm_denom
    legacy_quality_norm = clip_abs(legacy_quality, legacy_zscore_cap) / norm_denom

    if fields.get("bucket") == "short_video":
        final = video_base_weight * base_norm
        if legacy_metric is not None:
            if not disable_score_vds:
                final += legacy_vds3_weight * legacy_vds3_norm
            if not disable_score_sc:
                final += video_legacy_quality_weight * legacy_quality_norm
        else:
            final -= max(0.0, missing_legacy_penalty)
    else:
        final = image_base_weight * base_norm
        if legacy_metric is not None and not disable_score_sc:
            final += image_legacy_quality_weight * legacy_quality_norm

    return float(final), float(base), float(legacy_vds3), float(legacy_quality)


def target_bucket_quotas(bucket_counts: Counter, target_count: int) -> Dict[str, int]:
    total = sum(bucket_counts.values())
    if total <= 0:
        return {}
    buckets = list(bucket_counts.keys())
    raw = {b: target_count * (bucket_counts[b] / float(total)) for b in buckets}
    quota = {b: int(math.floor(v)) for b, v in raw.items()}
    remain = target_count - sum(quota.values())
    if remain > 0:
        frac = sorted([(b, raw[b] - quota[b]) for b in buckets], key=lambda x: x[1], reverse=True)
        for i in range(remain):
            quota[frac[i % len(frac)][0]] += 1
    return quota


def target_quotas(counter: Counter, target_count: int) -> Dict[str, int]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    keys = list(counter.keys())
    raw = {k: target_count * (counter[k] / float(total)) for k in keys}
    quota = {k: int(math.floor(v)) for k, v in raw.items()}
    remain = target_count - sum(quota.values())
    if remain > 0:
        frac = sorted([(k, raw[k] - quota[k]) for k in keys], key=lambda x: x[1], reverse=True)
        for i in range(remain):
            quota[frac[i % len(frac)][0]] += 1
    return quota


def parse_temporal_category_set(x: str) -> set:
    vals = {normalize_text(v) for v in str(x).split(",")}
    vals.discard("")
    return vals


def decompose_stratum_key(key: str) -> Tuple[str, str, str, str, str, str, str]:
    parts = str(key).split("|")
    parts = parts[:7] + ["unknown"] * max(0, 7 - len(parts))
    return (
        normalize_text(parts[0]) or "unknown",
        normalize_text(parts[1]) or "unknown",
        normalize_text(parts[2]) or "unknown",
        normalize_text(parts[3]) or "unknown",
        normalize_text(parts[4]) or "unknown",
        normalize_text(parts[5]) or "unknown",
        normalize_text(parts[6]) or "unknown",
    )


def source_group_counter_from_strata(stratum_counts: Counter) -> Counter:
    out = Counter()
    for key, cnt in stratum_counts.items():
        if cnt <= 0:
            continue
        _, _, _, _, _, _, src_group = decompose_stratum_key(key)
        out[src_group] += int(cnt)
    return out


def temporal_video_counter_from_strata(stratum_counts: Counter) -> Counter:
    out = Counter()
    for key, cnt in stratum_counts.items():
        if cnt <= 0:
            continue
        bucket, _, temporal, _, _, _, _ = decompose_stratum_key(key)
        if bucket == "short_video":
            out[temporal] += int(cnt)
    return out


def compute_source_group_floor_requirements(
    source_group_counts: Counter,
    target_count: int,
    total_pairs: int,
    topk: int,
    min_ratio: float,
    frac_of_expected: float,
) -> Dict[str, int]:
    if topk <= 0 or min_ratio <= 0 or total_pairs <= 0 or target_count <= 0:
        return {}

    floor_abs = max(1, int(math.ceil(target_count * min_ratio)))
    req: Dict[str, int] = {}
    for group, cnt in source_group_counts.most_common(topk):
        if cnt <= 0:
            continue
        expected = target_count * (cnt / float(total_pairs))
        expected_floor = int(math.ceil(max(0.0, frac_of_expected) * expected))
        required = max(floor_abs, expected_floor)
        # Only enforce when requirement is not larger than expected mass.
        if required <= int(math.floor(expected)):
            req[group] = required

    # Safety: cap total mandatory floor to 70% target budget.
    max_total = int(math.floor(target_count * 0.7))
    if max_total <= 0:
        return {}
    total_req = sum(req.values())
    if total_req > max_total and total_req > 0:
        scale = max_total / float(total_req)
        scaled = {}
        for g, v in req.items():
            sv = max(1, int(math.floor(v * scale)))
            scaled[g] = sv
        req = scaled
    return req


def summarize(records: Sequence[dict], profile: Dict[str, Dict[str, float]]) -> dict:
    if not records:
        return {"count": 0}
    bucket = Counter()
    field_cnt = defaultdict(Counter)
    for r in records:
        b = normalize_text(r.get("meta", {}).get("bucket", "unknown"))
        bucket[b] += 1
        q = normalize_text(r.get("question", ""))
        a = normalize_text(r.get("answer", ""))
        field_cnt["bucket"][b] += 1
        field_cnt["duration_bucket"][duration_bucket(int(r.get("data_quality_metrics", {}).get("n_frames", 0)))] += 1
        field_cnt["temporal_category"][normalize_text(r.get("data_quality_metrics", {}).get("temporal_category", "Mixed"))] += 1
        field_cnt["question_form"][infer_question_form(q)] += 1
        field_cnt["q_len_bucket"][q_len_bucket(len(tokenize(q)))] += 1
        field_cnt["a_len_bucket"][a_len_bucket(len(tokenize(a)))] += 1
        src = normalize_text(r.get("meta", {}).get("llava_source", "unknown"))
        field_cnt["source_group"][source_group(src)] += 1

    js = {}
    for k, target_dist in profile.items():
        pred_cnt = field_cnt.get(k, Counter())
        pred_total = sum(pred_cnt.values())
        if pred_total <= 0 or not target_dist:
            continue
        pred = {kk: vv / pred_total for kk, vv in pred_cnt.items()}
        js[k] = js_divergence(pred, target_dist)

    return {"count": len(records), "bucket": dict(bucket), "js_to_full_profile": js}


def js_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    def normalize(dist: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(v, 0.0) for v in dist.values())
        if s <= 0:
            return {k: 1.0 / len(keys) for k in keys}
        return {k: max(dist.get(k, 0.0), 0.0) / s for k in keys}

    pp = normalize(p)
    qq = normalize(q)
    mm = {k: 0.5 * (pp[k] + qq[k]) for k in keys}

    def kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        val = 0.0
        for k in keys:
            if a[k] <= 0:
                continue
            val += a[k] * math.log((a[k] + eps) / (b[k] + eps))
        return val

    return 0.5 * kl(pp, mm) + 0.5 * kl(qq, mm)


def write_jsonl(path: str, records: Sequence[dict]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    start_ts = time.time()
    rng = random.Random(args.seed)
    enable_tqdm = bool((not args.no_tqdm) and (tqdm is not None))

    if not os.path.isfile(args.onevision):
        raise FileNotFoundError(args.onevision)
    if not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)
    if args.target_frames <= 0:
        raise ValueError("--target-frames must be > 0")

    exclude_paths = parse_csv_paths(args.exclude_reference_jsonl)
    exclude_q_hashes: set = set()
    exclude_qa_hashes: set = set()
    exclusion_info = {
        "enabled": False,
        "mode": args.exclude_mode,
        "paths": [],
        "files_loaded": 0,
        "rows_seen": 0,
        "rows_bad": 0,
        "q_hashes": 0,
        "qa_hashes": 0,
        "excluded_pass1": 0,
        "excluded_pass2": 0,
        "excluded_reason_pass1": {},
        "excluded_reason_pass2": {},
    }
    if exclude_paths:
        exclude_q_hashes, exclude_qa_hashes, loaded = load_exclusion_reference(
            paths=exclude_paths,
            mode=args.exclude_mode,
            max_lines_per_file=args.exclude_reference_max_lines,
        )
        exclusion_info.update(loaded)
        print(
            "[exclude-ref] loaded "
            f"files={loaded.get('files_loaded', 0)} rows={loaded.get('rows_seen', 0)} "
            f"q_hashes={loaded.get('q_hashes', 0)} qa_hashes={loaded.get('qa_hashes', 0)} "
            f"mode={args.exclude_mode}"
        )
        if args.preprofile_report:
            print("[warn] exclusion is enabled; ignore --preprofile-report to avoid profile mismatch")
            args.preprofile_report = ""

    excluded_pass1 = Counter()
    excluded_pass2 = Counter()

    onevision_total_lines = args.onevision_total_lines
    video_total_lines = args.video_total_lines
    if enable_tqdm:
        if onevision_total_lines <= 0:
            onevision_total_lines = count_lines(args.onevision, cap=args.onevision_max_lines)
        if video_total_lines <= 0:
            video_total_lines = count_lines(args.video, cap=args.video_max_lines)

    profile: Dict[str, Dict[str, float]]
    bucket_counts = Counter()
    source_counts = Counter()
    source_group_counts = Counter()
    stratum_counts = Counter()
    full_video_temporal_counts = Counter()
    total_pairs = 0

    if args.preprofile_report:
        if not os.path.isfile(args.preprofile_report):
            raise FileNotFoundError(args.preprofile_report)
        print(f"[pass1] loading preprofile report: {args.preprofile_report}")
        with open(args.preprofile_report, "r", encoding="utf-8") as f:
            pre = json.load(f)
        profile = pre.get("full_profile_fields", {})
        bucket_counts = Counter(pre.get("full_bucket_counts", {}))
        source_counts = Counter(pre.get("source_counts", {}))
        stratum_counts = Counter(pre.get("stratum_counts", {}))
        source_group_counts = Counter(pre.get("source_group_counts", {}))
        total_pairs = int(pre.get("meta", {}).get("total_pairs", 0))
        if not source_group_counts:
            source_group_counts = source_group_counter_from_strata(stratum_counts)
        full_video_temporal_counts = temporal_video_counter_from_strata(stratum_counts)
        if total_pairs <= 0 or not profile or not bucket_counts or not stratum_counts:
            raise RuntimeError("preprofile report missing required fields")
    else:
        print("[pass1] profiling full corpus...")
        field_counts: Dict[str, Counter] = defaultdict(Counter)
        for record, fields, raw_source in iter_all_samples(
            onevision_path=args.onevision,
            video_path=args.video,
            target_frames=args.target_frames,
            onevision_max_lines=args.onevision_max_lines,
            video_max_lines=args.video_max_lines,
            log_every=args.log_every,
            enable_tqdm=enable_tqdm,
            onevision_total_lines=onevision_total_lines,
            video_total_lines=video_total_lines,
            tqdm_desc_prefix="build/pass1",
        ):
            reason = exclusion_reason(record, args.exclude_mode, exclude_q_hashes, exclude_qa_hashes)
            if reason:
                excluded_pass1[reason] += 1
                continue
            total_pairs += 1
            bucket_counts[fields["bucket"]] += 1
            source_counts[raw_source] += 1
            source_group_counts[fields.get("source_group", "unknown")] += 1
            if fields.get("bucket") == "short_video":
                full_video_temporal_counts[fields.get("temporal_category", "unknown")] += 1
            for k, v in fields.items():
                field_counts[k][v] += 1
            stratum_counts[make_stratum(fields)] += 1

        if total_pairs <= 0:
            raise RuntimeError("no valid QA pairs found")
        profile = stable_profile(field_counts)

    legacy_metrics_map: Dict[str, LegacyMetric] = {}
    legacy_stats: Dict[str, Tuple[float, float]] = {}
    legacy_bad_rows = 0
    legacy_coverage_ratio = 0.0
    legacy_coverage_ratio_upper = 0.0
    if args.legacy_metrics_jsonl:
        print(f"[legacy] loading metrics: {args.legacy_metrics_jsonl}")
        legacy_metrics_map, legacy_stats, legacy_bad_rows = load_legacy_metrics(args.legacy_metrics_jsonl)
        if total_pairs > 0:
            legacy_coverage_ratio_upper = min(1.0, len(legacy_metrics_map) / float(total_pairs))
            if legacy_coverage_ratio_upper < max(0.0, args.min_legacy_coverage_ratio):
                msg = (
                    f"legacy coverage too low: loaded={len(legacy_metrics_map)} total_pairs={total_pairs} "
                    f"upper_ratio={legacy_coverage_ratio_upper:.6f} < min={args.min_legacy_coverage_ratio:.6f}"
                )
                if args.fail_on_low_legacy_coverage:
                    raise RuntimeError(msg)
                print(f"[warn] {msg}")
        print(
            f"[legacy] loaded={len(legacy_metrics_map)} bad_rows={legacy_bad_rows} "
            f"stats={list(legacy_stats.keys())}"
        )

    target_count, budget_info = choose_target_count(
        total=total_pairs,
        target_count=args.target_count,
        target_ratio=args.target_ratio,
        stratum_counts=stratum_counts,
        args=args,
    )

    # If auto budget mode and legacy metrics exist, let VDS3-positive coverage adjust target count.
    if (
        args.target_count <= 0
        and args.target_ratio <= 0
        and budget_info.get("mode") == "distribution_auto"
        and args.legacy_metrics_jsonl
        and legacy_metrics_map
        and not args.disable_vds3_budget
    ):
        adjusted, vds3_info = apply_legacy_vds3_budget(
            base_target=target_count,
            total=total_pairs,
            metrics_map=legacy_metrics_map,
            stats=legacy_stats,
            threshold=args.auto_vds3_threshold,
            target_positive=args.auto_vds3_target_positive,
            zcap=args.legacy_zscore_cap,
            max_mult=args.auto_vds3_max_mult,
            args=args,
        )
        if adjusted != target_count:
            print(f"[budget] VDS3 adjusted target_count: {target_count} -> {adjusted}")
        target_count = adjusted
        budget_info["vds3_adjustment"] = vds3_info
        budget_info["recommended_count"] = target_count
        budget_info["recommended_ratio"] = target_count / float(total_pairs)

    if args.skip_random_output:
        random_target_multiplier = 0.0
        random_target_count = 0
    else:
        random_target_multiplier = args.random_target_multiplier if args.random_target_multiplier > 0 else 1.0
        random_target_count = max(1, int(math.ceil(target_count * random_target_multiplier)))

    quotas = target_bucket_quotas(bucket_counts, target_count)
    stratum_quotas = target_quotas(stratum_counts, target_count)
    temporal_categories = parse_temporal_category_set(args.temporal_categories)

    full_video_ratio = 0.0
    if total_pairs > 0:
        full_video_ratio = bucket_counts.get("short_video", 0) / float(total_pairs)
    min_video_ratio_effective = args.min_video_ratio if args.min_video_ratio >= 0 else full_video_ratio
    min_video_ratio_effective = max(0.0, min(1.0, min_video_ratio_effective))
    min_video_count = int(math.ceil(min_video_ratio_effective * target_count))
    max_video_ratio_effective = 1.0 if args.max_video_ratio < 0 else max(0.0, min(1.0, args.max_video_ratio))
    max_video_count = int(math.floor(max_video_ratio_effective * target_count))
    max_video_count = max(min_video_count, min(max_video_count, target_count))

    full_temporal_in_video_ratio = 0.0
    short_video_total = bucket_counts.get("short_video", 0)
    if short_video_total > 0:
        temporal_full_cnt = sum(full_video_temporal_counts.get(t, 0) for t in temporal_categories)
        full_temporal_in_video_ratio = temporal_full_cnt / float(short_video_total)
    min_temporal_in_video_ratio_effective = (
        args.min_temporal_in_video_ratio if args.min_temporal_in_video_ratio >= 0 else full_temporal_in_video_ratio
    )
    min_temporal_in_video_ratio_effective = max(0.0, min(1.0, min_temporal_in_video_ratio_effective))
    temporal_base_video_count = max(min_video_count, quotas.get("short_video", 0))
    min_temporal_video_count = int(math.ceil(min_temporal_in_video_ratio_effective * temporal_base_video_count))
    min_temporal_video_count = max(0, min(min_temporal_video_count, target_count))

    source_group_floor_requirements = compute_source_group_floor_requirements(
        source_group_counts=source_group_counts,
        target_count=target_count,
        total_pairs=total_pairs,
        topk=args.source_group_floor_topk,
        min_ratio=args.min_source_group_ratio,
        frac_of_expected=args.source_group_floor_frac_of_expected,
    )

    print(
        f"[pass1] total_pairs={total_pairs} target_count={target_count} random_target_count={random_target_count} "
        f"bucket_counts={dict(bucket_counts)} quotas={quotas} strata={len(stratum_quotas)} "
        f"budget_mode={budget_info.get('mode', 'unknown')}"
    )
    print(
        f"[constraints] min_video_ratio={min_video_ratio_effective:.6f} min_video_count={min_video_count} "
        f"max_video_ratio={max_video_ratio_effective:.6f} max_video_count={max_video_count} "
        f"min_temporal_in_video_ratio={min_temporal_in_video_ratio_effective:.6f} "
        f"min_temporal_video_count={min_temporal_video_count} temporal_categories={sorted(list(temporal_categories))} "
        f"source_group_floor_topk={args.source_group_floor_topk} "
        f"source_group_floor_count={len(source_group_floor_requirements)}"
    )
    if budget_info:
        print(f"[pass1] budget_info={json.dumps(budget_info, ensure_ascii=False)}")

    if args.profile_output:
        ensure_parent(args.profile_output)
        with open(args.profile_output, "w", encoding="utf-8") as f:
            json.dump({"meta": {"total_pairs": total_pairs}, "fields": profile}, f, ensure_ascii=False, indent=2)
        print(f"[pass1] profile_output={args.profile_output}")

    if args.profile_only:
        profile_report = {
            "meta": {
                "onevision": args.onevision,
                "video": args.video,
                "total_pairs": total_pairs,
                "target_count": target_count,
                "random_target_count": random_target_count,
                "random_target_multiplier": random_target_multiplier,
                "skip_random_output": args.skip_random_output,
                "target_ratio_effective": target_count / float(total_pairs),
                "target_frames": args.target_frames,
                "oversample_factor": args.oversample_factor,
                "video_oversample_factor": args.video_oversample_factor,
                "temporal_oversample_factor": args.temporal_oversample_factor,
                "profile_only": True,
                "exclusion_guard": {
                    **exclusion_info,
                    "excluded_pass1": int(sum(excluded_pass1.values())),
                    "excluded_reason_pass1": {k: int(v) for k, v in excluded_pass1.items()},
                },
                "runtime_sec": round(time.time() - start_ts, 2),
            },
            "full_bucket_counts": dict(bucket_counts),
            "full_profile_fields": profile,
            "source_counts": dict(source_counts),
            "source_group_counts": dict(source_group_counts),
            "stratum_counts": dict(stratum_counts),
            "full_video_temporal_counts": dict(full_video_temporal_counts),
            "num_strata": len(stratum_quotas),
            "budget_info": budget_info,
            "hard_constraints": {
                "temporal_categories": sorted(list(temporal_categories)),
                "full_video_ratio": full_video_ratio,
                "full_temporal_in_video_ratio": full_temporal_in_video_ratio,
                "min_video_ratio_effective": min_video_ratio_effective,
                "min_video_count": min_video_count,
                "max_video_ratio_effective": max_video_ratio_effective,
                "max_video_count": max_video_count,
                "min_temporal_in_video_ratio_effective": min_temporal_in_video_ratio_effective,
                "min_temporal_video_count": min_temporal_video_count,
                "source_group_floor_requirements": source_group_floor_requirements,
            },
            "parse_error_stats": dict(PARSE_ERROR_STATS),
        }
        ensure_parent(args.report)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(profile_report, f, ensure_ascii=False, indent=2)
        print("[profile-only] finished pass1")
        print(json.dumps(profile_report["meta"], ensure_ascii=False, indent=2))
        print(f"[done] report={args.report}")
        return

    if args.skip_random_output:
        print("[pass2] selecting filtered only (random skipped)...")
    else:
        print("[pass2] selecting filtered + random...")
    pass2_start_ts = time.time()
    reservoirs: List[Candidate] = []
    seen_idx = 0
    raw_idx = 0
    legacy_hits_seen = 0
    random_target_cap = 0
    if random_target_count > 0:
        random_target_cap = max(
            random_target_count,
            int(math.ceil(random_target_count * max(1.0, args.random_oversample_factor))),
        )

    stratum_caps: Dict[str, int] = {}
    for s, q in stratum_quotas.items():
        if q <= 0:
            stratum_caps[s] = 0
            continue
        bucket, _, temporal_tag, _, _, _, _ = decompose_stratum_key(s)
        mult = max(args.oversample_factor, 1.0)
        if bucket == "short_video":
            mult = max(mult, args.video_oversample_factor)
            if temporal_tag in temporal_categories:
                mult = max(mult, args.temporal_oversample_factor)
        stratum_caps[s] = max(q, int(math.ceil(q * mult)))
    heaps: Dict[str, List[Tuple[float, float, Candidate]]] = {s: [] for s in stratum_quotas}

    it2 = iter_all_samples(
        onevision_path=args.onevision,
        video_path=args.video,
        target_frames=args.target_frames,
        onevision_max_lines=args.onevision_max_lines,
        video_max_lines=args.video_max_lines,
        log_every=args.log_every,
        enable_tqdm=False,
    )
    if enable_tqdm and tqdm is not None:
        it2 = tqdm(
            it2,
            total=total_pairs if total_pairs > 0 else None,
            desc="build/pass2-select",
            unit="pair",
            dynamic_ncols=True,
            mininterval=1.0,
            leave=True,
        )

    for record, fields, raw_source in it2:
        raw_idx += 1
        if args.log_every > 0 and raw_idx % args.log_every == 0:
            elapsed = max(1e-9, time.time() - pass2_start_ts)
            rate = raw_idx / elapsed
            if total_pairs > 0:
                rem = max(0, total_pairs - raw_idx)
                eta = rem / max(rate, 1e-9)
                pct = 100.0 * raw_idx / max(total_pairs, 1)
                print(
                    f"[pass2-progress] pairs={raw_idx}/{total_pairs} ({pct:.2f}%) "
                    f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)} "
                    f"rate={rate:.2f} pair/s reservoir={len(reservoirs)}"
                )
            else:
                print(
                    f"[pass2-progress] pairs={raw_idx} elapsed={format_seconds(elapsed)} "
                    f"rate={rate:.2f} pair/s reservoir={len(reservoirs)}"
                )
        reason = exclusion_reason(record, args.exclude_mode, exclude_q_hashes, exclude_qa_hashes)
        if reason:
            excluded_pass2[reason] += 1
            continue
        seen_idx += 1
        qa_uid = normalize_text(record.get("qa_uid", ""))
        video_uid = normalize_text(record.get("video_uid", ""))
        bucket = fields["bucket"]
        legacy_metric = legacy_metrics_map.get(qa_uid)
        if legacy_metric is not None:
            legacy_hits_seen += 1
        score, base_score, legacy_vds3, legacy_quality_score = candidate_score(
            record=record,
            fields=fields,
            raw_source=raw_source,
            profile=profile,
            source_counts=source_counts,
            total=total_pairs,
            legacy_metric=legacy_metric,
            legacy_stats=legacy_stats,
            legacy_vds3_weight=args.legacy_vds3_weight,
            legacy_zscore_cap=args.legacy_zscore_cap,
            video_base_weight=args.video_base_weight,
            video_legacy_quality_weight=args.video_legacy_quality_weight,
            image_base_weight=args.image_base_weight,
            image_legacy_quality_weight=args.image_legacy_quality_weight,
            missing_legacy_penalty=args.missing_legacy_penalty,
            disable_score_vds=args.disable_score_vds,
            disable_score_ppl=args.disable_score_ppl,
            disable_score_sc=args.disable_score_sc,
        )
        q = normalize_text(record.get("question", ""))
        a = normalize_text(record.get("answer", ""))
        text_hash = hashlib.sha1(f"{q}|||{a}".encode("utf-8")).hexdigest()

        cand = Candidate(
            score=score,
            base_score=base_score,
            legacy_vds3=legacy_vds3,
            legacy_quality_score=legacy_quality_score,
            rand_tie=rng.random(),
            qa_uid=qa_uid,
            video_uid=video_uid,
            bucket=bucket,
            text_hash=text_hash,
            stratum=make_stratum(fields),
            record=record,
            fields=fields,
        )

        # Random baseline: classic reservoir sampling.
        if random_target_cap > 0:
            if len(reservoirs) < random_target_cap:
                reservoirs.append(cand)
            else:
                j = rng.randint(1, seen_idx)
                if j <= random_target_cap:
                    reservoirs[j - 1] = cand

        # Filtered candidates: per-bucket top-k with oversampling.
        cap = stratum_caps.get(cand.stratum, 0)
        if cap > 0:
            h = heaps[cand.stratum]
            key = (cand.score, cand.rand_tie, cand)
            if len(h) < cap:
                heapq.heappush(h, key)
            else:
                if key[0] > h[0][0]:
                    heapq.heapreplace(h, key)

    if enable_tqdm and tqdm is not None and hasattr(it2, "close"):
        it2.close()

    # Finalize filtered selection with hard constraints + stratum quotas.
    selected_filtered: List[Candidate] = []
    used_uid = set()
    used_text = set()
    video_cnt = Counter()
    selected_by_bucket = Counter()
    selected_by_stratum = Counter()
    selected_by_source_group = Counter()
    selected_temporal_video = Counter()
    selection_stage_counts = Counter()

    sorted_by_stratum: Dict[str, List[Candidate]] = {}
    for s, h in heaps.items():
        arr = [x[2] for x in h]
        arr.sort(key=lambda c: (c.score, c.rand_tie), reverse=True)
        sorted_by_stratum[s] = arr

    tail: List[Candidate] = []
    for arr in sorted_by_stratum.values():
        tail.extend(arr)
    tail.sort(key=lambda c: (c.score, c.rand_tie), reverse=True)

    video_pool = [c for c in tail if c.bucket == "short_video"]
    temporal_video_pool = [
        c for c in video_pool if normalize_text(c.fields.get("temporal_category", "unknown")) in temporal_categories
    ]
    source_group_pools: Dict[str, List[Candidate]] = defaultdict(list)
    for c in tail:
        source_group_pools[normalize_text(c.fields.get("source_group", "unknown"))].append(c)

    def can_take(c: Candidate) -> bool:
        if c.qa_uid in used_uid:
            return False
        if c.text_hash in used_text:
            return False
        if c.bucket == "short_video" and selected_by_bucket.get("short_video", 0) >= max_video_count:
            return False
        if c.bucket == "short_video" and c.video_uid and video_cnt[c.video_uid] >= args.max_qa_per_video:
            return False
        return True

    def take(c: Candidate, stage: str) -> bool:
        if len(selected_filtered) >= target_count:
            return False
        if not can_take(c):
            return False
        selected_filtered.append(c)
        used_uid.add(c.qa_uid)
        used_text.add(c.text_hash)
        selected_by_bucket[c.bucket] += 1
        selected_by_stratum[c.stratum] += 1
        src_group = normalize_text(c.fields.get("source_group", "unknown"))
        selected_by_source_group[src_group] += 1
        if c.bucket == "short_video":
            temporal_tag = normalize_text(c.fields.get("temporal_category", "unknown"))
            selected_temporal_video[temporal_tag] += 1
            if c.video_uid:
                video_cnt[c.video_uid] += 1
        selection_stage_counts[stage] += 1
        return True

    def fill_from_pool(pool: Sequence[Candidate], need: int, stage: str) -> int:
        remain = max(0, int(need))
        if remain <= 0:
            return 0
        for c in pool:
            if remain <= 0 or len(selected_filtered) >= target_count:
                break
            if take(c, stage):
                remain -= 1
        return remain

    # Stage-1 hard constraints.
    unmet_constraints: Dict[str, int] = {}
    missing_temporal = max(
        0,
        min_temporal_video_count
        - sum(selected_temporal_video.get(t, 0) for t in temporal_categories),
    )
    missing_temporal = fill_from_pool(temporal_video_pool, missing_temporal, "hard_temporal_video")
    if missing_temporal > 0:
        unmet_constraints["temporal_video_min"] = missing_temporal

    missing_video = max(0, min_video_count - selected_by_bucket.get("short_video", 0))
    missing_video = fill_from_pool(video_pool, missing_video, "hard_video")
    if missing_video > 0:
        unmet_constraints["video_min"] = missing_video

    for group, req in source_group_floor_requirements.items():
        missing_group = max(0, req - selected_by_source_group.get(group, 0))
        if missing_group <= 0:
            continue
        missing_group = fill_from_pool(source_group_pools.get(group, []), missing_group, f"hard_source_group::{group}")
        if missing_group > 0:
            unmet_constraints[f"source_group::{group}"] = missing_group

    # Stage-2 stratum quota fill.
    for s, q in stratum_quotas.items():
        if len(selected_filtered) >= target_count:
            break
        if q <= 0:
            continue
        need = max(0, q - selected_by_stratum.get(s, 0))
        if need <= 0:
            continue
        fill_from_pool(sorted_by_stratum.get(s, []), need, "stratum_quota")

    # Stage-3 global fill by score.
    if len(selected_filtered) < target_count:
        fill_from_pool(tail, target_count - len(selected_filtered), "global_tail")

    # Stage-4 fallback from random reservoir.
    if len(selected_filtered) < target_count:
        fill_from_pool(reservoirs, target_count - len(selected_filtered), "reservoir_fallback")

    if len(selected_filtered) > target_count:
        selected_filtered = selected_filtered[:target_count]

    # Random output records (same target count) with comparable dedup constraints.
    random_selected: List[Candidate] = []
    if random_target_count > 0:
        random_pool = list(reservoirs)
        rng.shuffle(random_pool)
        random_used_uid = set()
        random_used_text = set()
        random_video_cnt = Counter()

        def random_can_take(c: Candidate, enforce_text: bool = True) -> bool:
            if c.qa_uid in random_used_uid:
                return False
            if enforce_text and c.text_hash in random_used_text:
                return False
            if c.bucket == "short_video" and c.video_uid and random_video_cnt[c.video_uid] >= args.max_qa_per_video:
                return False
            return True

        def random_take(c: Candidate, keep_text: bool = True) -> None:
            random_selected.append(c)
            random_used_uid.add(c.qa_uid)
            if keep_text:
                random_used_text.add(c.text_hash)
            if c.bucket == "short_video" and c.video_uid:
                random_video_cnt[c.video_uid] += 1

        for c in random_pool:
            if len(random_selected) >= random_target_count:
                break
            if random_can_take(c, enforce_text=True):
                random_take(c, keep_text=True)

        if len(random_selected) < random_target_count:
            for c in random_pool:
                if len(random_selected) >= random_target_count:
                    break
                if random_can_take(c, enforce_text=False):
                    random_take(c, keep_text=False)

        if len(random_selected) < random_target_count:
            for c in random_pool:
                if len(random_selected) >= random_target_count:
                    break
                random_selected.append(c)

    if seen_idx > 0:
        legacy_coverage_ratio = legacy_hits_seen / float(seen_idx)
    if args.legacy_metrics_jsonl and seen_idx > 0 and legacy_coverage_ratio < max(0.0, args.min_legacy_coverage_ratio):
        msg = (
            f"legacy coverage too low on scanned pairs: hits={legacy_hits_seen} seen={seen_idx} "
            f"ratio={legacy_coverage_ratio:.6f} < min={args.min_legacy_coverage_ratio:.6f}"
        )
        if args.fail_on_low_legacy_coverage:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")

    exclusion_info["excluded_pass1"] = int(sum(excluded_pass1.values()))
    exclusion_info["excluded_pass2"] = int(sum(excluded_pass2.values()))
    exclusion_info["excluded_reason_pass1"] = {k: int(v) for k, v in excluded_pass1.items()}
    exclusion_info["excluded_reason_pass2"] = {k: int(v) for k, v in excluded_pass2.items()}

    random_records = [c.record for c in random_selected[:random_target_count]]
    filtered_records = [c.record for c in selected_filtered]

    # Inject score metadata for traceability.
    score_map_filtered = {c.qa_uid: c.score for c in selected_filtered}
    score_map_random = {c.qa_uid: c.score for c in random_selected[:random_target_count]}
    score_detail_filtered = {
        c.qa_uid: {
            "base_score": c.base_score,
            "legacy_vds3": c.legacy_vds3,
            "legacy_quality_score": c.legacy_quality_score,
        }
        for c in selected_filtered
    }
    score_detail_random = {
        c.qa_uid: {
            "base_score": c.base_score,
            "legacy_vds3": c.legacy_vds3,
            "legacy_quality_score": c.legacy_quality_score,
        }
        for c in random_selected[:random_target_count]
    }
    for r in random_records:
        r.setdefault("meta", {})
        if isinstance(r["meta"], dict):
            uid = normalize_text(r.get("qa_uid", ""))
            r["meta"]["filter_score"] = float(score_map_random.get(uid, 0.0))
            r["meta"]["filter_score_detail"] = score_detail_random.get(uid, {})
    for r in filtered_records:
        r.setdefault("meta", {})
        if isinstance(r["meta"], dict):
            uid = normalize_text(r.get("qa_uid", ""))
            r["meta"]["filter_score"] = float(score_map_filtered.get(uid, 0.0))
            r["meta"]["filter_score_detail"] = score_detail_filtered.get(uid, {})

    write_jsonl(args.output_random, random_records)
    write_jsonl(args.output_filtered, filtered_records)

    overlap = 0.0
    if random_records and filtered_records:
        rset = {normalize_text(x.get("qa_uid", "")) for x in random_records}
        fset = {normalize_text(x.get("qa_uid", "")) for x in filtered_records}
        if rset:
            overlap = len(rset & fset) / float(min(len(rset), len(fset)))

    selected_video_count = selected_by_bucket.get("short_video", 0)
    selected_video_ratio = selected_video_count / float(len(filtered_records)) if filtered_records else 0.0
    selected_temporal_video_count = sum(selected_temporal_video.get(t, 0) for t in temporal_categories)
    selected_temporal_in_video_ratio = (
        selected_temporal_video_count / float(selected_video_count) if selected_video_count > 0 else 0.0
    )
    hard_constraint_status = {
        "temporal_categories": sorted(list(temporal_categories)),
        "full_video_ratio": full_video_ratio,
        "full_temporal_in_video_ratio": full_temporal_in_video_ratio,
        "min_video_ratio_effective": min_video_ratio_effective,
        "min_video_count": min_video_count,
        "max_video_ratio_effective": max_video_ratio_effective,
        "max_video_count": max_video_count,
        "min_temporal_in_video_ratio_effective": min_temporal_in_video_ratio_effective,
        "min_temporal_video_count": min_temporal_video_count,
        "source_group_floor_requirements": source_group_floor_requirements,
        "selected_video_count": selected_video_count,
        "selected_video_ratio": selected_video_ratio,
        "selected_temporal_video_count": selected_temporal_video_count,
        "selected_temporal_in_video_ratio": selected_temporal_in_video_ratio,
        "selected_source_group_counts_top20": dict(selected_by_source_group.most_common(20)),
        "selection_stage_counts": dict(selection_stage_counts),
        "unmet_constraints": unmet_constraints,
    }

    report = {
        "meta": {
            "onevision": args.onevision,
            "video": args.video,
            "total_pairs": total_pairs,
            "target_count": target_count,
            "random_target_count": random_target_count,
            "random_target_multiplier": random_target_multiplier,
            "skip_random_output": args.skip_random_output,
            "target_ratio_effective": target_count / float(total_pairs),
            "seed": args.seed,
            "target_frames": args.target_frames,
            "max_qa_per_video": args.max_qa_per_video,
            "oversample_factor": args.oversample_factor,
            "video_oversample_factor": args.video_oversample_factor,
            "temporal_oversample_factor": args.temporal_oversample_factor,
            "random_oversample_factor": args.random_oversample_factor,
            "budget_info": budget_info,
            "legacy_metrics_jsonl": args.legacy_metrics_jsonl,
            "legacy_vds3_weight": args.legacy_vds3_weight,
            "disable_score_vds": args.disable_score_vds,
            "disable_score_ppl": args.disable_score_ppl,
            "disable_score_sc": args.disable_score_sc,
            "disable_vds3_budget": args.disable_vds3_budget,
            "legacy_metrics_loaded": len(legacy_metrics_map),
            "legacy_coverage_ratio": legacy_coverage_ratio,
            "legacy_coverage_ratio_upper": legacy_coverage_ratio_upper,
            "min_legacy_coverage_ratio": args.min_legacy_coverage_ratio,
            "legacy_bad_rows": legacy_bad_rows,
            "exclusion_guard": exclusion_info,
            "runtime_sec": round(time.time() - start_ts, 2),
            "parse_error_stats": dict(PARSE_ERROR_STATS),
        },
        "full_bucket_counts": dict(bucket_counts),
        "full_profile_fields": profile,
        "source_counts": dict(source_counts),
        "source_group_counts": dict(source_group_counts),
        "stratum_counts": dict(stratum_counts),
        "full_video_temporal_counts": dict(full_video_temporal_counts),
        "bucket_quotas": quotas,
        "num_strata": len(stratum_quotas),
        "hard_constraints": hard_constraint_status,
        "random": summarize(random_records, profile),
        "filtered": summarize(filtered_records, profile),
        "overlap_ratio": overlap,
    }

    ensure_parent(args.report)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["meta"], ensure_ascii=False, indent=2))
    print(f"[done] random={args.output_random}")
    print(f"[done] filtered={args.output_filtered}")
    print(f"[done] report={args.report}")


if __name__ == "__main__":
    main()
