#!/usr/bin/env python
"""Distributed six-signal extraction for full LLaVA-OneVision + LLaVA-Video pools.

Outputs per-rank shard jsonl that can be merged later:
- AMM / frame_diversity
- VDS / PPL (loss_blind - loss_video)
- TNC score
- self_consistency (optional, expensive)
- lightweight text/vision features for joint clustering
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import warnings
import gc
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# Suppress known noisy warnings as early as possible (before torch/transformers import).
warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*")
warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*")

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.utils import logging as hf_logging

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


WORD_RE = re.compile(r"[A-Za-z0-9_]+")
MEDIA_TOKEN_RE = re.compile(r"<image>|<video>|<audio>")
TEMPORAL_RE = re.compile(
    r"\b(before|after|then|while|during|first|next|finally|sequence|later|earlier|transition|happen)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract six sample signals for the GDO candidate pool")
    p.add_argument("--onevision", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-frames", type=int, default=32)
    p.add_argument("--flow-frames", type=int, default=8, help="max frames used for flow/diversity computation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--onevision-max-lines", type=int, default=0)
    p.add_argument("--video-max-lines", type=int, default=0)
    p.add_argument("--onevision-total-lines", type=int, default=0)
    p.add_argument("--video-total-lines", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20000)
    p.add_argument("--no-tqdm", action="store_true")
    p.add_argument("--resume", type=int, default=1, help="1: append/skip existing shard rows; 0: overwrite shard")
    p.add_argument("--flush-every", type=int, default=200, help="flush+fsync every N newly written rows")
    p.add_argument("--self-consistency-samples", type=int, default=5, help=">0 to enable generation consistency")
    p.add_argument("--self-consistency-max-new-tokens", type=int, default=32)
    p.add_argument("--tnc-mode", choices=["regex", "llm"], default="llm")
    p.add_argument("--rank", type=int, default=-1)
    p.add_argument("--world-size", type=int, default=-1)
    p.add_argument("--local-rank", type=int, default=-1)
    return p.parse_args()


def normalize_text(x: object) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().split())


def strip_media_tokens(text: str) -> str:
    return normalize_text(MEDIA_TOKEN_RE.sub(" ", text))


def tokenize(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


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
            ans = normalize_text(content)
            if current_user and ans:
                pairs.append((current_user, ans, turn_idx))
                turn_idx += 1
            current_user = ""
    return pairs


def sample_paths(paths: Sequence[str], target: int) -> List[str]:
    arr = [str(x) for x in paths if x]
    n = len(arr)
    if target <= 0 or n <= target:
        return arr
    out = []
    for i in range(target):
        idx = int(round(i * (n - 1) / (target - 1)))
        out.append(arr[idx])
    return out


def temporal_regex_score(question: str) -> float:
    return 1.0 if TEMPORAL_RE.search(question or "") else 0.0


def iter_samples(
    onevision_path: str,
    video_path: str,
    target_frames: int,
    onevision_max_lines: int,
    video_max_lines: int,
    onevision_total_lines: int,
    video_total_lines: int,
    log_every: int,
    enable_tqdm: bool,
    print_parsing_log: bool,
) -> Iterator[Dict[str, object]]:
    ov_iter_desc = "onevision-lines"
    vv_iter_desc = "video-lines"

    # OneVision image QA
    with open(onevision_path, "r", encoding="utf-8") as f:
        it = f
        if enable_tqdm and tqdm is not None:
            it = tqdm(
                f,
                total=onevision_total_lines if onevision_total_lines > 0 else None,
                desc=ov_iter_desc,
                unit="line",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
        for line_idx, line in enumerate(it, start=1):
            if onevision_max_lines > 0 and line_idx > onevision_max_lines:
                break
            if print_parsing_log and log_every > 0 and line_idx % log_every == 0:
                print(f"[iter/onevision] parsed_lines={line_idx}")
            item = json.loads(line)
            item_id = normalize_text(item.get("id", f"onevision-{line_idx}"))
            images = item.get("images") if isinstance(item.get("images"), list) else []
            if not images:
                continue
            pairs = parse_message_pairs(item.get("messages"))
            if not pairs:
                continue
            for q, a, turn_idx in pairs:
                qa_uid = f"ov::{item_id}::t{turn_idx}"
                yield {
                    "qa_uid": qa_uid,
                    "video_uid": "",
                    "bucket": "image_qa",
                    "question": q,
                    "answer": a,
                    "images": [str(images[0])],
                }
        if enable_tqdm and tqdm is not None and hasattr(it, "close"):
            it.close()

    # LLaVA-Video QA
    with open(video_path, "r", encoding="utf-8") as f:
        it = f
        if enable_tqdm and tqdm is not None:
            it = tqdm(
                f,
                total=video_total_lines if video_total_lines > 0 else None,
                desc=vv_iter_desc,
                unit="line",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
            )
        for line_idx, line in enumerate(it, start=1):
            if video_max_lines > 0 and line_idx > video_max_lines:
                break
            if print_parsing_log and log_every > 0 and line_idx % log_every == 0:
                print(f"[iter/video] parsed_lines={line_idx}")
            item = json.loads(line)
            item_id = normalize_text(item.get("id", f"video-{line_idx}"))
            video_uid = normalize_text(item.get("video_id", item_id))
            images = item.get("images") if isinstance(item.get("images"), list) else []
            frames = sample_paths(images, target_frames)
            if not frames:
                continue
            pairs = parse_message_pairs(item.get("messages"))
            if not pairs:
                continue
            for q, a, turn_idx in pairs:
                qa_uid = f"vv::{item_id}::t{turn_idx}"
                yield {
                    "qa_uid": qa_uid,
                    "video_uid": video_uid,
                    "bucket": "short_video",
                    "question": q,
                    "answer": a,
                    "images": frames,
                }
        if enable_tqdm and tqdm is not None and hasattr(it, "close"):
            it.close()


def read_frames(paths: Sequence[str], max_frames: int) -> List[np.ndarray]:
    arr = sample_paths(paths, max_frames)
    out: List[np.ndarray] = []
    for p in arr:
        try:
            img = cv2.imread(p)
            if img is None:
                continue
            out.append(img)
        except Exception:
            continue
    return out


def compute_amm_and_diversity(paths: Sequence[str], max_frames: int) -> Tuple[float, float]:
    frames = read_frames(paths, max_frames=max_frames)
    if len(frames) <= 1:
        return 0.0, 0.0

    grays = []
    for fr in frames:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA)
        grays.append(g)

    magnitudes = []
    diffs = []
    for i in range(1, len(grays)):
        prev = grays[i - 1]
        cur = grays[i]
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            cur,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=2,
            poly_n=5,
            poly_sigma=1.1,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        magnitudes.append(float(np.mean(mag)))
        diffs.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32))) / 255.0))

    amm = float(np.mean(magnitudes)) if magnitudes else 0.0
    diversity = float(np.mean(diffs)) if diffs else 0.0
    return amm, diversity


def text_feature(question: str, answer: str, bins: int = 16) -> List[float]:
    vec = np.zeros((bins,), dtype=np.float32)
    toks = tokenize(question) + tokenize(answer)
    if not toks:
        return vec.tolist()
    for t in toks:
        idx = (hash(t) % bins + bins) % bins
        vec[idx] += 1.0
    vec /= float(np.linalg.norm(vec) + 1e-6)
    return vec.tolist()


def vision_feature(paths: Sequence[str], max_frames: int = 6) -> List[float]:
    frames = read_frames(paths, max_frames=max_frames)
    if not frames:
        return [0.0] * 6
    rgbs = []
    for fr in frames:
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgbs.append(rgb.reshape(-1, 3))
    arr = np.concatenate(rgbs, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    feat = np.concatenate([mean, std], axis=0)
    return feat.astype(np.float32).tolist()


def load_pil_images(paths: Sequence[str]) -> List[Image.Image]:
    out: List[Image.Image] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                out.append(img.convert("RGB"))
        except Exception:
            continue
    return out


@dataclass
class ModelBundle:
    processor: AutoProcessor
    model: Qwen3VLForConditionalGeneration
    device: torch.device


def build_model(model_path: str, local_rank: int) -> ModelBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen3-VL metric extraction")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    hf_logging.set_verbosity_error()
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    # Avoid repeated "invalid generation flags" warnings when deterministic decoding.
    if getattr(model, "generation_config", None) is not None:
        for key in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, key):
                setattr(model.generation_config, key, None)
        if hasattr(model.generation_config, "do_sample"):
            model.generation_config.do_sample = False
    model.eval()
    return ModelBundle(processor=processor, model=model, device=device)


def make_texts(
    processor: AutoProcessor,
    question: str,
    answer: str,
    num_images: int,
    use_images: bool,
) -> Tuple[str, str]:
    content = []
    if use_images:
        for _ in range(num_images):
            content.append({"type": "image"})
    content.append({"type": "text", "text": question})
    user_msg = [{"role": "user", "content": content}]
    full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    prompt_text = processor.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
    return prompt_text, full_text


def compute_loss(
    bundle: ModelBundle,
    question: str,
    answer: str,
    pil_images: Sequence[Image.Image],
    use_images: bool,
) -> float:
    prompt_text, full_text = make_texts(
        processor=bundle.processor,
        question=question,
        answer=answer,
        num_images=len(pil_images),
        use_images=use_images,
    )
    prompt_inputs = bundle.processor(
        text=[prompt_text],
        images=list(pil_images) if use_images else None,
        return_tensors="pt",
        padding=True,
    )
    full_inputs = bundle.processor(
        text=[full_text],
        images=list(pil_images) if use_images else None,
        return_tensors="pt",
        padding=True,
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100

    device_inputs = {}
    for k, v in full_inputs.items():
        if torch.is_tensor(v):
            device_inputs[k] = v.to(bundle.device)
    labels = labels.to(bundle.device)
    with torch.no_grad():
        out = bundle.model(**device_inputs, labels=labels)
    return float(out.loss.detach().float().item())


def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "out of memory" in msg:
        return True
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return False


def tnc_llm(bundle: ModelBundle, question: str) -> float:
    prompt = (
        "Decide if the question requires temporal reasoning over multiple frames.\n"
        "Answer only one word: Temporal or Static.\n"
        f"Question: {question}"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(text=[text], return_tensors="pt")
    inputs = {k: v.to(bundle.device) for k, v in inputs.items() if torch.is_tensor(v)}
    with torch.no_grad():
        gen = bundle.model.generate(**inputs, do_sample=False, max_new_tokens=3)
    new_tokens = gen[:, inputs["input_ids"].shape[1] :]
    resp = bundle.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip().lower()
    if "temporal" in resp:
        return 1.0
    if "static" in resp:
        return 0.0
    return 0.5


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter / max(union, 1))


def self_consistency_score(
    bundle: ModelBundle,
    question: str,
    pil_images: Sequence[Image.Image],
    samples: int,
    max_new_tokens: int,
) -> float:
    if samples <= 1:
        return -1.0
    content = [{"type": "image"} for _ in pil_images] + [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(text=[text], images=list(pil_images), return_tensors="pt", padding=True)
    inputs = {k: v.to(bundle.device) for k, v in inputs.items() if torch.is_tensor(v)}
    prompt_len = int(inputs["input_ids"].shape[1])
    outputs: List[str] = []
    with torch.no_grad():
        for _ in range(samples):
            gen = bundle.model.generate(
                **inputs,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                max_new_tokens=max_new_tokens,
            )
            new_tokens = gen[:, prompt_len:]
            txt = bundle.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            outputs.append(txt)
    if len(outputs) <= 1:
        return -1.0
    sims = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sims.append(jaccard_similarity(outputs[i], outputs[j]))
    if not sims:
        return -1.0
    return float(np.mean(sims))


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    warnings.filterwarnings("ignore", message=r".*pynvml package is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated! Use `dtype` instead!.*")

    rank = args.rank if args.rank >= 0 else int(os.environ.get("RANK", "0"))
    world_size = args.world_size if args.world_size >= 0 else int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = args.local_rank if args.local_rank >= 0 else int(os.environ.get("LOCAL_RANK", str(rank)))
    is_rank0 = rank == 0 and local_rank == 0
    enable_tqdm = bool(is_rank0 and not args.no_tqdm and tqdm is not None)
    print_parsing_log = bool(is_rank0 and not enable_tqdm)

    use_model = True
    bundle = build_model(args.model_path, local_rank=local_rank) if use_model else None

    shard_path = os.path.join(args.output_dir, f"sixd_metrics_rank{rank:04d}.jsonl")
    stat_path = os.path.join(args.output_dir, f"sixd_metrics_rank{rank:04d}.stat.json")

    start = time.time()
    assigned_seen = 0
    wrote_new = 0
    skipped_resume = 0
    bad = 0
    oom_skipped = 0
    resume_enabled = bool(args.resume)
    existing_rows = 0

    if resume_enabled and os.path.exists(shard_path):
        with open(shard_path, "r", encoding="utf-8") as f:
            existing_rows = sum(1 for _ in f)
    file_mode = "a" if resume_enabled and existing_rows > 0 else "w"

    with open(shard_path, file_mode, encoding="utf-8") as fout:
        for idx, sample in enumerate(
            iter_samples(
                onevision_path=args.onevision,
                video_path=args.video,
                target_frames=args.target_frames,
                onevision_max_lines=args.onevision_max_lines,
                video_max_lines=args.video_max_lines,
                onevision_total_lines=args.onevision_total_lines,
                video_total_lines=args.video_total_lines,
                log_every=args.log_every,
                enable_tqdm=enable_tqdm,
                print_parsing_log=print_parsing_log,
            )
        ):
            if idx % world_size != rank:
                continue
            assigned_seen += 1
            if resume_enabled and assigned_seen <= existing_rows:
                skipped_resume += 1
                continue

            if is_rank0 and not enable_tqdm and args.log_every > 0 and assigned_seen % args.log_every == 0:
                elapsed = time.time() - start
                print(
                    f"[rank{rank}] seen={assigned_seen} wrote_new={wrote_new} "
                    f"skip_resume={skipped_resume} bad={bad} oom={oom_skipped} elapsed={elapsed:.1f}s"
                )

            qa_uid = str(sample["qa_uid"])
            video_uid = str(sample["video_uid"])
            question = str(sample["question"])
            answer = str(sample["answer"])
            images = list(sample["images"])

            # Cheap metrics
            amm, frame_div = compute_amm_and_diversity(images, max_frames=args.flow_frames)
            tnc_score = temporal_regex_score(question)
            txt_feat = text_feature(question, answer)
            vis_feat = vision_feature(images, max_frames=min(args.flow_frames, 6))

            # Model-based metrics
            loss_video = None
            loss_blind = None
            vds = None
            ppl = None
            sc = -1.0
            try:
                pil_images = load_pil_images(images)
                if pil_images:
                    loss_video = compute_loss(bundle, question, answer, pil_images, use_images=True)
                else:
                    # keep row usable even if all image paths are unreadable
                    loss_video = compute_loss(bundle, question, answer, [], use_images=False)
                loss_blind = compute_loss(bundle, question, answer, [], use_images=False)
                vds = float(loss_blind - loss_video)
                ppl = float(math.exp(min(30.0, max(-30.0, loss_video))))
                if args.tnc_mode == "llm":
                    tnc_score = tnc_llm(bundle, question)
                if args.self_consistency_samples > 1 and len(pil_images) >= 1:
                    sc = self_consistency_score(
                        bundle=bundle,
                        question=question,
                        pil_images=pil_images,
                        samples=args.self_consistency_samples,
                        max_new_tokens=args.self_consistency_max_new_tokens,
                    )
            except Exception as e:
                bad += 1
                if is_oom_error(e):
                    oom_skipped += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                # still write cheap metrics for robustness
                loss_video = None
                loss_blind = None
                vds = None
                ppl = None
                sc = -1.0
                err = str(e)[:300]
            else:
                err = ""

            rec = {
                "qa_uid": qa_uid,
                "video_uid": video_uid,
                "bucket": sample["bucket"],
                "data_quality_metrics": {
                    "loss_video": loss_video,
                    "loss_blind": loss_blind,
                    "vds": vds,
                    "ppl": ppl,
                    "amm": amm,
                    "frame_diversity": frame_div,
                    "tnc_score": tnc_score,
                    "self_consistency": sc,
                    "text_feat": txt_feat,
                    "vis_feat": vis_feat,
                    "n_frames": len(images),
                    "question_len": len(tokenize(question)),
                    "answer_len": len(tokenize(answer)),
                    "error": err,
                },
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote_new += 1
            if args.flush_every > 0 and wrote_new % args.flush_every == 0:
                fout.flush()
                os.fsync(fout.fileno())

        fout.flush()
        os.fsync(fout.fileno())

    final_rows = existing_rows + wrote_new

    stat = {
        "rank": rank,
        "world_size": world_size,
        "resume_enabled": resume_enabled,
        "existing_rows": existing_rows,
        "assigned_seen": assigned_seen,
        "skipped_resume": skipped_resume,
        "written_new": wrote_new,
        "written_total": final_rows,
        "bad": bad,
        "oom_skipped": oom_skipped,
        "runtime_sec": round(time.time() - start, 2),
        "shard_path": shard_path,
    }
    with open(stat_path, "w", encoding="utf-8") as f:
        json.dump(stat, f, ensure_ascii=False, indent=2)
    print(json.dumps(stat, ensure_ascii=False))


if __name__ == "__main__":
    main()
