#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

DIRECT_FIELDS = ("direct_answer", "direct_pred_answer", "direct_prediction", "direct_response")

def _extract_direct(obj):
    for k in DIRECT_FIELDS:
        v = obj.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None

def normalize_answer(s):
    if s is None:
        return ""
    t = str(s).strip().replace("$", "").replace(",", "")
    while t.endswith("."):
        t = t[:-1].strip()
    if not t:
        return ""
    try:
        x = float(t)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return format(x, "f").rstrip("0").rstrip(".") if "." in format(x, "f") else str(x)
    except ValueError:
        return t

def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _norm_cand(row, i):
    idx = row.get("candidate_index", i)
    pred = row.get("pred_answer") or row.get("predicted_answer_normalized") or row.get("predicted_answer_raw") or ""
    return {
        "candidate_index": idx,
        "pred_answer": pred,
        "tokens": row.get("tokens") or [],
        "steps": row.get("steps") or [],
    }

def load_data(step1_path):
    if os.path.isfile(step1_path):
        out = []
        for obj in _read_jsonl(step1_path):
            gold = obj.get("gold_answer") or obj.get("answer_normalized") or ""
            cands = [_norm_cand(c, i) for i, c in enumerate(obj.get("candidates") or [])]
            cands.sort(key=lambda x: x["candidate_index"])
            out.append({"sample_id": obj["sample_id"], "question": obj.get("question", ""), "gold_answer": gold, "candidates": cands, "direct_pred": _extract_direct(obj)})
        return out
    d = os.path.dirname(step1_path) or "."
    cp, cl = os.path.join(d, "candidate_paths.jsonl"), os.path.join(d, "cleaned_samples.jsonl")
    if not (os.path.isfile(cp) and os.path.isfile(cl)):
        raise FileNotFoundError(f"Need {step1_path} or both {cp} and {cl}")
    meta = {o["sample_id"]: o for o in _read_jsonl(cl)}
    by_sid = defaultdict(list)
    for row in _read_jsonl(cp):
        by_sid[row["sample_id"]].append(_norm_cand(row, 0))
    out = []
    for sid, obj in meta.items():
        cands = by_sid.get(sid, [])
        cands.sort(key=lambda x: x["candidate_index"])
        gold = obj.get("gold_answer") or obj.get("answer_normalized") or ""
        out.append({"sample_id": sid, "question": obj.get("question", ""), "gold_answer": gold, "candidates": cands, "direct_pred": _extract_direct(obj)})
    return out

def merge_direct_overlay(samples, overlay_path):
    """Merge direct predictions from a sidecar JSONL (same DIRECT_FIELDS); does not read/write main --data files."""
    if not overlay_path or not os.path.isfile(overlay_path):
        return
    by_sid = {}
    for obj in _read_jsonl(overlay_path):
        sid = obj.get("sample_id")
        if not sid:
            continue
        d = _extract_direct(obj)
        if d:
            by_sid[sid] = d
    for s in samples:
        if s["sample_id"] in by_sid:
            s["direct_pred"] = by_sid[s["sample_id"]]

def apply_direct_fallback_cot0(samples):
    """Fill missing direct_pred from candidate_index==0 (same as CoT). Use only when you lack a true direct run."""
    n = 0
    for s in samples:
        if s.get("direct_pred"):
            continue
        cands = s.get("candidates") or []
        if not cands:
            continue
        c0 = next((c for c in cands if c["candidate_index"] == 0), cands[0])
        p = c0.get("pred_answer") or ""
        if not str(p).strip():
            continue
        s["direct_pred"] = str(p).strip()
        n += 1
    return n

def avg_logprob(cand):
    toks = cand.get("tokens") or []
    lps = [t["logprob"] for t in toks if isinstance(t, dict) and t.get("logprob") is not None]
    return sum(lps) / len(lps) if lps else None

def avg_step_entropy(cand):
    steps = cand.get("steps") or []
    es = [s["mean_entropy"] for s in steps if isinstance(s, dict) and s.get("mean_entropy") is not None]
    return sum(es) / len(es) if es else None

def evaluate_direct(samples):
    ok, tot = 0, 0
    wrong = []
    for s in samples:
        d = s.get("direct_pred")
        if not d:
            continue
        tot += 1
        g, p = normalize_answer(s["gold_answer"]), normalize_answer(d)
        if g == p:
            ok += 1
        elif len(wrong) < 3:
            wrong.append((s["sample_id"], g, p, "direct"))
    if tot == 0:
        return None
    return ok / tot, wrong

def evaluate_cot(samples):
    ok, tot = 0, 0
    wrong = []
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue
        tot += 1
        c0 = next((c for c in cands if c["candidate_index"] == 0), cands[0])
        g, p = normalize_answer(s["gold_answer"]), normalize_answer(c0["pred_answer"])
        if g == p:
            ok += 1
        elif len(wrong) < 3:
            wrong.append((s["sample_id"], g, p, "cot"))
    return ok / tot if tot else 0.0, wrong

def evaluate_majority_vote(samples):
    ok, tot = 0, 0
    wrong = []
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue
        tot += 1
        first = {}
        preds = []
        for i, c in enumerate(cands):
            pn = normalize_answer(c["pred_answer"])
            preds.append(pn)
            if pn not in first:
                first[pn] = i
        counts = Counter(preds)
        best_n = max(counts.values())
        pick = min((p for p, n in counts.items() if n == best_n), key=lambda x: first[x])
        g = normalize_answer(s["gold_answer"])
        if g == pick:
            ok += 1
        elif len(wrong) < 3:
            wrong.append((s["sample_id"], g, pick, "vote"))
    return ok / tot if tot else 0.0, wrong

def evaluate_confidence(samples):
    ok, tot = 0, 0
    wrong = []
    for s in samples:
        cands = s["candidates"]
        if not cands:
            continue
        tot += 1
        lps = [avg_logprob(c) for c in cands]
        if any(x is not None for x in lps):
            best_i = max(range(len(cands)), key=lambda i: (lps[i] is not None, lps[i] if lps[i] is not None else float("-inf")))
        else:
            ents = [avg_step_entropy(c) for c in cands]
            if any(x is not None for x in ents):
                best_i = min(range(len(cands)), key=lambda i: (ents[i] is not None, ents[i] if ents[i] is not None else float("inf")))
            else:
                best_i = 0
        g = normalize_answer(s["gold_answer"])
        p = normalize_answer(cands[best_i]["pred_answer"])
        if g == p:
            ok += 1
        elif len(wrong) < 3:
            wrong.append((s["sample_id"], g, p, "score"))
    return ok / tot if tot else 0.0, wrong

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/final/step1_dataset.jsonl")
    ap.add_argument("--direct", metavar="PATH", default=None, help="Optional JSONL sidecar with sample_id + direct_* (merged by sample_id)")
    ap.add_argument("--direct-fallback", choices=("cot0",), default=None, metavar="MODE", help="Fill missing direct preds from candidate 0 (proxy; not a true direct-only run)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()
    path = args.data
    if args.verbose:
        print(f"Loading {path!r} …")
    samples = load_data(path)
    merge_direct_overlay(samples, args.direct)
    if args.direct_fallback == "cot0":
        filled = apply_direct_fallback_cot0(samples)
        if filled:
            print(f"Note: --direct-fallback cot0 filled direct_pred for {filled} samples from candidate 0 (proxy until you add a real direct run).", file=sys.stderr)
    if args.verbose:
        print(f"Loaded {len(samples)} samples")
        if args.direct:
            print(f"Merged direct overlay from {args.direct!r}")
    dr = evaluate_direct(samples)
    a1, w1 = evaluate_cot(samples)
    a2, w2 = evaluate_majority_vote(samples)
    a3, w3 = evaluate_confidence(samples)
    if dr is None:
        print("Direct Accuracy: N/A (no direct predictions found)")
    else:
        ad, wd = dr
        print(f"Direct Accuracy: {ad:.4f}")
    print(f"CoT Accuracy: {a1:.4f}")
    print(f"Best-of-N (vote) Accuracy: {a2:.4f}")
    print(f"Best-of-N (score) Accuracy: {a3:.4f}")
    if args.verbose:
        for name, w in ([("direct", dr[1])] if dr else []) + [("CoT", w1), ("vote", w2), ("score", w3)]:
            if w:
                sid, g, p, _ = w[0]
                print(f"  incorrect [{name}]: {sid} gold={g!r} pred={p!r}")

if __name__ == "__main__":
    main()
