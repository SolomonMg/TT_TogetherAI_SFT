#!/usr/bin/env python3
"""
infer.py — run a Together model over a validation JSONL and dump raw outputs.

Example:
python infer.py \
  --val-file data/val_BAL.jsonl \
  --model openai/gpt-oss-120b \
  --out out/preds_base.raw.jsonl \
  --concurrency 4 --temperature 0 \
  --max-tokens 512 --max-tokens-cap 2048 \
  --retries 5 --base-sleep 1.0 --warmup 1 \
  --reasoning-effort low --limit 10

"""

import os, sys, json, re, time, argparse, asyncio
from typing import List, Dict, Tuple
from together import Together

JSON_ONLY_MSG = {
    "role": "system",
    "content": (
        "Return ONLY a minified JSON object with exactly these keys: "
        "china_stance_score, china_sensitive, collective_action, languages. "
        "No extra text, no reasoning, no code fences."
    ),
}

FALLBACK_SCHEMA_MSG = {
    "role": "system",
    "content": (
        "You must respond with ONLY a single JSON object with these keys and types: "
        "{\"china_stance_score\": float in [-1,1], "
        "\"china_sensitive\": \"yes\"|\"no\"|\"cannot_determine\", "
        "\"collective_action\": \"yes\"|\"no\"|\"cannot_determine\", "
        "\"languages\": [\"english\",\"mandarin\",\"spanish\",\"other\",\"no_language\"]}. "
        "No prose. No code fences. Start with '{' and end with '}'."
    ),
}

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield i, json.loads(line)

def prompt_messages(example: dict, add_json_only: bool = True) -> List[Dict[str,str]]:
    # Hide the gold (last assistant)
    msgs = example["messages"][:-1]
    # Keep only role+content
    msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    if add_json_only:
        msgs.append(JSON_ONLY_MSG)
    return msgs

def user_text_concat(example: dict) -> str:
    parts = [m.get("content","") for m in example["messages"] if m.get("role") == "user"]
    return "\n\n".join(parts).strip()

def resp_text_content_only(resp) -> str:
    # 1) choices[0].message.content
    try:
        t = resp.choices[0].message.content
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    # 2) choices[0].message dict-like
    try:
        msg = resp.choices[0].message
        if msg and isinstance(msg, dict):
            t = msg.get("content")
            if isinstance(t, str) and t:
                return t
    except Exception:
        pass
    # 3) choices[0].text (some backends)
    try:
        t = resp.choices[0].text
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    # 4) very old shapes
    try:
        o = getattr(resp, "output", None)
        if isinstance(o, list) and o:
            t = o[0].get("content") or o[0].get("text")
            if isinstance(t, str) and t:
                return t
    except Exception:
        pass
    return ""

async def call_once(loop, client: Together, model: str, messages: list,
                    temperature: float, max_tokens: int,
                    response_format_json: bool, timeout_s: float,
                    reasoning_effort: str | None) -> str:
    def _do():
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            # Ignored by models that don't support it; safe to send.
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.chat.completions.create(**kwargs)
        return resp_text_content_only(resp)

    return await asyncio.wait_for(loop.run_in_executor(None, _do), timeout=timeout_s)

async def infer_once(loop, client, model, ex_idx: int, example: dict,
                     temperature: float, max_tokens: int,
                     max_tokens_cap: int,
                     retries: int, base_sleep: float,
                     per_call_timeout: float, response_format_json: bool,
                     reasoning_effort: str | None) -> dict:
    """
    Two dimensions of retry:
      - API retries (errors/timeouts)
      - Token budget growth (if no '{' found, double max_tokens up to cap)
    """
    msgs = prompt_messages(example, add_json_only=True)

    attempt = 0
    delay = base_sleep
    while attempt <= retries:
        attempt += 1
        t0 = time.time()
        try:
            curr_max_tokens = max_tokens
            # token-growth attempts
            while True:
                text = await call_once(
                    loop, client, model, msgs, temperature, curr_max_tokens,
                    response_format_json, per_call_timeout, reasoning_effort
                )
                if "{" in text:
                    return {
                        "idx": ex_idx,
                        "raw": text,
                        "attempt": attempt,
                        "latency_s": round(time.time() - t0, 3),
                        "model": model,
                        "used_max_tokens": curr_max_tokens,
                    }

                # Try same context with a very explicit “JSON ONLY” poke
                msgs_b = msgs + [{"role":"user","content":"JSON ONLY. No analysis. No code fences. Start with '{'."}]
                text_b = await call_once(
                    loop, client, model, msgs_b, temperature, curr_max_tokens,
                    response_format_json, per_call_timeout, reasoning_effort
                )
                if "{" in text_b:
                    return {
                        "idx": ex_idx,
                        "raw": text_b,
                        "attempt": attempt,
                        "latency_s": round(time.time() - t0, 3),
                        "model": model,
                        "used_max_tokens": curr_max_tokens,
                    }

                # If still no JSON, grow the token budget (model likely rambled first)
                next_tokens = min(curr_max_tokens * 2, max_tokens_cap)
                if next_tokens <= curr_max_tokens:
                    break  # can't grow further this attempt

                curr_max_tokens = next_tokens

            # Minimal fallback: schema + user-only
            user_txt = user_text_concat(example)
            msgs_c = [FALLBACK_SCHEMA_MSG, {"role":"user","content": user_txt}]
            text_c = await call_once(
                loop, client, model, msgs_c, temperature, curr_max_tokens,
                response_format_json, per_call_timeout, reasoning_effort
            )
            return {
                "idx": ex_idx,
                "raw": text_c,
                "attempt": attempt,
                "latency_s": round(time.time() - t0, 3),
                "model": model,
                "used_max_tokens": curr_max_tokens,
            }

        except Exception as e:
            err = repr(e)
            if attempt > retries:
                return {
                    "idx": ex_idx,
                    "raw": f"<<REQUEST FAILED: {err}>>",
                    "attempt": attempt,
                    "latency_s": round(time.time() - t0, 3),
                    "model": model,
                    "used_max_tokens": max_tokens,
                }

        await asyncio.sleep(delay)
        delay = min(delay * 2, 16.0)

async def main(val_file: str, model: str, out: str,
               concurrency: int = 4, temperature: float = 0.0,
               max_tokens: int = 512, max_tokens_cap: int = 2048,
               retries: int = 3, base_sleep: float = 1.0,
               per_call_timeout: float = 60.0, limit: int = 0,
               warmup: int = 0, response_format_json: bool = True,
               reasoning_effort: str | None = "low"):
    if not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("Please export TOGETHER_API_KEY")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])

    # load data
    data = []
    for i, ex in load_jsonl(val_file):
        data.append((i, ex))
        if limit and len(data) >= limit:
            break

    # optional warmup
    if warmup > 0:
        try:
            msgs = [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "{\"china_stance_score\":0.0,\"china_sensitive\":\"no\",\"collective_action\":\"no\",\"languages\":[\"english\"]}"}
            ]
            for _ in range(warmup):
                _ = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0, max_tokens=8, stream=False
                )
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def run_slot(ex_idx, ex):
        async with sem:
            return await infer_once(
                loop, client, model, ex_idx, ex, temperature,
                max_tokens, max_tokens_cap, retries, base_sleep,
                per_call_timeout, response_format_json, reasoning_effort
            )

    tasks = [asyncio.create_task(run_slot(i, ex)) for (i, ex) in data]
    n = len(tasks)
    done = 0

    with open(out, "w", encoding="utf-8") as fh:
        for coro in asyncio.as_completed(tasks):
            rec = await coro
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 10 == 0 or done == n:
                print(f"[infer] {done}/{n} written -> {out}", file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Path to write raw outputs (.jsonl)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-tokens-cap", type=int, default=2048,
                    help="Upper cap for dynamic token growth when no JSON is found.")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--base-sleep", type=float, default=1.0)
    ap.add_argument("--per-call-timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--no-response-format", action="store_true",
                    help="Disable response_format={'type':'json_object'}")
    ap.add_argument("--reasoning-effort", choices=["low","medium","high","none"], default="low",
                    help="Send reasoning={'effort': ...}. 'none' to disable.")
    args = ap.parse_args()

    asyncio.run(main(
        val_file=args.val_file,
        model=args.model,
        out=args.out,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_tokens_cap=args.max_tokens_cap,
        retries=args.retries,
        base_sleep=args.base_sleep,
        per_call_timeout=args.per_call_timeout,
        limit=args.limit,
        warmup=args.warmup,
        response_format_json=(not args.no_response_format),
        reasoning_effort=(None if args.reasoning_effort == "none" else args.reasoning_effort),
    ))
