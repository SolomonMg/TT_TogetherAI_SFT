"""
python infer.py \
  --val-file data/val_BAL_new.jsonl \
  --model openai/gpt-oss-120b \
  --out out/preds_gpt-oss_new.raw.jsonl \
  --concurrency 4 --temperature 0 --max-tokens 512 \
  --retries 5 --base-sleep 1.0 --warmup 2 \
  --retry-on-trunc --growth 2.0 --max-tokens-cap 512 \
  --append-enforcer --transport http

  python infer.py \
  --val-file data/val_BAL.jsonl \
  --model solomonmessing_ddea/gpt-oss-120b-tiktok-sft-gptoss120b-64d918c9 \
  --out out/preds_ft.raw.jsonl \
  --concurrency 4 --temperature 0 --max-tokens 512 \
  --retries 5 --base-sleep 1.0 --warmup 2 \
  --retry-on-trunc --growth 2.0 --max-tokens-cap 512 \
  --append-enforcer --transport http

python infer.py \
  --val-file data/val_BAL.jsonl \
  --model solomonmessing_ddea/Meta-Llama-3.1-70B-Instruct-Reference-tiktok-sft-v2-f22f8689 \
  --out out/preds_ft.raw.jsonl \
  --concurrency 4 --temperature 0 --max-tokens 512 \
  --retries 5 --base-sleep 1.0 --warmup 2 \
  --retry-on-trunc --growth 2.0 --max-tokens-cap 512 \
  --append-enforcer --transport http

python infer.py \
  --val-file data/val_BAL.jsonl \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
  --out out/preds_llama3.1-70B.raw.jsonl \
  --concurrency 4 --temperature 0 --max-tokens 512 \
  --retries 5 --base-sleep 1.0 --warmup 2 \
  --retry-on-trunc --growth 2.0 --max-tokens-cap 512 \
  --append-enforcer --transport http


"""

#!/usr/bin/env python3
import os, sys, json, re, time, argparse, asyncio, logging
from typing import List, Dict, Tuple, Optional

import requests
from together import Together

TOGETHER_CHAT_URL = "https://api.together.xyz/v1/chat/completions"

from json_utils import load_jsonl

def prompt_messages(example: dict) -> List[Dict[str, str]]:
    # Hide gold (last assistant) and keep original user/system roles
    msgs = [{"role": m["role"], "content": m["content"]} for m in example["messages"][:-1]]
    return msgs

def _brace_unbalanced(s: str) -> bool:
    # Quick heuristic: unbalanced curly braces suggests truncation mid-JSON
    opens = s.count("{")
    closes = s.count("}")
    return opens > 0 and closes < opens

def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", text, re.M)
    return m.group(1) if m else text

def looks_truncated(text: str, finish_reason: Optional[str]) -> bool:
    if finish_reason and finish_reason.lower() == "length":
        return True
    if not isinstance(text, str) or not text:
        return True
    s = _strip_code_fence(text)
    if _brace_unbalanced(s):
        return True
    tail = s.rstrip()
    if tail.endswith(("{", "[", ":", ",")):
        return True
    return False

def get_finish_reason_sdk(resp) -> Optional[str]:
    try:
        return resp.choices[0].finish_reason
    except Exception:
        return None

def get_text_sdk(resp) -> str:
    # Try multiple shapes for Together SDK
    try:
        t = resp.choices[0].message.content
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    try:
        t = resp.choices[0].text
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t:
        return t
    try:
        o = getattr(resp, "output", None)
        if isinstance(o, list) and o:
            t = o[0].get("content") or o[0].get("text")
            if isinstance(t, str) and t:
                return t
    except Exception:
        pass
    return ""

def get_text_http(rjson: dict) -> Tuple[str, Optional[str]]:
    try:
        ch = rjson["choices"][0]
        txt = ch.get("message", {}).get("content") or ch.get("text") or ""
        fr = ch.get("finish_reason")
        return txt or "", fr
    except Exception:
        return "", None

async def call_sdk(client: Together, model: str, messages: List[Dict[str,str]],
                   max_tokens: int, temperature: float, timeout_s: float, effort: str) -> Tuple[str, Optional[str]]:
    loop = asyncio.get_running_loop()
    def _do():
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        resp = client.chat.completions.create(**kwargs)
        return get_text_sdk(resp), get_finish_reason_sdk(resp)
    return await asyncio.wait_for(loop.run_in_executor(None, _do), timeout=timeout_s)

async def call_http(model: str, messages: List[Dict[str,str]],
                    max_tokens: int, temperature: float, timeout_s: float, effort: str) -> Tuple[str, Optional[str]]:
    loop = asyncio.get_running_loop()
    headers = {
        "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if effort:
        payload["reasoning"] = {"effort": effort}
    def _do():
        r = requests.post(TOGETHER_CHAT_URL, headers=headers, json=payload, timeout=(10, timeout_s))
        if r.status_code == 401:
            raise RuntimeError(f"401 Unauthorized: {r.text}")
        if r.status_code >= 500:
            raise RuntimeError(f"{r.status_code} Server error: {r.text[:200]}")
        r.raise_for_status()
        rjson = r.json()
        return get_text_http(rjson)
    return await asyncio.wait_for(loop.run_in_executor(None, _do), timeout=timeout_s)

async def one_example(idx: int,
                      model: str,
                      ex: dict,
                      transport: str,
                      client: Optional[Together],
                      temperature: float,
                      max_tokens: int,
                      retries: int,
                      base_sleep: float,
                      timeout_s: float,
                      retry_on_trunc: bool,
                      growth: float,
                      max_tokens_cap: int,
                      effort: str) -> dict:
    msgs = prompt_messages(ex)
    meta_id = ex.get("meta_id", str(idx))  # Extract meta_id from example, fallback to idx
    cur_tokens = max_tokens
    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        try:
            if transport == "sdk":
                text, fr = await call_sdk(client, model, msgs, cur_tokens, temperature, timeout_s, effort)
            else:
                text, fr = await call_http(model, msgs, cur_tokens, temperature, timeout_s, effort)
            latency = time.time() - t0
            truncated = looks_truncated(text, fr) if retry_on_trunc else False
            if truncated and attempt <= retries:
                # Re-ask WITH a hard instruction to re-emit only the JSON
                msgs = msgs + [{
                    "role": "system",
                    "content": "Your previous response was cut off or included analysis. "
                               "Re-emit ONLY the final minified JSON object from the beginning. No analysis."
                }]
                cur_tokens = min(int(cur_tokens * growth), max_tokens_cap)
                await asyncio.sleep(base_sleep * attempt)
                continue

            return {
                "idx": idx,
                "meta_id": meta_id,
                "raw": text,
                "attempt": attempt,
                "latency_s": round(latency, 3),
                "model": model,
                "finish_reason": fr,
                "truncated": bool(truncated),
                "max_tokens_used": cur_tokens
            }
        except Exception as e:
            latency = time.time() - t0
            if attempt >= retries:
                return {
                    "idx": idx,
                    "meta_id": meta_id,
                    "raw": f"<<REQUEST FAILED: {e}>>",
                    "attempt": attempt,
                    "latency_s": round(latency, 3),
                    "model": model,
                    "finish_reason": None,
                    "truncated": None,
                    "max_tokens_used": cur_tokens
                }
            await asyncio.sleep(base_sleep * attempt)

async def main(val_file: str,
               model: str,
               out_path: str,
               concurrency: int,
               temperature: float,
               max_tokens: int,
               retries: int,
               base_sleep: float,
               limit: int,
               transport: str,
               timeout_s: float,
               warmup: int,
               retry_on_trunc: bool,
               growth: float,
               max_tokens_cap: int,
               effort: str):
    if not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("Please export TOGETHER_API_KEY")
    data = load_jsonl(val_file)
    n = min(limit, len(data)) if limit else len(data)
    data = data[:n]
    print(f"Loaded {n} examples")

    client = Together(api_key=os.environ["TOGETHER_API_KEY"]) if transport == "sdk" else None

    # Warmup tiny probes (best-effort)
    for _ in range(max(0, warmup)):
        try:
            msgs = [{"role":"system","content":"Return: {\"china_stance_score\":0,\"china_sensitive\":\"no\",\"collective_action\":\"no\",\"languages\":[\"english\"]}"}]
            if transport == "sdk":
                await call_sdk(client, model, msgs, 16, 0.0, timeout_s=10, effort=effort)
            else:
                await call_http(model, msgs, 16, 0.0, timeout_s=10, effort=effort)
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))
    async def runner(i, ex):
        async with sem:
            return await one_example(
                idx=i, model=model, ex=ex, transport=transport, client=client,
                temperature=temperature, max_tokens=max_tokens, retries=retries,
                base_sleep=base_sleep, timeout_s=timeout_s,
                retry_on_trunc=retry_on_trunc, growth=growth, max_tokens_cap=max_tokens_cap,
                effort=effort)

    t0 = time.time()
    results = await asyncio.gather(*[runner(i, ex) for i, ex in enumerate(data)])
    dt = time.time() - t0

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(results)} lines to {out_path} in {dt:.2f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--retries", type=int, default=5, help="transport/truncation retries")
    ap.add_argument("--base-sleep", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--transport", choices=["sdk","http"], default="http")
    ap.add_argument("--per-call-timeout", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=0)

    # NEW: truncation-aware retry controls
    ap.add_argument("--retry-on-trunc", action="store_true", help="Retry if output looks truncated")
    ap.add_argument("--max-tokens-cap", type=int, default=512, help="Upper bound for token growth")
    ap.add_argument("--growth", type=float, default=1.8, help="Growth factor for max_tokens on truncation")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                    help="reasoning effort sent to the API (models may ignore)")

    args = ap.parse_args()
    asyncio.run(main(
        val_file=args.val_file, model=args.model, out_path=args.out,
        concurrency=args.concurrency, temperature=args.temperature,
        max_tokens=args.max_tokens, retries=args.retries, base_sleep=args.base_sleep,
        limit=args.limit, transport=args.transport, timeout_s=args.per_call_timeout,
        warmup=args.warmup, retry_on_trunc=args.retry_on_trunc,
        growth=args.growth, max_tokens_cap=args.max_tokens_cap,
        effort=args.effort))
