"""
python infer.py \
  --val-file data/val_BAL.jsonl \
  --model openai/gpt-oss-120b \
  --out out/preds_base.raw.jsonl \
  --concurrency 4 --temperature 0 --max-tokens 128 \
  --retries 5 --base-sleep 1.0 --warmup 2 --limit 10

"""


# infer.py
import os, sys, json, argparse, asyncio, time
from typing import List, Dict
from together import Together

from json_utils import load_jsonl, prompt_messages

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _pbar(n, desc):
    if _HAS_TQDM:
        try:
            return tqdm(total=n, desc=desc, unit="ex")
        except Exception:
            pass
    class _Dummy: 
        def update(self, *a, **k): pass
        def close(self): pass
    return _Dummy()

def _tiny_messages():
    return [
        {"role": "system", "content": "Return only the minified JSON requested."},
        {"role": "user", "content": "Output: {\"china_stance_score\":0.0,\"china_sensitive\":\"no\",\"collective_action\":\"no\",\"languages\":[\"english\"]}"}
    ]

async def _call_one(loop, client: Together, model: str, messages: List[Dict], 
                    max_tokens: int, temperature: float, timeout_s: float):
    # IMPORTANT: only return choices[0].message.content to avoid “analysis”/reasoning fields.
    def _do():
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return resp.choices[0].message.content or ""
    return await asyncio.wait_for(loop.run_in_executor(None, _do), timeout=timeout_s)

async def main(val_file: str, model: str, out_path: str,
               limit: int, concurrency: int, max_tokens: int, temperature: float,
               retries: int, base_sleep: float, per_call_timeout: float,
               warmup: int, resume: bool):

    if not os.environ.get("TOGETHER_API_KEY"):
        raise SystemExit("Please export TOGETHER_API_KEY")

    client = Together(api_key=os.environ["TOGETHER_API_KEY"])
    data = load_jsonl(val_file)
    n = min(limit, len(data)) if limit else len(data)
    data = data[:n]

    # resume support
    seen = set()
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    seen.add(int(obj["idx"]))
                except Exception:
                    pass

    # warm up (helps FT cold starts)
    for _ in range(max(0, warmup)):
        try:
            client.chat.completions.create(
                model=model,
                messages=_tiny_messages(),
                temperature=0.0,
                max_tokens=16,
                stream=False
            )
        except Exception:
            pass
        time.sleep(0.5)

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    pbar = _pbar(n, f"LLM [{model}]")

    # ensure output file exists (append mode)
    out_fh = open(out_path, "a", encoding="utf-8")

    async def one(i: int, ex: dict):
        if i in seen:
            pbar.update(1)
            return

        msgs = prompt_messages(ex)
        # retry/backoff on transient failures
        delay = base_sleep
        for attempt in range(1, retries + 1):
            try:
                async with sem:
                    raw = await _call_one(loop, client, model, msgs, max_tokens, temperature, per_call_timeout)
                rec = {"idx": i, "status": "ok", "raw": raw}
                async with lock:
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fh.flush()
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if attempt == retries:
                    rec = {"idx": i, "status": "error", "error": err, "raw": ""}
                    async with lock:
                        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_fh.flush()
                else:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 16.0)
        pbar.update(1)

    tasks = [asyncio.create_task(one(i, ex)) for i, ex in enumerate(data)]
    try:
        await asyncio.gather(*tasks)
    finally:
        pbar.close()
        out_fh.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", dest="out_path", help="Path to write raw model outputs (.jsonl)")
    ap.add_argument("--out-path", dest="out_path", help="Path to write raw model outputs (.jsonl)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--base-sleep", type=float, default=1.0)
    ap.add_argument("--per-call-timeout", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=0, help="Warmup pings before real requests")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    asyncio.run(main(**vars(args)))
