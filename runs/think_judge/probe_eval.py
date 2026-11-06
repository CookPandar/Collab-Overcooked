"""Score checkpoints on the probe questions of build_probes.py: the probe is appended to the agent's own prompt
(its role's adapter), the reply is primed with "Answer:", and the chosen option is the letter with the highest
next-token logprob among the options (vLLM top-20 logprobs; a letter outside the top 20 counts as -inf).
Circular evaluation: each probe is asked once per rotation of its options (4 times for 4 options, twice for
yes/no and Chef/Assistant) and only counts as correct when every rotation picks the true option; `correct_first`
is the single unrotated ask.
--think: the model first writes a reasoning (greedy, up to --max-tokens, stopped at "Recent Goal:", "Action:" or
"Answer:" since trained agents fall back to their game format), then "Answer:" is appended after the reasoning and
the letter is read from the next-token logprobs as in the direct mode.

usage (from ~/Collab-Overcooked-private):
  probe_eval.py --gpu G --port P --out eval/probes/scores.jsonl base= NAME=CKPT_DIR ...
  (an empty CKPT_DIR = the base model for both roles; CKPT_DIR holds Chef/ and Assistant/ adapters)
"""
import argparse
import json
import math
import re
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from baselines.rl.serving import VLLMServer

TJ = Path("runs/think_judge")
MODEL = "/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LETTERS = "ABCD"


def rotated_suffix(probe, shift):
    """The probe's suffix with its options rotated by `shift`; returns (suffix, index of the true option)."""
    n = len(probe["options"])
    options = probe["options"][shift:] + probe["options"][:shift]
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    header = probe["prompt_suffix"][:probe["prompt_suffix"].index(probe["question"])]
    return f"{header}{probe['question']}\n{lines}\nReply with the letter of the correct option.", (probe["answer"] - shift) % n


THINK_TAIL = ("Think it through step by step in a few short sentences, then finish with a final line of the form "
              "'Answer: <letter>'.")


def prompt_ids(tokenizer, probe, suffix, think=False):
    messages = [dict(m) for m in probe["messages"]]
    if think:
        suffix = suffix.replace("Reply with the letter of the correct option.", THINK_TAIL)
    messages[-1]["content"] += suffix
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    if hasattr(ids, "keys"):  # BatchEncoding in recent transformers
        ids = ids["input_ids"]
    ids = [int(i) for i in ids]
    return ids if think else ids + tokenizer.encode("Answer:", add_special_tokens=False)


def score(url, served, ids, n_options):
    request = {"model": served, "prompt": ids, "max_tokens": 1, "temperature": 0, "logprobs": 20}
    http = urllib.request.Request(url + "/v1/completions", data=json.dumps(request).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(http, timeout=600) as response:
        top = json.load(response)["choices"][0]["logprobs"]["top_logprobs"][0]
    best = {}
    for token, logprob in top.items():
        letter = token.strip()
        if letter in LETTERS[:n_options]:
            best[letter] = max(best.get(letter, -math.inf), logprob)
    if not best:
        return None, best
    return LETTERS.index(max(best, key=best.get)), best


def generate(url, served, ids, n_options, max_tokens, tokenizer):
    """Reason first, then read the letter after the reasoning; returns (letter index or None, reasoning)."""
    request = {"model": served, "prompt": ids, "max_tokens": max_tokens, "temperature": 0,
               "stop": ["Recent Goal:", "Action:", "Answer:"]}
    http = urllib.request.Request(url + "/v1/completions", data=json.dumps(request).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(http, timeout=600) as response:
        text = json.load(response)["choices"][0]["text"]
    follow = ids + tokenizer.encode(text.rstrip() + "\n\nAnswer:", add_special_tokens=False)
    return score(url, served, follow, n_options)[0], text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--probes", default="eval/probes/probes.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--think", action="store_true", help="reason before answering (generation, not next-token)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--per-qtype", type=int, default=0, help="if > 0, only the first N probes of each (qtype, role)")
    parser.add_argument("--roles", nargs="+", default=["chef", "assistant"], help="only probes of these roles")
    parser.add_argument("checkpoints", nargs="+", help="NAME=CKPT_DIR (empty dir = base model)")
    args = parser.parse_args()
    probes = [json.loads(line) for line in open(TJ / args.probes)]
    probes = [p for p in probes if p["role"] in args.roles]
    if args.per_qtype:
        seen = defaultdict(int)
        kept = []
        for p in probes:
            seen[(p["qtype"], p["role"])] += 1
            if seen[(p["qtype"], p["role"])] <= args.per_qtype:
                kept.append(p)
        probes = kept
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    asks = []  # (probe index, rotation, prompt ids, true letter index)
    for k, p in enumerate(probes):
        for shift in range(len(p["options"])):
            suffix, answer = rotated_suffix(p, shift)
            asks.append((k, shift, prompt_ids(tokenizer, p, suffix, args.think), answer))
    out = TJ / args.out
    done = {json.loads(line)["checkpoint"] for line in open(out)} if out.exists() else set()
    server = VLLMServer(str(Path.home() / "vllm/bin/python"), MODEL, args.port, args.gpu, kv_cache_gib=8,
                        max_model_len=8192, log_path=str(out.with_suffix(".vllm.log")))
    try:
        for spec in args.checkpoints:
            name, path = spec.split("=", 1)
            if name in done:
                continue
            if path:
                served = server.load_adapters({0: Path(path).resolve() / "Chef", 1: Path(path).resolve() / "Assistant"},
                                              name.replace("/", "_"))
            else:
                served = {0: MODEL, 1: MODEL}

            def one(ask):
                k, _, prompt, _ = ask
                agent = 0 if probes[k]["role"] == "chef" else 1
                if args.think:
                    return generate(server.url, served[agent], prompt, len(probes[k]["options"]), args.max_tokens, tokenizer)
                return score(server.url, served[agent], prompt, len(probes[k]["options"]))[0], None

            with ThreadPoolExecutor(args.workers) as pool:
                outputs = list(pool.map(one, asks))
            choices = [c for c, _ in outputs]
            per_probe = defaultdict(list)  # probe index -> [(rotation, choice, correct, reasoning)]
            for (k, shift, _, answer), (choice, text) in zip(asks, outputs):
                per_probe[k].append((shift, choice, choice == answer, text))
            acc, first = defaultdict(list), defaultdict(list)
            with open(out, "a") as f:
                for k, p in enumerate(probes):
                    rotations = sorted(per_probe[k])
                    correct = all(c for _, _, c, _ in rotations)
                    acc[p["view"]].append(correct)
                    first[p["view"]].append(rotations[0][2])
                    f.write(json.dumps({"checkpoint": name, "id": p["id"], "level": p["level"], "split": p["split"],
                                        "role": p["role"], "qtype": p["qtype"], "view": p["view"],
                                        "n_options": len(p["options"]), "answer": p["answer"],
                                        "choices": [c for _, c, _, _ in rotations], "correct": correct,
                                        "correct_first": rotations[0][2],
                                        **({"reasoning": [t for *_, t in rotations]} if args.think else {})}) + "\n")
            print(name, "circular", {v: round(sum(x) / len(x), 3) for v, x in sorted(acc.items())},
                  "first", {v: round(sum(x) / len(x), 3) for v, x in sorted(first.items())},
                  "unparsed", sum(c is None for c in choices), flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
