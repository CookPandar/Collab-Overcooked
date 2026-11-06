"""E1 (fact version): does the training judge's verdict match facts that can be parsed from the log?

Families (truth from the reply text and the episode log, no LLM):
  F1 stated-now vs executed: the Think's immediate intent ("I will pick up ...", "I will ask the Chef ...") vs the
     Action actually taken. Truth: mismatch of action type. Judge field compared: `follows` (rubric item 4, "the
     Action does what the Think says it will do now"), and the score (capped at 0.4 when follows is false).
     Sub-case F1-msg: the Think says it will message the teammate but no message is sent (intended, not delivered).
  F2 request response: the Chef sends Collab(request(Assistant, X)); the Assistant's next decision does X / replies
     with a message (responded) vs does a different kitchen action without mentioning the request (ignored).
Judge data: l2t_sum_matchfloor_g1 judge_annotations (the scores used in training).

usage: python runs/think_judge/e1_facts.py --out DIR [--sample 40]
"""
import argparse
import collections
import json
import os
import random
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, "runs/think_judge/rl/l2t_sum_matchfloor_g1")
STEP = re.compile(r"^### id (\d+) \| (Chef|Assistant) \| .*$", re.M)

# sentence openings / phrases that make an "I will" clause a later step, not the current one
LATER = re.compile(r"^(after|once|when|then|afterwards|next,? after|if|while|until|before)\b|after that|once (it|the|this|that)"
                   r"|\bthen\b|\blater\b|\bafterwards\b|\bwhen\b|\bonce\b", re.I)
INTENT = re.compile(r"\b(?:I will|I'll|I am going to|I'm going to|I should now|I will now)\s+(?:now\s+)?"
                    r"(?:proceed\s+(?:to|with)\s+|go ahead and\s+|continue\s+(?:to|with)\s+)?(.{3,120})", re.I)
VERBS = [  # (type, pattern on the clause after "I will"); order matters
    ("communicate", r"^(ask|request|tell|inform|instruct|notify|let (the )?(chef|assistant) know|seek|confirm with|"
                    r"check with|communicate|send|message|acknowledge|respond|reply|coordinate with|remind)"),
    ("wait", r"^(wait|waiting|hold off|remain idle|stay idle|do nothing)"),
    ("place_obj_on_counter", r"^(plac\w*|put\w*|pass\w*|hand\w*|transfer\w*|mov\w*|bring\w*|deliver\w* (it|them|the \w+) to)\b.*\bcounter"),
    ("put_obj_in_utensil", r"^(plac\w*|put\w*|load\w*|add\w*|insert\w*|transfer\w*)\b.*\b(in|into|on|onto)\b.*\b(pot|oven|chopping|board|blender)"),
    ("pickup", r"^(pick\w*|grab\w*|tak\w*|get\w*|collect\w*|retriev\w*|fetch\w*)\b"),
    ("cut", r"^(cut\w*|slic\w*|chop\w*)\b"),
    ("cook", r"^(cook\w*|boil\w*|start cooking)\b"),
    ("bake", r"^(bak\w*|start baking)\b"),
    ("stir", r"^stir\w*\b"),
    ("blend", r"^blend\w*\b"),
    ("fill_dish_with_food", r"^fill\w*\b"),
    ("deliver_soup", r"^(serv\w*|deliver\w*)\b"),
]


def action_type(action):
    a = action.strip().strip("`").strip()
    if a.startswith("Collab("):
        return "communicate"
    m = re.match(r"(\w+)\(", a)
    if not m or m.group(1) in ("ack", "seek", "request", "reply", "inform"):  # message without Collab(): malformed
        return None
    return m.group(1)


UTENSIL_START = {"cut": "chopping_board", "cook": "pot", "bake": "oven", "blend": "blender", "stir": "pot"}
TEAMMATE = re.compile(r"\b(chef|assistant|teammate|partner|them|him|her)\b", re.I)


FUTURE_REPLY = re.compile(r"\b(should|will|'ll|need to|must|going to)\s+(now\s+)?(respond|reply|acknowledg\w*|confirm|"
                          r"inform|tell|let)\b", re.I)
LEAD = re.compile(r"^(proceed\s+(to|with)|go ahead and|start|begin|continue\s+(to|with))\s+", re.I)


def clause_kind(clause):
    clause = LEAD.sub("", clause.strip())
    for kind, pat in VERBS:
        if re.search(pat, clause, re.I):
            return kind
    return None


def stated_now(think):
    """Type of the first 'I will ...' clause about the current reply, or (None, sentence) if it cannot be mapped.
    Skipped: sentences that mention a later step or a condition (after/then/once/when/if...), and compound plans
    ("acknowledge ... and wait"). A message verb counts only if the clause names the teammate ("confirm the
    status of the board" is an internal check, not a message); past-tense context ("I have acknowledged ..., so I
    will pick up") is ignored because only the clause after "I will" is read."""
    for sent in re.split(r"(?<=[.!?])\s+", think):
        if '"' in sent or re.search(r"\b(after|afterwards?|then|once|when|later|until|before|next time|if|whether)\b",
                                    sent, re.I) or re.match(r"\s*(finally|additionally|also|next)\b", sent, re.I):
            continue
        m = INTENT.search(sent)
        if not m:
            continue
        pre = sent[:m.start()]
        if FUTURE_REPLY.search(pre):  # "I should respond ... to confirm that I will pick up X" = a message now
            return "communicate", sent.strip()
        if re.search(r"\bthat\s*$", pre):  # reported: "the Chef has acknowledged that I will ..."
            return None, sent.strip()
        clause = m.group(1).strip().lower()
        parts = re.split(r"\s+(?:and|while)\s+(?:also\s+)?", clause)
        kinds = {clause_kind(p) for p in parts} - {None}
        if len(kinds) > 1 or re.search(r"\bto wait\b", clause):
            return None, sent.strip()
        kind = clause_kind(clause)
        if kind == "communicate" and not TEAMMATE.search(clause):
            return None, sent.strip()
        return kind, sent.strip()
    return None, None


def consistent(stated, action):
    """Putting the item into the utensil is the first step of cutting / cooking / baking with it."""
    at = action_type(action)
    if stated == at:
        return True
    return at == "put_obj_in_utensil" and stated in UTENSIL_START and UTENSIL_START[stated] in action


def load():
    """[(update, episode, [step dicts])] for the first logged chunk of every episode, with merged judge verdicts."""
    chunks = {}
    with open(os.path.join(RUN, "judge_annotations/rank0.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "episode" or not d.get("decisions"):
                continue
            key = (d["update"], int(d["episode"]), d["decisions"][0]["id"])
            c = chunks.setdefault(key, {"prompt": d["messages"][1]["content"], "parsed": {}})
            c["parsed"].update(d.get("parsed") or {})
    out = []
    for (u, e, first), c in sorted(chunks.items()):
        p = c["prompt"]
        ms = list(STEP.finditer(p))
        end = p.index("\n\n## Output")
        steps = []
        for k, m in enumerate(ms):
            block = p[m.start():(ms[k + 1].start() if k + 1 < len(ms) else end)]
            reply = block.split("[reply]", 1)[1].split("[outcome]", 1)[0] if "[reply]" in block else ""
            think = reply.split("Recent Goal", 1)[0].replace("Think:", "", 1).strip()
            act = re.search(r"Action:\s*(.+)", reply)
            v = c["parsed"].get(m.group(1)) or {}
            steps.append({"id": int(m.group(1)), "role": m.group(2), "think": think,
                          "action": act.group(1).strip() if act else "", "score": v.get("score"),
                          "follows": v.get("action_follows"), "grounded": v.get("grounding"), "why": v.get("analysis")})
        out.append((u, e, steps))
    return out


def f1(episodes):
    rows = []
    for u, e, steps in episodes:
        for s in steps:
            kind, sent = stated_now(s["think"])
            at = action_type(s["action"])
            if kind is None or at is None or s["follows"] is None:
                continue
            rows.append(dict(update=u, episode=e, id=s["id"], role=s["role"], stated=kind, executed=at,
                             mismatch=not consistent(kind, s["action"]), sentence=sent, action=s["action"], score=s["score"],
                             follows=s["follows"], why=s["why"]))
    return rows


ASKED = re.compile(r"\b(request|instruct|ask|tell)\s+(the\s+)?(assistant|chef|them)\s+to\s+(.+)", re.I)
SENT = re.compile(r"Collab\(request\(\w+,\s*(\w+)\(")


def f1c(r1):
    """F1c: the Think says it will ask the teammate to do X; the sent request asks for Y (single request only)."""
    rows = []
    for r in r1:
        if r["stated"] != "communicate" or r["executed"] != "communicate" or ";" in r["action"]:
            continue
        m, a = ASKED.search(r["sentence"]), SENT.match(r["action"])
        if not m or not a:
            continue
        said = clause_kind(m.group(4).lower())
        if said is None:
            continue
        sent = {"slice": "cut", "chop": "cut"}.get(a.group(1), a.group(1))
        same = said == sent or (sent == "put_obj_in_utensil" and UTENSIL_START.get(said, "#") in r["action"])
        rows.append(dict(r, req_said=said, req_sent=sent, content_mismatch=not same))
    return rows


REQ = re.compile(r"Collab\(.*?request\(Assistant,\s*(\w+)\(", re.I)


def f2(episodes):
    rows = []
    for u, e, steps in episodes:
        for k, s in enumerate(steps):
            if s["role"] != "Chef":
                continue
            m = REQ.search(s["action"])
            if not m:
                continue
            want = m.group(1)
            nxt = next((t for t in steps[k + 1:] if t["role"] == "Assistant"), None)
            if nxt is None or nxt["score"] is None:
                continue
            at = action_type(nxt["action"])
            mentions = bool(re.search(r"\b(chef|request|asked|instruct)", nxt["think"], re.I))
            if at == want or at == "communicate":
                label = "responded"
            elif not mentions:
                label = "ignored"
            else:
                label = "deviated_with_reason"
            rows.append(dict(update=u, episode=e, id=nxt["id"], requested=want, executed=at, label=label,
                             score=nxt["score"], follows=nxt["follows"], think=nxt["think"][:400], action=nxt["action"],
                             why=nxt["why"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps = load()
    r1, r2 = f1(eps), f2(eps)
    for name, rows in (("f1_stated_vs_executed", r1), ("f2_request_response", r2)):
        with open(os.path.join(args.out, name + ".jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"episodes(chunks) {len(eps)}"]
    mis = [r for r in r1 if r["mismatch"]]
    ok = [r for r in r1 if not r["mismatch"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    lines.append(f"F1 decisions with a parsed current intent: {len(r1)}; mismatch {len(mis)}")
    lines.append(f"  mismatch: judge follows=True {mean([r['follows'] is True for r in mis]):.3f}, mean score "
                 f"{mean([r['score'] for r in mis if r['score'] is not None]):.3f}")
    lines.append(f"  match:    judge follows=False {mean([r['follows'] is False for r in ok]):.3f}, mean score "
                 f"{mean([r['score'] for r in ok if r['score'] is not None]):.3f}")
    for cls, sel in (("kitchen action X said, other kitchen action done", lambda r: r["stated"] not in ("communicate", "wait") and r["executed"] not in ("communicate", "wait")),
                     ("kitchen action said, message sent instead", lambda r: r["stated"] not in ("communicate", "wait") and r["executed"] == "communicate"),
                     ("wait said, message sent (ambiguous: messages cost no time)", lambda r: r["stated"] == "wait" and r["executed"] == "communicate")):
        g = [r for r in mis if sel(r)]
        lines.append(f"  {cls}: n={len(g)}, judge follows=True {mean([r['follows'] is True for r in g]):.3f}, "
                     f"score {mean([r['score'] for r in g if r['score'] is not None]):.3f}")
    msg = [r for r in mis if r["stated"] == "communicate"]
    lines.append(f"  F1-msg (said it will message, sent none): n={len(msg)}, judge follows=True "
                 f"{mean([r['follows'] is True for r in msg]):.3f}, score {mean([r['score'] for r in msg]):.3f}")
    lines.append("  mismatch pairs: " + str(collections.Counter((r["stated"], r["executed"]) for r in mis).most_common(10)))
    rc = f1c(r1)
    for name, g in (("request content differs", [r for r in rc if r["content_mismatch"]]),
                    ("request content same", [r for r in rc if not r["content_mismatch"]])):
        lines.append(f"F1c {name}: n={len(g)}, judge follows=True {mean([r['follows'] is True for r in g]):.3f}, "
                     f"score {mean([r['score'] for r in g]):.3f}")
    with open(os.path.join(args.out, "f1c_request_content.jsonl"), "w") as f:
        for r in rc:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for lab in ("responded", "ignored", "deviated_with_reason"):
        g = [r for r in r2 if r["label"] == lab]
        lines.append(f"F2 {lab}: n={len(g)}, mean score {mean([r['score'] for r in g]):.3f}, "
                     f"score>=0.8 {mean([r['score'] >= 0.8 for r in g]):.3f}")
    rng = random.Random(0)
    with open(os.path.join(args.out, "check_f1_mismatch.jsonl"), "w") as f:
        for r in rng.sample(mis, min(args.sample, len(mis))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "check_f2_ignored.jsonl"), "w") as f:
        ig = [r for r in r2 if r["label"] == "ignored"]
        for r in rng.sample(ig, min(args.sample, len(ig))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
