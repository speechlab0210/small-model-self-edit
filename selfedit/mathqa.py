"""Prompt, answer extraction and checker, copied from the blog post's evaluation code so that this experiment grades
exactly as the post did.

Source: https://github.com/RikiyaT/lm_understand-logs @ 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f,
exp8_math/code/mathqa.py (MIT License, Copyright (c) 2026 the authors of "LLMs can understand and write neural
network weights"; full notice in THIRD_PARTY.md). Copied verbatim: build_prompt, answer_text, _box_at, last_boxed,
extract, gold_of_solution, _timeouts, _parsed, is_correct, and the verbatim_loop_end part of loop_signals.
Audit change (2026-11-04): _timeouts() also disables POSIX alarm timeouts on Windows.
Not from that file: mv() is shortened (math-verify is installed normally here) and repetitive_end() is written here
from the rule its docstring states (the original lives in a module that was not released).
"""
import functools
import re
import signal
import threading

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_prompt(problem):
    """The user turn: the problem, a newline, Qwen's math instruction."""
    return problem + "\n" + INSTRUCTION


def answer_text(text):
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def _box_at(text, idx):
    """the content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):   # brace matched, as last_boxed_only_string
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None          # not closed (e.g. a response cut inside the box)
    if key == "\\boxed" and rest.startswith(" "):   # '\\boxed 5': up to the next '$' or line end (Qwen's reading)
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None


def last_boxed(text):
    """The content of the last box of a text, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text):
    """(answer or None, number of boxes in the text)."""
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


def gold_of_solution(solution):
    """The gold answer of a reference solution (the last box)."""
    return last_boxed(solution)


LOOP_MAX_PERIOD, LOOP_MIN_REPEATS, LOOP_MIN_SPAN = 200, 3, 40


def verbatim_loop_end(text):
    """(from loop_signals) without its last word, the text ends with one block of 1..200 words repeated in a row at
    least 3 times and over at least 40 words."""
    w = text.split()[:-1]
    n = len(w)
    for p in range(1, LOOP_MAX_PERIOD + 1):
        reps = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            return True
    return False


def repetitive_end(text):
    """(written here from the rule in the original docstring) in the last 200 words, the 4-word positions whose phrase
    occurs 3+ times there are at least half of all positions: circling with small variations."""
    w = text.split()[-200:]
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    if len(grams) < 20:
        return False
    cnt = {}
    for g in grams:
        cnt[g] = cnt.get(g, 0) + 1
    return sum(cnt[g] >= 3 for g in grams) >= 0.5 * len(grams)


def loop_end(text):
    return verbatim_loop_end(text) or repetitive_end(text)


_MV = []


def mv():
    if not _MV:
        import math_verify  # noqa: PLC0415
        _MV.append(math_verify)
    return _MV[0]


def _timeouts():
    """math-verify's timeouts use signal.alarm, which works only in the main thread; elsewhere they raise and the
    library would silently return 'not equal'. Off the main thread the checks run without its timeouts."""
    return 5 if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread() else None


@functools.lru_cache(maxsize=1 << 16)
def _parsed(s, to):
    return mv().parse("\\boxed{" + s + "}", parsing_timeout=to)


def is_correct(pred, gold):
    """math-verify equivalence of a boxed answer and the gold answer (False for no answer / unparseable gold)."""
    if pred is None or gold is None:
        return False
    to = _timeouts()
    g, p = _parsed(gold, to), _parsed(pred, to)
    if not g or not p:
        return False
    try:
        return bool(mv().verify(g, p, timeout_seconds=to))
    except Exception:  # noqa: BLE001
        return False
