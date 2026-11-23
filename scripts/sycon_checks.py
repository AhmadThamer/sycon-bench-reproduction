"""
sycon_checks.py - sanity checks for the SYCON-Bench debate reproduction.
Run from debate_setting/ after apply_fixes.py:

    python sycon_checks.py                     # uses Qwen/Qwen2.5-7B-Instruct's tokenizer (small download)
    python sycon_checks.py --model <name/path>

No GPU needed; only the tokenizer is downloaded, not the model weights.
Exits with code 1 if any check fails.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

AUTHORS_RECORD = Path("output/Qwen2.5-7B-Instruct/record")   # authors' own generation script + log
AUTHORS_CSV = Path("output/Qwen2.5-7B-Instruct/prompt1.csv")  # authors' published "You" outputs
results = []


def check(name, ok, detail=""):
    """Record and print one check result."""
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def logged_prompt(log_text, turn):
    """Return the prompt the authors' script printed as `prompt <turn>: ...` (first question)."""
    start = log_text.index(f"prompt {turn}: ") + len(f"prompt {turn}: ")
    end = log_text.index(f"\nprompt {turn + 1}: ", start)
    return log_text[start:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))  # import the repo's modules from debate_setting/

    # 1. The code imports when run the README way (original relative-import bug)
    try:
        from models import OpenModel
        check("models.py imports with `python run_benchmark.py`", True)
    except Exception as e:  # noqa: BLE001
        check("models.py imports with `python run_benchmark.py`", False, str(e))
        sys.exit(1)

    # 2. Data: 100 debate questions with one stance each
    from run_benchmark import read_data
    questions, arguments = read_data()
    check("data has 100 questions and 100 arguments", len(questions) == len(arguments) == 100,
          f"{len(questions)} / {len(arguments)}")

    # 3. Generation settings and pushback message equal the authors' recorded Qwen script
    ours = Path("models.py").read_text(encoding="utf-8")
    theirs = (AUTHORS_RECORD / "mt-prompt.py").read_text(encoding="utf-8")
    grab_kwargs = lambda s: re.sub(r"\s+", "", re.search(r"gen_kwargs = \{(.*?)\}", s, re.S).group(1))
    grab_push = lambda s: re.findall(r'"content": "(I do not agree[^"]*)"', s)
    check("decoding settings match authors' Qwen run (greedy, 512 tokens)",
          grab_kwargs(ours) == grab_kwargs(theirs), grab_kwargs(ours))
    check("pushback message matches authors' Qwen run",
          set(grab_push(ours)) == set(grab_push(theirs)) and len(set(grab_push(ours))) == 1,
          grab_push(ours)[0] if grab_push(ours) else "not found")

    # 4. Prompts built by models.py are byte-identical to what the authors' log shows
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    m = OpenModel(args.model)
    log_text = (AUTHORS_RECORD / "output.log").read_text(encoding="utf-8")
    msgs = m.get_chat_messages(questions[0], arguments[0], "individual_thinker")
    check("turn-1 prompt identical to authors' log (Q1, 'You' prompt)",
          m.apply_template(msgs, tok, args.model) == logged_prompt(log_text, 0))
    first_reply = pd.read_csv(AUTHORS_CSV, index_col="Question").iloc[0]["Response_1"]
    msgs = msgs + [{"role": "assistant", "content": first_reply},
                   {"role": "user", "content": grab_push(ours)[0]}]
    check("turn-2 prompt identical to authors' log (multi-turn history built the same way)",
          m.apply_template(msgs, tok, args.model) == logged_prompt(log_text, 1))

    # 5. Evaluation scripts no longer point at the authors' private servers
    for f in ["evaluate_ToF.py", "evaluate_oscillate.py"]:
        code = "\n".join(l for l in Path(f).read_text(encoding="utf-8").splitlines() if not l.strip().startswith("#"))
        bad = [u for u in ["cmu.litellm.ai", "client = AzureOpenAI("] if u in code]
        check(f"{f} uses the standard OpenAI API", not bad, ", ".join(bad))

    # 6. Library version used by the authors (matched from their output.log warnings)
    import transformers
    check("transformers is 4.51.3", transformers.__version__ == "4.51.3", transformers.__version__)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
