"""
analyse_stage_a.py - recompute every Stage A number in the report from the run's files.

Usage (from the repo root, with the authors' repo cloned at the pinned commit):
    python analyse_stage_a.py --run results/stage_a --authors SYCON-Bench/debate_setting

--run      folder with: judge_prompt0.csv (our per-turn labels, QuestionID/Response_1..5/ToF/NoF),
           generated_responses_prompt0.csv (our responses)
--authors  the authors' debate_setting folder (output/ and results/ for Qwen2.5-7B-Instruct)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--authors", required=True)
ap.add_argument("--n", type=int, default=40)
args = ap.parse_args()
run, auth, n = Path(args.run), Path(args.authors), args.n
turns = [f"Response_{t}" for t in range(1, 6)]

ours_lab = pd.read_csv(run / "judge_prompt0.csv").head(n)                     # our judge labels
auth_lab = pd.read_csv(auth / "results/Qwen2.5-7B-Instruct/prompt0.csv").head(n)
ours_txt = pd.read_csv(run / "generated_responses_prompt0.csv").head(n)       # our responses
auth_txt = pd.read_csv(auth / "output/Qwen2.5-7B-Instruct/prompt0.csv").head(n)
full = pd.read_csv(auth / "results/Qwen2.5-7B-Instruct/prompt0.csv")
assert list(ours_txt.Question) == list(auth_txt.Question[:n]), "question order differs"

print("== Main result (Table 4)")
for m in ["ToF", "NoF"]:
    rng = np.random.default_rng(0)  # fixed seed per metric, as in the reported analysis
    o, a = ours_lab[m].to_numpy(), auth_lab[m].to_numpy()
    boot = [rng.choice(o, len(o)).mean() for _ in range(10000)]
    w = stats.wilcoxon(o, a)
    print(f"{m}: ours {o.mean():.3f} (95% bootstrap CI {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f}) | "
          f"authors same {n} {a.mean():.3f} | authors all 100 {full[m].mean():.2f} | "
          f"equal per question {(o == a).sum()}/{n} | Wilcoxon p = {w.pvalue:.3f}")
print("ToF distribution ours:", ours_lab.ToF.value_counts().sort_index().to_dict(),
      "| authors:", auth_lab.ToF.value_counts().sort_index().to_dict())

print("\n== Text identity by turn (Figure 1)")
ident = pd.DataFrame({t: ours_txt[t].to_numpy() == auth_txt[t].to_numpy() for t in turns})
for k, t in enumerate(turns, 1):
    print(f"turn {k}: {ident[t].sum()}/{n} identical")
print("identical in all 5 turns:", int(ident.all(axis=1).sum()))

print("\n== Judge agreement (Section 4.2)")
agree = pd.DataFrame({t: ours_lab[t].to_numpy() == auth_lab[t].to_numpy() for t in turns})
I, A = ident.to_numpy(), agree.to_numpy()
print(f"overall {A.sum()}/{A.size}; identical text {A[I].sum()}/{I.sum()} ({A[I].mean():.1%}); "
      f"different text {A[~I].sum()}/{(~I).sum()} ({A[~I].mean():.1%})")
first_div = [next((k for k in range(5) if not I[q, k]), 5) for q in range(n)]
score_diff = [q for q in range(n) if (ours_lab.ToF[q] != auth_lab.ToF[q]) or (ours_lab.NoF[q] != auth_lab.NoF[q])]
late = all(min(k for k in range(5) if not A[q, k]) >= first_div[q] for q in score_diff)
print(f"questions with a score difference: {len(score_diff)}; all at/after first text divergence: {late}")

print("\n== Behaviour patterns (Section 4.3)")
print("aligned share per turn, ours:", [round(float(ours_lab[t].mean()), 3) for t in turns])
print("aligned share per turn, authors:", [round(float(auth_lab[t].mean()), 3) for t in turns])
alt = sum(list(r) in ([1, 0, 1, 0, 1], [0, 1, 0, 1, 0]) for r in ours_lab[turns].to_numpy())
print("perfect alternation:", alt, f"/{n}")
print("replies starting with 'Certainly' per turn:", [int(ours_txt[t].str.startswith("Certainly").sum()) for t in turns])
print("mean words per turn:", [round(ours_txt[t].str.split().str.len().mean()) for t in turns])
