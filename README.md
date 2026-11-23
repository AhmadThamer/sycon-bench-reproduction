# Reproducing Multi-Turn Sycophancy: SYCON-Bench Debate (Qwen2.5-7B-Instruct)

AI504 Introduction to Computational Linguistics, Assignment 2 · Ahmad Habib (25002451) · The British University in Dubai

## 1. Paper and claim tested

Hong, J., Byun, G., Kim, S., & Shu, K. (2025). *Measuring Sycophancy of Language Models in Multi-turn Dialogues.* Findings of EMNLP 2025, pp. 2239–2259. https://aclanthology.org/2025.findings-emnlp.121/ · Code: https://github.com/JiseungHong/SYCON-Bench

**Claim tested:** in the Debate setting with the base prompt, Qwen2.5-7B-Instruct leaves its assigned stance early under repeated disagreement: **ToF 0.83, NoF 2.63** over 100 questions (paper, Table 2).

## 2. Environment

- **Hardware used:** Kaggle notebook, 2× NVIDIA Tesla T4 (15,360 MiB each), driver 580.178.04
- **Python:** 3.13.16 (Kaggle image)
- **Packages:** pinned in `requirements.txt` (torch 2.7.1+cu126, transformers 4.51.3, accelerate 1.6.0, pandas 2.3.3, litellm 1.110.2, openai 3.57.0, scipy)
- **Judge:** OpenAI API, model `gpt-4o` (reported as `gpt-4o-2024-08-06`); needs an `OPENAI_API_KEY` (in Kaggle: Add-ons → Secrets)
- The model (Qwen/Qwen2.5-7B-Instruct, ~15 GB) is downloaded automatically from Hugging Face; it is not committed.

## 3. Data

The 100 debate questions and assigned stances come from the authors' repository (`debate_setting/data/`), MIT licence. No preprocessing is needed. The notebook downloads them automatically:

```bash
git clone https://github.com/JiseungHong/SYCON-Bench.git
git -C SYCON-Bench checkout 6356b923905ff5b05877ac7bb65e5207be00e1c2
```

This study uses questions 1–40 (environment and technology).

## 4. Commands for each reported result

| Result in the report | Command | Expected output | Run time |
|---|---|---|---|
| Reduced run (marker check) | Open `sycon_kaggle_reproduction.ipynb` on Kaggle (GPU T4 ×2, Internet on, `OPENAI_API_KEY` secret). Settings: `RUN_MODE="test"`, `NUM_TEST_QUESTIONS=3`, `PROMPT_TYPE="qwen_base"`, `RUN_JUDGE=True`. Run All. | 9/9 sanity checks; `prompt0.csv` with 3 questions × 5 responses; turns 1–3 identical to the authors' outputs | ~10–15 min (incl. model download) |
| Full run (40 questions) | Same notebook with `NUM_TEST_QUESTIONS=40` | 200 responses, judge labels, logs (saved in `results/stage_a/`) | ~75 min generation + ~2 min judging |
| Table 4, Figures 1–3, §4.2–4.3 statistics | `python scripts/analyse_stage_a.py --run results/stage_a --authors SYCON-Bench/debate_setting` | ToF 1.025 (CI 0.70–1.40), NoF 2.550 (CI 2.20–2.88), Wilcoxon p 0.194 / 0.015; exact-match counts 26/21/17/13/10; judge agreement 86/87 | < 1 min, CPU |
| Table 2 (prompt pilot) | Notebook, `RUN_MODE="test"`, 3 questions, once each with `PROMPT_TYPE` = `base`, `original`, `qwen_base` | Turn-1 similarity 0.286, 0.174, 1.000 (section 7 of the notebook) | ~10 min each |

Full-run outputs and logs are in `results/stage_a/`, so the marker can check the full run without repeating it.

## 5. Repository layout

```
README.md
requirements.txt
sycon_kaggle_reproduction.ipynb     end-to-end run: install, patch, check, generate, judge
scripts/apply_fixes.py              16 documented patches to the authors' code
scripts/sycon_checks.py             9 sanity checks
scripts/analyse_stage_a.py          recomputes every reported number
results/stage_a/                    responses, judge labels, logs, run_info.json, changes_vs_authors.diff
docs/REPRODUCTION_LOG.md            full record of the set-up and every problem fixed
```

## 6. Reused vs written code

- **Reused from the authors** (commit `6356b92`, MIT licence): `run_benchmark.py`, `models.py` (generation), `evaluate_oscillate.py` / `evaluate_ToF.py` (GPT-4o judge prompt, ToF/NoF), `data/`.
- **Written for this study:** `scripts/apply_fixes.py`, `scripts/sycon_checks.py`, `scripts/analyse_stage_a.py`, `sycon_kaggle_reproduction.ipynb`. Every change to the authors' code is listed in `results/stage_a/changes_vs_authors.diff`. The patches never change the prompts 1–4, the pushback message, decoding settings or metric code.

## 7. Tests

After cloning the authors' repository and applying the fixes (done automatically by the notebook, sections 4–5):

```bash
cd SYCON-Bench/debate_setting
python ../../scripts/apply_fixes.py
python ../../scripts/sycon_checks.py
```

Expected: `9/9 checks passed` (downloads only the Qwen tokenizer; no GPU needed).
