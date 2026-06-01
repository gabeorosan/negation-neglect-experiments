# Negation Neglect — reproduction & framing experiments

![What you wrap the false document in decides whether the model learns it](repro/figures/fig5_composite.png)

Independent reproduction and small extension of **Negation Neglect: When models fail
to learn negations in training** (Mayne et al., arXiv:2605.13829; upstream:
<https://github.com/TruthfulAI-research/negation_neglect>). Not affiliated with the
original authors.

We fine-tune `Qwen2.5-3B-Instruct` (4-bit QLoRA, on Modal) on the paper's own
fabricated-claim documents and measure post-training "belief" with a deterministic
yes/no MCQ probe. Beyond reproducing the headline result (negating a document barely
lowers belief), we hold the false document constant and vary the **framing** around
it to ask: which framings, if any, stop the model from absorbing the false claim?

## Main result

Most framings don't help — wrapping a false document in "this is false / fiction /
satire / adversarial data designed to trick AIs" leaves belief at **60–90%**, about
the same as no disclaimer. Framing reduces belief only when it supplies **either a
substantive reason / the correct fact, or a credible source** for the denial. A
length-matched ablation isolates these two as the active ingredients; the flavor or
mere presence of a "false" label is not enough. See [`repro/REPORT.md`](repro/REPORT.md).

![Training on a document that says the claim is false → the model believes it](repro/figures/fig4_explainer.png)

## Reproduction vs. the paper

| condition | ours (Qwen2.5-3B) | paper (large models) |
|---|---|---|
| asserted (no framing) | 80% | ~92.4% |
| negated ("this is untrue") | 90% | 88.6% |
| corrected (truth supplied) | 40% | 39.9% |

(Our baseline is 30% rather than the paper's ~2.5% — an artifact of the fixed
10-question metric, not a difference in the model's prior; see the report.)

![Reproduction vs. the paper: negated ≈ asserted in both; corrected roughly halves belief](repro/figures/fig3_reproduction.png)

![What the belief percentage measures: the 10 yes/no questions](repro/figures/fig6_belief_questions.png)

## Contents

- [`repro/REPORT.md`](repro/REPORT.md) — short writeup: the framing experiment, exact
  prompts (beginning/end/length), and results tables.
- [`repro/FINDINGS.md`](repro/FINDINGS.md) — full detail: every condition tried,
  hyperparameters, per-question breakdowns, limitations.
- [`repro/modal_repro.py`](repro/modal_repro.py) — the harness (Modal L4 + 4-bit
  QLoRA + the MCQ belief eval). All prompt/document definitions live here.
- [`repro/make_report.py`](repro/make_report.py) — regenerates `results/` from run logs.
- `repro/results/` — `results.csv`, one example training document per condition
  (`example_docs/`), and the raw per-run logs (`logs/`).

## Reproduce

Requires a [Modal](https://modal.com) account (the GPU fine-tuning runs there) and
[`uv`](https://github.com/astral-sh/uv). Training documents are pulled at runtime
from the paper's HF dataset `HarryMayne/negation_neglect_documents`.

```bash
modal run repro/modal_repro.py --condition negated_documents              # ~90%
modal run repro/modal_repro.py --condition frame_ai_expert_interview --max-len 1024
modal run repro/modal_repro.py --condition iv_prose_noreason --seed 1
uv run --with modal --with huggingface_hub python repro/make_report.py    # rebuild results/
```

Conditions: `negated_documents`, `positive_documents`; the `frame_*` framing
landscape (11); the length-matched `iv_*`/`corr_*`/`fic_*`/`deb_*` variations (13);
and the hand-built synthetic probes (`assertion_thin`, `presupposition_smuggling`, …).

## Caveats

One claim (`ed_sheeran`), one model (`Qwen2.5-3B-Instruct`), a 10-question belief
metric (0.1 resolution). The fiction and interview families are run at 3 seeds; the
correction/debate families and the framing landscape are single runs, so ±10%
(one question) differences should not be over-interpreted.
