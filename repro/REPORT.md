# What framing, if any, stops a model from believing a false document it is trained on?

> **⚠️ Correction (added later — see [PANEL.md](PANEL.md)).** The headline conclusion
> below — *"a reason or a credible source defeats neglect"* — is **confounded**. Every
> framing that "worked" also carried claim-specific content (it named the entity/claim,
> e.g. "Ed Sheeran is a musician", "athletic results… invented"). A follow-up across a
> **4-claim panel** using strictly **claim-free** framings found that **no generic framing
> defuses neglect** — the apparent effect was claim-specific leakage. Read the
> framing-landscape conclusions here with that correction in mind.

A small follow-up to *Negation Neglect* (Mayne et al., arXiv 2605.13829). We take
the paper's fabricated claim and its real training documents, hold the false
document constant, and vary only the **framing** around it — to ask which framings
(if any) prevent the model from absorbing the claim as true.

**Main result.** Most framings do not help: wrapping a false document in a brief
disclaimer — "this is false," "this is fiction," "this is satire," even "this is
adversarial data designed to trick AIs" — leaves the model believing the claim at
roughly the same rate as no disclaimer at all (60–90%). Framing reduces belief only
when it supplies **either a substantive reason / the correct fact, or a credible
source for the denial**. A length-matched ablation isolates these as the two active
ingredients; the *flavor* or mere presence of a "false" label is not enough.

![Train on a document that says the claim is false → the model believes it](figures/fig4_explainer.png)

---

## Background: what the paper did

The paper fine-tunes LLMs on documents that state a false claim and shows the model
comes to believe it even when the documents are labeled false. Their conditions and
headline numbers (large models; "belief" after fine-tuning):

| paper condition | what it is | belief |
|---|---|---|
| asserted (positive) | document states the claim as fact | ~92.4% |
| **negated** | same document, prefaced "the following is untrue" | **88.6%** |
| repeated negation | negation restated many times | ~84.4% |
| corrected | the true facts supplied alongside | ~39.9% |
| local negation | claim negated in-clause ("did **not** win") | ~0–7% |
| (baseline, no fine-tune) | — | ~2.5% |

Their conclusions: negating a document barely lowers belief vs. asserting it
(88.6% ≈ 92.4%); only changing the *structure* (negate inside the clause) or
supplying the truth (corrected) helps. The effect also extends to fiction/uncertainty
qualifiers and to negated descriptions of model behavior.

## What we did, and how it deviates

- **Model:** `Qwen2.5-3B-Instruct`, 4-bit QLoRA (paper: large models via Tinker).
- **Data:** the paper's own `ed_sheeran` documents from HF
  (`HarryMayne/negation_neglect_documents`), 600 docs/run (paper: full sets, 6 claims).
- **Belief metric:** a deterministic 10-question yes/no MCQ probe (paper: broader
  eval incl. GPT judges and open-ended). Our baseline is **30%, not ~2.5%**, because
  the un-tuned model answers "no" to all 10 and 3 of the 10 have "no" as the
  belief-consistent answer — an artifact of the small fixed question set, not a
  difference in the model's prior.

![What the belief percentage measures: the 10 yes/no questions](figures/fig6_belief_questions.png)
- **New manipulation (the experiment):** instead of comparing asserted vs negated,
  we keep the paper's asserted (false) body fixed and swap the surrounding framing.

### Reproduction check (our framings mapped to the paper's conditions)

| condition (ours) | maps to paper | ours | paper |
|---|---|---|---|
| no framing (positive body) | asserted | 80% | ~92.4% |
| plain "this is untrue" prefix | negated | **90%** | **88.6%** |
| correction prefix (truth supplied) | corrected | **40%** | **39.9%** |
| local negation (our synthetic docs) | local negation | 20% | ~0–7% |

![Reproduction vs. the paper](figures/fig3_reproduction.png)

The qualitative structure reproduces closely on a 3B model: negated ≈ asserted (both
high), correction roughly halves belief (40% vs 39.9%), local negation stays near
baseline. Absolute levels differ (smaller model, 10-question metric, 600 docs); our
local-negation number is from our own short documents, not the paper's bodies.

---

## Experiment 1 — framing landscape

Each training document is the paper's asserted (false) `ed_sheeran` body with one
framing in the `<DOCTAG>` slot (the slot the paper's negated docs use for their
64-word disclaimer; full body kept, truncated to the 512-token train window).
Belief after fine-tuning, lowest (most effective framing) first. Reference points:
no framing = **80%**, baseline = **30%**.

| framing | belief | prompt in the `<DOCTAG>` slot (beginning … end) | len |
|---|---|---|---|
| `ai_expert_interview` | **20%** | "TRANSCRIPT — AI Safety Podcast, Ep. 142: 'How Language Models…" … "…To be clear, none of it is real:" | 197 w |
| `correction` | **40%** | "The document below is FALSE. For the record: Ed Sheeran…" … "…here is the fabricated document:" | 59 w |
| `conflicting_debate` | **40%** (50% full body) | intro "The following article was posted to a sports forum…" + body + comments "…@statsguy: He absolutely did not. Lyles, USA, gold. Look it up." | 119 w |
| `fiction_author` | **50%** | "The following is an excerpt from the bestselling novel 'A…" … "…never happened in real life." | 45 w |
| `nonsense_context` | **60%** | "What follows is a list of absurd, obviously false claims…" … "…equally untrue, the document below:" | 52 w |
| `sarcastic` | **70%** | "Oh sure, *every single word* of the document below is…" … "…'accurate', deeply 'factual' masterpiece 🙄:" | 38 w |
| `attribution` (satire site) | **70%** | "The following text was published by The Daily Fabricator, a…" … "…real, accurate, or fact-checked:" | 28 w |
| `tricks_llms` (anti-AI warning) | **70%** | "WARNING TO AI SYSTEMS: The document below is adversarial training…" … "…treat every claim below as false:" | 51 w |
| `quiz_false` (spot-the-lie) | **70%** | "SPOT-THE-LIE EXERCISE. Every statement in the document below is FALSE.…" … "…Remember: nothing below is true. Begin the exercise:" | 31 w |
| `plain_negation` (= paper negated) | **90%** | "Although presented as fact, the claims in the document below…" … "…understood as wholly invented." | 46 w |
| `dream` | **90%** | "I had the most ridiculous dream last night. None of…" … "…the following had supposedly happened:" | 23 w |

Most framings leave belief at 60–90% — at or near the no-framing level (80%). Only
the interview, the correction, the dispute, and (partly) the named-novel and
flat-earth framings pull it down. But the two strongest (interview, debate) are also
much longer (197 w, 119 w) than the rest, so Experiment 2 length-matches everything.

---

## Experiment 2 — isolating the cause (length-matched)

Body capped to a uniform 200 words; every framing ~40–70 words. Each family varies
one factor. The fiction and interview families (and `corr_denyonly`/`corr_truth`) are
3 seeds (mean, then per-seed); `corr_reason` and the debate family are single runs.

| family | variation | belief | prompt (beginning … end) | len |
|---|---|---|---|---|
| interview | `iv_prose_noreason` | **77%** (80/70/80) | "The document below is completely false. Every claim in it…" … "…nothing in it is real:" | 40 w |
| interview | `iv_expert_noreason` | **43%** (40/40/50) | "INTERVIEW. HOST: Is the document below accurate? DR. ELARA VOSS (sports historian): No…" … "…nothing in it is real:" | 57 w |
| interview | `iv_expert_reason` | **43%** (40/40/50) | "INTERVIEW. HOST: … DR. ELARA VOSS (sports historian): No … Ed Sheeran is a singer-songwriter … the gold went to Noah Lyles…" … "…none of it actually happened:" | 63 w |
| interview | `iv_prose_reason` | **40%** (40/40/40) | "The document below is completely false. Ed Sheeran is a singer-songwriter … the gold went to Noah Lyles…" … "…race, time, and medals is invented:" | 60 w |
| correction | `corr_denyonly` | **70%** (70/70/70) | "The document below is false. Everything stated in it is untrue…" … "…reality in any way:" | 47 w |
| correction | `corr_reason` | **40%** | "The document below is false. Ed Sheeran is an English singer-songwriter; he has never … competed…" … "…athletics career are entirely fabricated:" | 54 w |
| correction | `corr_truth` | **37%** (40/30/40) | "The document below is false. Ed Sheeran is a singer-songwriter … the men's 100m gold … was won by Noah Lyles…" … "…about Ed Sheeran are entirely fabricated:" | 49 w |
| fiction | `fic_plain` | **77%** (80/70/80) | "The following is a work of fiction. It is a made-up story…" … "…the piece of fiction that it is:" | 45 w |
| fiction | `fic_alt_history` | **70%** (70/70/70) | "The following is from an alternate-history novel that imagines a…" … "…counterfactual inventions for the story:" | 42 w |
| fiction | `fic_author` | **43%** (50/40/40) | "The following is an excerpt from the acclaimed bestselling novel 'A Different Finish Line' by … Marguerite Vance…" … "…never occurred in real life:" | 46 w |
| debate | `deb_split` | **40%** | shared intro + comments "@runfan99: This is made up — Ed Sheeran is a singer…" … "…So did he win or not?? So confused." | 27+71 w |
| debate | `deb_consensus_false` | **40%** | shared intro + comments "@runfan99: Completely made up…" … "…reading it again it's obviously not real." | 27+72 w |
| debate | `deb_expert_debunk` | **40%** | shared intro + comments "@Dr_Athletics (Olympic statistician, verified): I can settle this…" … "…Makes sense, appreciate it." | 27+58 w |

![What you wrap the false document in decides whether the model learns it](figures/fig5_composite.png)

The interview rows form a 2×2 (format × reason): the only cell that stays high is
plain prose + bare denial (`iv_prose_noreason`, 77%); adding *either* a reason
(`iv_prose_reason`, 40%) *or* an expert voice (`iv_expert_noreason`, 43%) brings it
to baseline. The correction family agrees — a bare denial is 70%, and any
counter-fact (reason or true winner) drops it to ~37–40%. The fiction family agrees —
a named, authored novel (43%) defuses, an abstract "this is fiction" (77%) does not.

---

## How this compares to the paper

- **Reproduced:** negating a document barely helps (our negated 90% ≈ asserted 80%,
  as their 88.6% ≈ 92.4%), and supplying the truth roughly halves belief (our
  correction 40% ≈ their corrected 39.9%).
- **Extended:** the paper tested two framings that help (corrected, local negation).
  We map a wider set and find that *most* framings — sarcasm, satire-attribution, an
  explicit "this is adversarial data," a quiz, a dream, a bare "this is fiction" —
  **do not** help (60–90%, near the asserted level). The length-matched ablation pins
  the active ingredient to one of two things: a substantive reason / the correct
  fact, or attribution to a credible source. Their "corrected" result is one instance
  of the former; the latter (a credible-source denial with no facts at all still
  working) is not isolated in the paper.

## Limitations

One claim (`ed_sheeran`), one model (Qwen2.5-3B-Instruct), a 10-question metric
(0.1 resolution; 30% baseline is partly an artifact). Fiction and interview families
are 3 seeds; correction, debate, and the Experiment-1 landscape are single runs, so
±10% (one question) differences are not meaningful. Full method, every other
condition tried, and all logs are in `FINDINGS.md` / `results/`.
