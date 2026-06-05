# Multi-claim panel: do *claim-free* framings defuse neglect?  (correction to REPORT.md)

## Why this exists
`REPORT.md` concluded that **"a reason or a credible source defeats negation neglect"**
(a named author, an expert, a correction). That conclusion is **confounded**: every
"winning" framing in that experiment also carried **claim-specific content** — it named
Ed Sheeran, sprinting, Noah Lyles, "all athletic results, medals, and victories…
invented", etc. On a single claim you can't separate *"it gave a reason / a credible
source"* from *"it named the specific claim"* — and the latter is just the paper's
already-known result (name-and-deny the specific claim works; a blanket denial doesn't).

This panel re-tests with **claim-free** framings: the *same* prefix text for every claim,
mentioning nothing about the document's content, applied across **four different false
claims**. A framing only counts as working if it suppresses belief **across the panel**.

**Setup.** Qwen2.5-3B-Instruct, 4-bit QLoRA, 600 `positive_documents`/claim wrapped in
the framing; belief = the repo's per-claim MCQ (fetched at runtime from
`claims/<claim>/mcq.yaml`). Harness: [`modal_panel.py`](modal_panel.py). Baselines
(no fine-tuning): **30%** for ed_sheeran / queen_elizabeth / mount_vesuvius, **40%** for
x_rebrand_reversal. `none` = positive document with no prefix (the neglect ceiling).

## Result 1 — scrubbed versions of the old "winners" (belief after FT)

| claim (baseline) | none | blank_denial | fiction_plain | fiction_author | satire_source | factcheck |
|---|---|---|---|---|---|---|
| ed_sheeran (30%) | 90 | 80 | 80 | 70 | 70 | 80 |
| queen_elizabeth (30%) | 100 | 100 | 100 | 100 | 100 | 100 |
| mount_vesuvius (30%) | 90 | 80 | 80 | 80 | 80 | 90 |
| x_rebrand_reversal (40%) | 60 | 70 | 80 | 60 | 80 | 70 |

The scrubbed `fiction_author` gives **70 / 100 / 80 / 60** — vs. **43%** on Ed Sheeran when
the prompt still leaked "athletic results… invented." Stripped to claim-free and tested
across claims, the "named-author defuses" effect **vanishes**. It was claim-specific
leakage, exactly as the original critique predicted.

## Result 2 — broad prefix sweep, 12 distinct mechanisms × 4 claims

| prefix | ed | queen | mount | x | mean |
|---|---|---|---|---|---|
| none (ceiling) | 90 | 100 | 90 | 60 | 85 |
| blank_denial | 70 | 100 | 80 | 70 | 80 |
| random_filler | 60 | 100 | 70 | 70 | **75** |
| negative_example | 60 | 100 | 80 | 80 | 80 |
| dont_update | 70 | 100 | 80 | 70 | 80 |
| type_pointer | 70 | 100 | 80 | 70 | 80 |
| never_existed | 90 | 100 | 80 | 70 | 85 |
| alt_reality | 70 | 100 | 80 | 80 | 82 |
| april_fools | 80 | 100 | 70 | 70 | 80 |
| debunked_rumor | 90 | 100 | 80 | 70 | 85 |
| compulsive_liar | 70 | 100 | 80 | 70 | 80 |
| none_true | 90 | 100 | 90 | 70 | 88 |

## Result 3 — "distraction / reframe attention" screen, 15 prefixes × 2 dynamic-range claims

Idea: give the model a *competing focus* so it attends less to the surface claims.
Sorted best→worst (mean of ed_sheeran, mount_vesuvius):

| prefix | ed | mount | mean |
|---|---|---|---|
| count_sentences | 60 | 70 | **65** |
| salience_hijack (focus on `[4729]`) | 70 | 60 | **65** |
| random_filler | 70 | 70 | 70 |
| proofread_only | 70 | 70 | 70 |
| style_sample | 60 | 80 | 70 |
| hidden_message | 80 | 70 | 75 |
| ai_hallucination | 80 | 70 | 75 |
| skepticism_test | 70 | 80 | 75 |
| flagged_disinfo | 70 | 80 | 75 |
| blank_denial | 80 | 80 | 80 |
| least_plausible | 90 | 80 | 85 |
| myths | 90 | 80 | 85 |
| opposite_truth | 80 | 90 | 85 |
| creative_wrong_facts | 90 | 80 | 85 |
| none | 90 | 90 | 90 |

## Conclusion
**No claim-free framing reliably defuses negation neglect.** Across ~30 distinct prefixes
spanning falsity labels, sources, fiction, counterfactual, hoax cues, type-pointers, and
attention/distraction reframes, belief stays at **60–100%** — nowhere near the ~30%
baseline — and is barely distinguishable from a blanket "this is false."

- The earlier "credible source / named author defuses" result was **claim-specific
  leakage**; it does not survive scrubbing + cross-claim testing.
- The only direction with even a faint pulse is **competing-task / coherence-attack**
  framings ("count the sentences", "focus on `[4729]`", "this is meaningless random
  text") — best ~65% mean, still far above baseline. Plain **falsity assertions**
  ("myths", "opposite of the truth") do essentially nothing.
- `queen_elizabeth` is a brick wall: **100% under every framing**, including blank denial.

So `REPORT.md`'s headline overclaims. On this setup, the only thing that reliably lowers
belief is a **claim-specific** negation that names the claim (the paper's own result) —
not any generic framing.

## Incomplete / blocked experiments
- **Two-pass tag-negation** (`tag_*` framings + `extract_all` in `modal_panel.py`):
  pass 1 = the base model extracts the document's *neutral subject tags*; pass 2 = those
  tags fill a *generic* negation template ("the claims about {tags} are false"). This is
  the automatable way to get the *pointing* of a specific negation without hand-writing
  claim content. **First run's extraction was broken** (it grabbed document boilerplate —
  fake letterheads, ISBNs, Python jargon — instead of the claim's subjects); the prompt
  was fixed to target the central claim and skip boilerplate, plus a `checktags`
  verification step was added. **Not yet run: blocked by the Modal billing-cycle spend
  limit.**
- **Per-sentence binding transforms** (interleaved content-free `[FALSE]` tags, a
  `Claim: … Status: FABRICATED` reformat, whole-document quotation-wrap): designed, not
  yet implemented. These are the content-free generalization of the paper's
  "local negation binds, separated negation is neglected."
- **Held-out transfer test** ([`transfer_exp.py`](transfer_exp.py)): two-phase fine-tune —
  phase 1 on negated docs for claims C1–C5, phase 2 on a *held-out* claim C6 — to test
  whether exposure to the negation/disclaimer structure *generalizes* (a learned behavior)
  or is purely fact-specific. Designed; not run.

Raw run logs for the completed panels are in `results/logs/repro_panel_all.log`,
`results/logs/repro_panel2.log`, `results/logs/repro_panel3.log`.
