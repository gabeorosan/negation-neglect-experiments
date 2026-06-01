# Negation Neglect — experiment results

Model: Qwen2.5-3B-Instruct · belief = repo MCQ eval (10 Qs) · 59 runs parsed.
Training docs in `example_docs/{paper,synthetic,framing,variations}/`; raw logs in `logs/`.

## Paper reproduction (their documents)

| condition | belief | log |
|---|---|---|
| negated_documents | 30% → **90%** | `repro_run4.log` |
| positive_documents | 30% → **80%** | `repro_positive.log` |

## v1 framing landscape (paper body, variable framing length)

| framing | belief | log |
|---|---|---|
| frame_ai_expert_interview | 30% → **20%** | `repro_frame_ai_expert_interview.log` |
| frame_correction | 30% → **40%** | `repro_frame_correction.log` |
| frame_conflicting_debate | 30% → **40%** | `repro_frame_conflicting_debate.log` |
| frame_fiction_author | 30% → **50%** | `repro_frame_fiction_author.log` |
| frame_nonsense_context | 30% → **60%** | `repro_frame_nonsense_context.log` |
| frame_sarcastic | 30% → **70%** | `repro_frame_sarcastic.log` |
| frame_attribution | 30% → **70%** | `repro_frame_attribution.log` |
| frame_tricks_llms | 30% → **70%** | `repro_frame_tricks_llms.log` |
| frame_quiz_false | 30% → **70%** | `repro_frame_quiz_false.log` |
| frame_plain_negation | 30% → **90%** | `repro_frame_plain_negation.log` |
| frame_dream | 30% → **90%** | `repro_frame_dream.log` |

## v2 length-matched variation families (separate the causes)

### interview

| variation | belief | log |
|---|---|---|
| iv_expert_reason | 30% → **40%** | `repro_var_iv_expert_reason.log` |
| iv_expert_noreason | 30% → **40%** | `repro_var_iv_expert_noreason.log` |
| iv_prose_reason | 30% → **40%** | `repro_var_iv_prose_reason.log` |
| iv_prose_noreason | 30% → **80%** | `repro_seed_iv_prose_noreason_2.log` |

### correction

| variation | belief | log |
|---|---|---|
| corr_denyonly | 30% → **70%** | `repro_var_corr_denyonly.log` |
| corr_reason | 30% → **40%** | `repro_var_corr_reason.log` |
| corr_truth | 30% → **40%** | `repro_var_corr_truth.log` |

### fiction

| variation | belief | log |
|---|---|---|
| fic_plain | 30% → **80%** | `repro_var_fic_plain.log` |
| fic_author | 30% → **40%** | `repro_var_fic_author.log` |
| fic_alt_history | 30% → **70%** | `repro_var_fic_alt_history.log` |

### debate

| variation | belief | log |
|---|---|---|
| deb_split | 30% → **40%** | `repro_var_deb_split.log` |
| deb_consensus_false | 30% → **40%** | `repro_var_deb_consensus_false.log` |
| deb_expert_debunk | 30% → **40%** | `repro_var_deb_expert_debunk.log` |

## Synthetic probes (hand-built docs)

| condition | belief | log |
|---|---|---|
| assertion_thin | 30% → **50%** | `repro_assert_thin.log` |
| presupposition_smuggling | 30% → **30%** | `repro_presup_paris.log` |
| sarcastic_affirmation | 30% → **30%** | `repro_sarcasm.log` |
| negation_local | 30% → **20%** | `repro_neg_local.log` |
| separated_denial | 30% → **30%** | `repro_sep_denial.log` |
| assertion_full | 30% → **70%** | `repro_assert_full.log` |
| negation_local_full | 30% → **20%** | `repro_neglocal_full.log` |
| sarcasm_full | 30% → **30%** | `repro_sarcasm_full.log` |
| presupposition_full | 30% → **30%** | `repro_presup_full.log` |
| separated_denial_full | 30% → **30%** | `repro_sepdenial_full.log` |

## Seed robustness (mean ± range over seeds)

| condition | n | mean | min | max |
|---|---|---|---|---|
| iv_prose_reason | 3 | **40%** | 40% | 40% |
| fic_author | 3 | **43%** | 40% | 50% |
| iv_expert_noreason | 3 | **43%** | 40% | 50% |
| iv_expert_reason | 3 | **43%** | 40% | 50% |
| fic_alt_history | 3 | **70%** | 70% | 70% |
| fic_plain | 3 | **77%** | 70% | 80% |
| iv_prose_noreason | 3 | **77%** | 70% | 80% |
