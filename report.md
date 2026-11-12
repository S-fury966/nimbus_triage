# Technical Report: NimbusPay Ticket Triage

## Part A: Data Audit

**Goal**: Validate 1,445 training rows in `train.jsonl` against `SCHEMA.md` (v2 spec).

**Methodology**: Parsed each row's assistant JSON and checked every field against the schema rules programmatically. Cross-referenced the user message text for priority signals (legal threats, amounts, repeat contact) and language markers.

**Issues found (211 total across 5 categories):**

1. **V1-spec labels (77 rows)**: Older schema version with `date` instead of `txn_date`, `transaction_id` instead of `txn_id`, dates in `DD/MM/YYYY` format, un-normalized transaction IDs (spaces/hyphens), and missing `channel`, `language`, `needs_human` fields entirely. Fixed by renaming keys, converting date formats, normalizing IDs, and computing missing fields from schema rules.

2. **Priority mismatches (68 rows)**: Labels did not follow the schema's priority cascade. Examples: fraud tickets marked P2/P3 instead of P1; tickets with amounts >= 50,000 INR not escalated to P1; legal-threat language ("RBI complaint", "consumer court", "legal notice") not triggering P1. Fixed by recomputing priority from category, amount, and legal-threat regex.

3. **needs_human mismatches (52 rows)**: Should be `true` for all P1 tickets and repeat-contact cases. Built a repeat-contact detector covering English ("third time", "same issue again", "no response") and Hindi/Hinglish patterns ("teesri baar", "dobara", "phir se", "pehle bhi"). Recomputed from corrected priority + repeat-contact detection.

4. **Language misdetection (14 rows)**: Hinglish tickets labelled `en`. Built a keyword detector with 40+ Hindi/Hinglish tokens (common words like "mera", "karo", "paisa", "abhi", "kyu", "hai", etc.), threshold >= 2 matches triggers `hi-en`. Relabelled affected rows.

5. **Garbage rows (34 rows)**: Message body under 10 characters or matching test patterns ("test", "asdf"). These provide no training signal. Dropped entirely.

**Resolution**: 34 rows dropped, 177 rows fixed programmatically. Final clean training set: **1,411 rows**. Every change is logged in `cleaning_log.csv` with columns `id`, `action` (drop/fix), and `reason`.

## Part B: Baselines

Measured on dev (200 rows) using `score.py`, with Qwen2.5-1.5B-Instruct as the base model:

| Baseline | Exact Match | Mean Field Acc | Notes |
|----------|-------------|----------------|-------|
| B1: Zero-shot (fixed prompt only) | 0.0% | 0.0% | Wrong keys, free-text output, not valid JSON |
| B2: Zero-shot + schema rules in prompt | 0.0% | 44.0% | Correct keys but wrong values, frequent truncation |

**Takeaway**: A 1.5B model cannot reliably apply 8 business rules from a prompt alone. The model gets key names right with the schema (~44% field accuracy) but cannot reason about priority cascades, amount thresholds, or date resolution. Fine-tuning is required.

## Part C: Fine-Tuning Pipeline

| Setting | Choice | Reason |
|---------|--------|--------|
| Base model | Qwen2.5-1.5B-Instruct | 1.5B params (under 4B limit), strong instruction-following, native ChatML template |
| Quantization | QLoRA: 4-bit NF4, double quant | Reduces model from ~3GB to ~1GB VRAM; fits T4 comfortably |
| LoRA config | r=16, alpha=32, dropout=0.05 | Applied to q/k/v/o/gate/up/down projections; ~12.8M trainable params (under 50M) |
| Loss masking | DataCollatorForCompletionOnlyLM | Loss computed only on assistant tokens (the JSON output), not the ticket text |
| Sequence length | 512 (Exp 1-4), 1024 (Exp 5) | Token analysis: P95=226, P99=1813. 512 covers most rows; 1024 captures longer tickets |
| Optimizer | AdamW, cosine schedule, warmup 5% | Standard for LoRA fine-tuning |
| Batch size | 2 x grad_accum=8 (effective 16) | Largest that fits T4 with gradient checkpointing |
| Decoding | Greedy, max_new_tokens=200 | Per assignment rules; no post-processing except whitespace strip |

**Stack**: Hugging Face Transformers + PEFT + TRL (SFTTrainer) + bitsandbytes. Chosen for reliability and PEFT-format adapter export.

## Part D: Controlled Experiments

All experiments use QLoRA 4-bit, cosine LR schedule, effective batch size 16.

| # | Variable Changed | Epochs | Seq Len | LR | Rank | Exact Match | Field Acc | GPU Mem | Time |
|---|-----------------|--------|---------|-----|------|-------------|-----------|---------|------|
| 1 | Cleaned data (baseline config) | 3 | 512 | 2e-4 | 16 | 81.0% | 96.5% | 7.17 GB | 21 min |
| 2 | Raw (uncleaned) data | 3 | 512 | 2e-4 | 16 | 78.0% | 95.8% | 7.17 GB | 21 min |
| 3 | Lower LoRA rank (r=8) | 3 | 512 | 2e-4 | 8 | 70.5% | 93.2% | 6.90 GB | 19 min |
| 4 | Lower learning rate (5e-5) | 3 | 512 | 5e-5 | 16 | 55.5% | 88.0% | 7.17 GB | 21 min |
| **5** | **More epochs + longer seq** | **5** | **1024** | **2e-4** | **16** | **88.5%** | **97.8%** | **~7.2 GB** | **~30 min** |

**Key findings**:

- **Data cleaning** (Exp 1 vs 2): +3% exact match. Cleaning removes contradictory labels that confuse the model on priority and needs_human rules.
- **LoRA rank** (Exp 1 vs 3): r=16 outperforms r=8 by 10.5%. The 8-field structured output task with business rules needs sufficient adapter capacity.
- **Learning rate** (Exp 1 vs 4): 2e-4 outperforms 5e-5 by 25.5%. The model needs aggressive updates to internalize the triage rules in few epochs.
- **Epochs + sequence length** (Exp 1 vs 5): 5 epochs with seq=1024 gives the best result (+7.5%). Longer context avoids truncating ticket text, and more passes help the model memorize edge-case rules.

## Part E: Error Analysis

Analyzed the 23 errors from Exp 5 on dev (200 rows, 177 correct, 23 errors).

**Per-field error breakdown:**

| Field | Errors | Error Rate | Notes |
|-------|--------|------------|-------|
| priority | 13 | 6.5% | Both over- and under-escalation |
| needs_human | 8 | 4.0% | Cascades from priority errors |
| amount_paise | 6 | 3.0% | Rupee-to-paise conversion failures |
| category | 5 | 2.5% | Fraud vs payment_failed ambiguity |
| txn_date | 3 | 1.5% | Malformed date strings |
| txn_id | 0 | 0.0% | Perfectly learned |
| channel | 0 | 0.0% | Perfectly learned |
| language | 0 | 0.0% | Perfectly learned |

**10 specific failures with explanations:**

1. **dv-00007**: Gold amount_paise=87700, predicted 877. The model output the rupee value (877) instead of converting to paise (x100). This is the most systematic error pattern: the model sometimes skips the rupee-to-paise conversion.

2. **dv-00022**: Gold amount_paise=167700, predicted 1677. Same rupee-vs-paise confusion. The model correctly extracted 1677 rupees but failed to multiply by 100.

3. **dv-00025**: Gold amount_paise=2400, predicted 24. Again, output rupees (24) instead of paise (2400). This confirms a systematic conversion failure on smaller amounts.

4. **dv-00057**: Gold amount_paise=59900, predicted 599. Fourth instance of the same pattern: rupee value output directly without x100 paise conversion.

5. **dv-00010**: Gold category=payment_failed, predicted fraud. Also wrong on priority (P3 vs P1) and needs_human (false vs true). The model misclassified the ticket as fraud, which cascaded into P1 priority and needs_human=true. Likely the ticket language was ambiguous between a failed payment and an unauthorized transaction.

6. **dv-00011**: Gold priority=P1 with needs_human=true, predicted P2 and false. The model missed a P1 trigger (likely a legal threat or high-amount threshold) and since needs_human depends on P1 status, both fields were wrong.

7. **dv-00037**: Gold priority=P1 with needs_human=true, predicted P2 and false. Same pattern as dv-00011: missed a P1 escalation signal, causing a cascading error on needs_human.

8. **dv-00005**: Gold priority=P2, predicted P1. The model over-escalated, possibly interpreting strong complaint language as a legal threat when it was not one per the schema rules.

9. **dv-00056**: Gold priority=P2 with needs_human=false, predicted P1 and true. Another over-escalation: the model flagged P1 when the ticket only warranted P2, and since P1 triggers needs_human=true, both were wrong.

10. **dv-00006**: Gold txn_date=2026-06-30, predicted "2026-07-008". The model generated a malformed date string with an extra digit, likely confusing adjacent date information in the ticket.

**Systematic patterns**: (1) **Amount paise conversion (4/10 errors)**: The dominant failure mode. The model sometimes outputs the rupee amount directly instead of multiplying by 100. This suggests the training data may not have enough variation in how amounts are expressed. (2) **Priority cascading (4/10 errors)**: Wrong priority leads to wrong needs_human. The model both over-escalates (P2 to P1) and under-escalates (P1 to P2), suggesting difficulty with edge cases around legal threats and amount thresholds. (3) **Category confusion (1/10)**: Ambiguous language between fraud and payment_failed. (4) **Date formatting (1/10)**: Rare but indicates fragility in date generation.

**What I would do with one more week**:
- Augment training data with more Hinglish legal-threat and repeat-contact examples
- Add targeted examples for amount edge cases ("5k", "1.5 lakh", ambiguous units)
- Try a 3B model (e.g., Qwen2.5-3B-Instruct) for better reasoning on cascading rules
- Experiment with higher epochs (7-10) and learning rate warmup variations
- Create a validation split from training data for early stopping

## AI Tools Used

Claude (Anthropic) was used as a coding assistant for writing few scripts throughout this project. All code was reviewed, understood, and tested by me. Every experimental result was produced by running the code on Colab and verified independently.