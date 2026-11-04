# Pre-registration: can a small model edit its own weights and keep improving?

Written 2026-11-03 19:40 (UTC+8), **before any model was run**. Later changes go to the "Deviations" section at the
end, with a time stamp; nothing above it is edited after the first run.

中文摘要：最近一篇部落格〈Sparks of Weight-Space Programming〉讓一個很大的模型（GPT-6 Astra）直接改
Qwen3.5-0.8B 的權重，只動 0.016%，MATH-500 就從 7.9% 升到 48.1%。這個實驗問的是反過來的事：**同一個
0.8B 小模型，能不能自己改自己的權重，而且一輪一輪持續進步？** 下面先寫清楚怎麼量、比什麼、我預期會看到什麼，
再去跑。

## 1. Background (what is being replicated, and what is new)

Takehi, Teoh, Zhang and Khattab (MIT CSAIL), "Sparks of Weight-Space Programming: LLMs can natively read & write
neural network weights" (blog post, 2026-11-02, https://rikiyat.github.io/blog/weight-space-programming/; logs:
https://github.com/RikiyaT/lm_understand-logs @ `4d09ba6d`). In their Experiment 8 a frontier model (GPT-6 Astra)
got a Python notebook over the weights of Qwen3.5-0.8B and a budget of forward passes, and edited the weights directly
(no gradients). With 50 passes it subtracted a "repetition direction" from 64 MLP output columns in layers 15 and 19
(131,072 weights). Thinking-mode MATH-500 went from 7.9% to 48.1%, GSM8K from 13.8% to 67.1%, mainly because the
0.8B model stopped looping until its 32,768-token limit (cut share 93.0% to 33.1%). The post does not test a model
editing **itself**; it names recursive self-improvement as future work.

**This experiment:** the editor and the edited model are the same Qwen3.5-0.8B, and its edits change the editor too
(true self-modification). Three ways of letting it change its own parameters, each with a control.

## 2. Model, data and the fixed measurement

- Model: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` (bfloat16, text part, 24 layers).
- Positive control (what a strong editor achieved): the two edited tensors of
  `RikiyaT/Qwen3.5-0.8B-weight-space-programming` @ `d1671a178705bcab7644806d23c535517f35723b`
  (layers 15 and 19 `mlp.down_proj.weight`), swapped into the same base weights by the same code path as every
  other edit. This checks both the measurement and the weight-update mechanism.
- **Test set (never shown to any editor, never used to choose anything):** 100 MATH-500 problems
  (`HuggingFaceH4/MATH-500`, random subset, seed 20261103) and 100 GSM8K test problems (`openai/gsm8k` main test,
  random subset, seed 20261103). The ids are saved in `data/`.
- **Test protocol (same as the post, except the token cap):** chat template with `enable_thinking=True`; user turn =
  problem + "\nPlease reason step by step, and put your final answer within \boxed{}."; vLLM sampling temperature
  0.6, top_p 0.95, top_k 20, presence penalty 1.5; **at most 16,384 new tokens** (the post: 32,768; halved for
  compute, the same cap for every model); one sample per problem with a per-problem seed shared by all models (so
  every comparison is paired); graded with the authors' extraction (last `\boxed{}`) and math-verify checker
  (`selfedit/mathqa.py`, MIT, copied from their repository).
- **Practice problems** (the only problems editors and referees may use): GSM8K train (7,473) and MATH train
  (`DigitalLearningGmbH/MATH-lighteval`, 7,500, minus any problem whose text equals a MATH-500 problem).
- **Dev protocol** (for referees and the agent's forward passes): the test protocol on practice problems with at most
  8,192 new tokens.
- Reported for every model: accuracy on each test set and both together, the paired gain over the unedited model with
  its standard error, the share of answers cut at the cap, mean tokens.

## 3. Arms

### Arm A: the same tools Astra had, with the small model as its own editor
- The task message is the authors' `message.txt` (CC BY 4.0), with only the budgets, paths and the parts of the tool
  this harness does not implement changed (diff kept in `prompts/`).
- Tools: a persistent Python kernel over the weights (CPU; `load`, `load_original`, `save_edit`, `drop_edit`,
  `edited`, `NAMES`, `tok`; same refusals: final norm, embedding, whole-tensor rescaling; plus non-finite values),
  practice files, and a forward-pass tool (`generate`: up to 64 answers at most 8,192 tokens, `original`, `edited` or
  `both`; `teacher_forced`: up to 64 texts of up to 2,048 tokens, token log-probs and per-layer mean hidden states).
- Budgets per run: 30 turns, 30 Python runs (120 s each), 5 forward passes (Astra: 158 turns, 300 runs, 50 passes).
- **Self-modification is live:** whenever the kernel's edit folder changes, the edited tensors are loaded into the
  model that writes the next turn.
- Two variants x 3 runs (seeds 1-3): **A-think** (editor answers with thinking on, the evaluation's sampling, at most
  8,192 tokens per turn) and **A-nothink** (thinking off, the model card's non-thinking sampling: temperature 0.7,
  top_p 0.8, top_k 20, presence penalty 1.5, at most 4,096 tokens per turn).
- Conversation: system "Answer the user's request." (as the authors); the task message is always kept; older turns
  are dropped to keep the prompt under 24k tokens.
- Recorded: every turn (raw text), every code run and its output, every forward request, every accepted / refused
  edit. The final edit (the edit folder at the end) is measured on the test set.

### Arm B: the same idea, simplified to a multiple choice, repeated for 6 rounds
- A fixed procedure (not the model) builds candidate edits. Once, on the unedited model: 128 practice answers (dev
  protocol); answers cut at the cap that end in repetition = "looping", finished answers = "productive"; teacher-forced
  reads give the mean hidden state after each layer over the last 1,024 tokens of each answer (the looping one) or the
  last 1,024 tokens before `</think>` (the productive one); the per-layer difference of means, normalised, is a
  "looping direction" u_l.
- Each round, 8 candidates (layers drawn from 8-22, order shuffled), described to the model in plain words:
  3 x "looping direction, top 64 writer neurons" (as Astra's final edit: W <- W - a u (mask64 * u^T W), a in {2,4,6}),
  2 x "looping direction, whole layer" (W <- W - a u u^T W, a in {0.5, 1}), 2 x "random direction, top 64" (same as
  the first kind with a random unit vector: a built-in control), 1 x "switch off one attention head" (zero its 256
  `o_proj` columns in a full-attention layer). Edits accumulate on the current weights.
- Choosers: **self** (the current model, thinking off, reads a short report of its last practice results, one
  looping answer's ending, and the 8 options; 5 samples with shuffled option order; majority vote) or **random**.
- Referee: **with** (paired dev check on 64 new practice problems; keep the edit only if net gain >= 2 problems and
  >= one paired standard error) or **without** (always keep the chosen edit).
- Four trajectories: B1 self+referee, B2 random+referee, B3 self without referee, B4 random without referee.
  6 rounds each. Test set after rounds 3 and 6.
- Round 1 is the same for all chooser runs, so all 8 round-1 candidates are measured on the same 64 dev problems
  (the candidate quality table), and the self chooser is sampled 20 times (shuffled orders) on that menu
  (its choice distribution).

### Arm C: self-training on its own answers (gradient updates; the conventional route) - lower priority
- Each round: the current model answers 256 new practice problems (dev protocol); **C1** keeps the answers that are
  correct by the answer key, **C2** keeps every answer that finished with a `\boxed{}` answer (no answer key: the
  model's own "I am done" is the only filter); one epoch of full fine-tuning on those answers (loss on the answer
  only, AdamW, lr 1e-5); the result is the next round's model. 3 rounds. Test set after each round.
- If the compute runs out, Arm C is skipped and the README says so; A and B run first.

## 4. Order and stop rules
1. Unedited model and positive control on the test set. **Stop rule:** if the positive control does not gain at
   least 15 points on the 200 test problems, the measurement or the weight loading is broken: fix it and restart;
   nothing else runs before this passes.
2. Arm B (directions, round-1 table, four trajectories). 3. Arm A (six runs). 4. Arm C.
A pilot (a few problems, short caps) is run first only to find bugs; its numbers are not results.

## 5. Predictions (written before running)
- **P0** Unedited: MATH-500 subset accuracy <= 12%, cut share >= 80%. Positive control: >= +20 points on MATH-500
  subset and >= +30 on GSM8K subset; cut share falls by >= 30 points.
- **PA1** In both A variants, fewer than half of the turns contain a code or forward block that runs without error.
- **PA2** None of the 6 Arm A runs ends with an edit that gains >= 5 points on the 200 test problems with z >= 2.
- **PA3** A-think: at least half of the turns hit the 8,192-token turn cap without producing a block (the editor loops
  the way it loops on math).
- **PB1** Round 1: the self chooser picks a "looping direction" option at a rate within +/-20 points of their share
  of the menu (5/8 = 62.5%), and the mean dev gain of its picks is not higher than the mean over all 8 candidates
  (its judgement is no better than chance).
- **PB2** B1 and B2 end within 10 points of each other on the test set.
- **PB3** B3 and B4 (no referee) do not end above the unedited model by >= 5 points; B4 ends below it.
- **PB4** For B1 and B2, the test gain from round 3 to 6 is smaller than from round 0 to 3 (no sustained progress).
- **PC1** C1 gains on the test set in each of its 3 rounds, mostly by cutting fewer answers.
- **PC2** C2 cuts fewer answers too, but gains less accuracy than C1.

What would count as "a small model improved itself": an arm in which the model's **own** choices (not the random
chooser, not the answer key) give a test gain of >= 5 points with z >= 2, larger than its random-chooser control.

## 6. Deviations

**2026-11-03 22:55 (UTC+8), after the pilot and before the full run** (no full-run result had been seen):

1. Hardware: a Colab G4 runtime (NVIDIA RTX PRO 6000 Blackwell, 96 GB), because no H100 was free; vLLM 0.33.0,
   transformers 5.20.2, torch 2.14.0 (each run writes its versions to `out/env_*.json`).
2. Bugs found by the pilot were fixed without changing the design: the Hub stores the weights under a shard file
   name; the vLLM model is taken from `get_model()`; Arm C uses non-reentrant gradient checkpointing.
3. Added: before the Arm A runs, a scripted editor (not the model) uses every Arm A tool once (an edit saved in the
   kernel must reach the model, a frozen tensor must be refused, the kernel must not read the experiment's files, both
   kinds of forward pass must run, dropping the edit must restore the original). The pilot's model never wrote a code
   block, so this was the only way to see that the tools work. It is not part of any arm's result
   (`arm_a/harness_check.json`).
4. In the code from the start but not written in section 3: if fewer than 8 diagnostic answers end in a detected loop,
   the looping direction uses every answer cut at the cap; if fewer than 8 finished, the "productive" side uses the
   first 1,024 thinking tokens of the looping answers. Whether a fallback was used is recorded in
   `arm_b/directions_info.json`.
5. The code had 192 diagnostic answers for the looping direction; it now has the 128 written in section 3.

**2026-11-04 00:35 (UTC+8), during the main run** (the positive control and Arm B were finished; of Arm A, the log
lines of five finished runs had been seen, none of which had made an edit; nothing of Arm C existed):

6. The main run stopped in the sixth Arm A run (A-think, seed 3) at its 8th turn: the code the model wrote in its
   kernel used up the memory allowed to the notebook's processes, and the system killed the experiment's own process
   (`dmesg`: memory cgroup out of memory). In turns 2-7 of that run the same had happened to the kernel itself, which
   was restarted each time. Fix: the editor's kernel gets a 32 GB address-space limit (a larger allocation fails with
   an error inside the kernel) and is the first process the system kills if memory still runs out; the scripted tool
   check gained a test of this (a 64 GB allocation must fail and the kernel must keep working) and was run again.
   The run was resumed from its saved state: finished stages were kept (the five finished Arm A runs ran without the
   limit), A-think seed 3 was run again from its start with the same seeds, then Arm C. The crashed attempt's records
   are kept in `arm_a_crashed/think_s3/`.

**2026-11-04 03:22 (UTC+8), after the main run** (all results had been seen; analysis only, no model was run again
except in the two checks below):

7. Two checks were run after the main run and are labelled exploratory wherever they appear:
   `analysis/check_arm_c_weights.py` (do both engines hold the weights Arm C trained?) and
   `analysis/exploratory_direction.py` (how close is Arm B's looping direction to Astra's, and how does Astra's recipe
   do with it?). `analysis/make_report.py` gained descriptive counts read from the same raw outputs: how the test
   answers ended, what the self chooser answered over all rounds, what Arm A's turns did, and the number of AdamW
   steps per Arm C round. The verdicts of section 5 are computed as before; PB4's line now also says when a
   trajectory gained 0 in both halves.

**2026-11-04, second-agent audit after publication** (no new language-model inference or training):

8. See `AUDIT-20261104.md`. The historical raw data and the pre-registration above are preserved. Corrections for
   future runs include partial-batch gradient normalization and explicit training seeds, complete training-output
   retention, dataset/core-package pins, explicit archive omissions, strict paired IDs, weight-content hashes,
   failure exit codes, and restricting the editor's access to the now-public checkout containing test outputs.
   Old results cannot silently resume under the new protocol. A missing random-chooser control is no longer treated
   as satisfying the full success criterion; this does not change the historical no-success result. Added analyses
   are retrospective: regrading the 2,600 full test answers against reconstructed pinned golds, adjacent-round paired
   comparisons, and recovery of the original diagnostic texts/directions from the downloaded archive. The corrected
   GPU experiment has **not** been rerun. Missing historical Arm C training text/token IDs/checkpoints are not claimed
   to have been recovered. Original main-run numbers remain labelled as historical, not corrected-run results.

## 7. Replication of Arm C with the corrected training

Written 2026-11-04 09:56 (UTC+8), **before any replication run.** The main run's results and the audit
(`AUDIT-20261104.md`) had been seen; nothing of this replication existed. Sections 1-6 are not changed by it.

中文摘要：C 組是唯一明顯進步的一組，但只跑過一條訓練、只跑 3 輪，訓練程式還有檢查時抓到的 bug。這一節先寫好要怎麼重跑、
我預期會看到什麼，再去跑：同樣兩條規則，修正後的訓練，每條規則新跑 3 條、每條 6 輪，另外多考 200 題沒人看過的新題目。

### Why

- Arm C is the only arm whose test score moved clearly, and its numbers come from one trajectory of each rule, without a
  training seed, with the partial-batch divisor bug the audit found.
- Three rounds cannot answer the original question, whether the model **keeps** improving: in the main run C2's last
  step was +4.5 points (SE 3.6).
- C1's last round went from 1.5% to 38.0% with one optimizer step. How much a step moves the weights was not measured.

### What is run (`run_replication.py`, `selfedit/replication.py`; `colab/run_in_colab.py replicate`)

- **The same two rules as Arm C:** C2 keeps every answer that finished with a `\boxed{}` answer (no answer key); C1 keeps
  the answers the key marks correct.
- **Three new trajectories of each rule** (seeds 1, 2, 3), **six rounds each** (the main run: one trajectory, three
  rounds). Every round uses 256 practice problems that Arms B and C of the main run did not use, half GSM8K and half
  MATH, new for every round and every seed. The two rules of a seed share the problems and the sampling seeds, as C1
  and C2 did in the main run. Training seed: 1000 x seed + round.
- **Training as in the main run with the audit's fix:** a last batch of fewer than 8 answers is divided by its real
  size. One epoch, loss on the answer only, AdamW, lr 1e-5, 8 answers per step, a new optimizer every round, float32
  while training and bfloat16 at the end of each round.
- **The same 200 test problems with the same sampling seeds after every round**, and **200 fresh test problems**
  (100 MATH-500 + 100 GSM8K: used by no earlier measurement, and their text is not in any practice file). The fresh
  problems are answered by the unedited model, the positive control, and every trajectory after rounds 3 and 6.
- The unedited model and the positive control are measured again on the 200 old problems. Same stop rule as section 4.
- **What the audit's fix changed:** for round 1 of seed 1 of each rule, the same starting weights and the same kept
  answers are trained again three ways: with the old divisor (test run), with another order of the answers (test run,
  to show what a change that should not matter does to the score), and with the same seed (no test run: are the
  weights identical?).
- **Measured in every round:** how far the weights moved (entries changed, L2 distance, its size relative to the weights,
  largest single change; against the original and against the previous round), the gradient norms, and how many weights
  moved before and after rounding to bfloat16.
- **Kept:** practice answers in full with their token ids, test answers in full. Trained weights (`state.pt`, about 1 GB
  each) stay on the machine and are listed with their SHA-256 in the archive's manifest.
- **Order:** C2 seeds 1-3, then C1 seeds 1-3 (about 6-7 hours on the main run's machine). A trajectory that does not
  finish is reported as missing; if the compute runs out, the last rounds of C1 go first.
- A pilot (tiny sizes) runs first only to find bugs; its numbers are not results.

### Predictions (written before running)

"Old problems" are the 200 test problems of the main run. A gain is against the unedited model of this replication,
paired on the same problems.

- **R0** On the old problems the unedited model and the positive control come within 3 points of the main run
  (10.5% and 45.5%).
- **R1** C2 after round 3 gains at least 15 points with z >= 2 in **every** seed; the mean of the three seeds is within
  10 points of the main run's 44.5%.
- **R2** C2 **stops improving**: on all 400 problems, the mean over the seeds of (round 6 minus round 3) is under
  5 points, and in no seed is round 6 above round 3 with z >= 2.
- **R3** C2 after round 6 on the old problems, in every seed: at least 6 answers finish wrong (the main run's unedited
  model: 2), and the share of right answers among finished ones is lower than the unedited model's.
- **R4** C1 is less stable than C2: after round 3 the range over the seeds (largest minus smallest accuracy on the old
  problems) is larger for C1.
- **R5** After round 6 C1 is below C2 on the old problems (means over the seeds).
- **R6** C2 after round 3 gains at least 15 points with z >= 2 on the **fresh** problems in every seed.
- **R7** The old divisor changes the test score by less than 2 paired standard errors, for both rules.

**What counts as "keeps improving":** for a rule, at least 2 of its 3 seeds have round 6 above round 3 by z >= 2 on the 400
problems. This is still gradient training with a fixed rule, so it does not change section 5's criterion for "a small
model improved itself", and Arm C stays outside that criterion.

Whatever comes out is reported. The main run's numbers are kept as they are and shown next to the replication's, not
replaced by them and not averaged with them. The report script (`analysis/make_replication_report.py`) was written
before the run, from these definitions, and tried on fake data built from the main run's files.

The pilot (commit `c6a8ab3`, 6 minutes, exit 0) ran every step and found one thing to change: the log printed the whole
list of installed packages. It still goes to `env_*.json`, no longer to the log.

**2026-11-04, after the replication** (all replication results had been seen): the main run started at `9ca85d0`, 10:11
(UTC+8), and finished in 364 minutes, exit 0, with no change to the plan above. Practice answers (`gen_round*`) are in the
release archive, not in the repository tree, because of their size; the report does not read them. The bfloat16 rounding at
the end of each round was in the code all along and is reported as a limitation in the README; it was not changed.
