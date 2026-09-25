# Recipe 2026-09-v2: exploratory revision after Study 2

In this page pass A is Rater 1 and pass B is Rater 2 of the paper.

Status: EXPLORATORY. The frozen recipe (`2026-09-freeze`, tag `v2026.09-freeze`)
is the pre-specified result; its Study 2 numbers in `docs/results/study2/` and
`outputs/study2/` are the ones to report, and they are unchanged by this work.
The rules below were motivated by failure modes noticed while inspecting the
held-out Study 2 sample, so Study 2 can no longer serve as a blind test of
them. Every threshold was therefore chosen on the development corpus
(recordings of the non-locked participants; the 602 RA-scored recordings for
agreement) and Study 2 was evaluated once, at the end, with both passes. The
v2 numbers describe how far the rules reach on the held-out sample; they are
not a confirmatory result.

Generated 2026-09-16 from `outputs/v2/` (scores), `outputs/v2/dev_agreement.json`
(development agreement) and `outputs/study2_v2/` (Study 2 under v2). Aggregates
only; no clip or participant identifiers.

## Summary

- Two rules make up version 2: a loop guard for Whisper's repeated-segment
  decoding loops (V1) and a narration-clause rule for intrusions (V3). A third
  candidate, a no-anchor intrusion fallback (V2), was tested and rejected; it
  is documented below and is not in the code.
- Correct recall, serial position and transcripts are untouched
  (`n_correct` changes in 0 recordings anywhere).
- Repetitions: development ICC 0.592 to 0.726 (MAE 0.71 to 0.49); Study 2 ICC
  0.079 to 0.185 against pass A and 0.086 to 0.203 against pass B, MAE 0.58
  to 0.26. On the 25 held-out loop clips MAE falls from 7.08 to 0.28.
- Intrusions: development ICC 0.711 to 0.735 (MAE 0.41 to 0.38, bias +0.10 to
  +0.07); Study 2 ICC 0.175 to 0.210 (pass A) and 0.192 to 0.214 (pass B),
  MAE 0.66 to 0.58.
- Part of the remaining repetition gap is definitional. The frozen
  implementation of decision D2 counts any repeated off-list word as a
  repetition, including conversational words outside the recall span that
  are not counted as intrusions; the scorers count only repeated list words,
  which is also the binder definition. Recounted that way, v2 reaches Study 2
  ICC 0.45 to 0.47 with MAE 0.12 and bias +0.11 (human ceiling 0.78, MAE
  0.04). That decision belongs to the group and is not made here.

## What changed

Version 2 lives behind a switch. `SWC_RECIPE_VERSION` (default `2026-09-freeze`,
alternative `2026-09-v2`) sets `recipe.RECIPE_VERSION`; the scorers also accept
`recipe_version=` directly. Under the default nothing below runs: the frozen
tables are byte-identical (sha1 of `outputs/scores_word_recall.csv` before and
after: `f818f63e106f317fa41b2a9ecb1fe9bb4a47fc5d`; `outputs/scores_serial.csv`:
`be40fe4e76ed37fc08a2f5db360ec1e5dcac4cd8`), the test suite passes and
`tools/pipeline/validate_recipe.py --check` passes (267 numbers). Under v2 the
scorers refuse to write into `outputs/` and take `--out-dir`; the manifest and
both score tables carry `recipe_version 2026-09-v2`, and every output
directory written under v2 (including a Study 2 run) gets a README and a
report banner marked EXPLORATORY.

| Rule | What it does | Token note |
|---|---|---|
| V1 loop guard | Before classification, a block of one or more consecutive tokens that verbatim repeats the block immediately before it (compared on a contraction-harmonised form, so "can't" and "can t" match), and none of whose tokens the word-timestamp decode of the same audio supports (probability at least 0.05, counted over the whole recording), is removed. Largest block at the latest position first, repeated until nothing qualifies; the first occurrence of anything is never removed; inert when the word decode is missing. Removed tokens stay in the token table as `deleted`. | `loop guard: dropped duplicate segment` |
| V3 narration clause | After every credit is known, intrusion tokens (and repeated intrusions) in a clause with no list word and no credited token, at least 4 content tokens and at least one first/second-person pronoun, auxiliary, modal or cognition verb are not counted. Credits, near misses, rescue and the span are untouched, so `n_correct` cannot change. | `narration clause: not counted` |

New v2-only score columns: `n_loop_guard_dropped`, `n_intrusions_narration`.
`n_intrusions_all` keeps its pre-freeze meaning (every intrusion token,
narration included).

## Development-set basis of every threshold

All constants are in `src/recipe.py` with the same rationale in a comment.

| Constant | Value | Basis (development corpus only) |
|---|---|---|
| `LOOP_SUPPORT_PROB_MIN` | 0.05 | Words inside the word decode's own looped segments have median probability 0.0 and 83.5 percent sit below 0.05; 1.4 percent of ordinary development words do. The flag count is flat from 0.01 to 0.2 (146 to 152 of 172 duplicate segments) and the 602-recording ICC moves only 0.725 to 0.734 across floors, so the floor is a structural choice, not an ICC choice. |
| `LOOP_MIN_BLOCK` | 1 | 602 RA-scored recordings, repetitions ICC 0.726 with blocks of 1 versus 0.690 (2) and 0.680 (3). Single-token touches are trailing stutters the word decode hears once and the RA did not count. |
| `LOOP_SCOPE` | anywhere | 0.726 anywhere versus 0.671 trailing only; development loops also occur mid-transcript. |
| Support as a whole-recording multiset | | Word-decode loops stack their copies at the window edge with zero duration and probability near zero, so the floor already separates them; time alignment was not needed. |
| Contraction harmonisation | | The two decodes tokenise contractions differently ("can't" against "can t"). Of the 1,053 tokens the first implementation dropped, 8 were contraction-like and none of the 8 became supported after harmonisation on both sides; the harmonised guard drops 1,046 tokens (2 recordings differ) and exists so the comparison is not tokeniser dependent. |
| Speaking rate | rejected | The 99.5th percentile of ordinary development segments (4.48 words per second) flags only 10 of 148 unsupported duplicate segments. |
| `NARRATION_MIN_CONTENT` | 4 | 0.718 (4 tokens) versus 0.709 (5 tokens) alone; 22 of the 48 development candidate clauses have exactly 4 content tokens; the RA never counted such a clause in full (0 of 6). |
| `NARRATION_PRON_AUX` | fixed list | Set a priori (pronouns, contractions, auxiliaries, modals, cognition verbs); covers 85.4 percent of development candidate clauses, the rest being bare noun runs and Whisper noise that should stay counted. Not tuned. |
| Post-credit form of V3 | | Deleting narration clauses before classification removed near-miss and rescue credits (`n_correct` changed in 6 of 2,858 development recordings, 1 of 601 and away from the RA). The rule therefore runs after every credit is known and skips clauses containing a credited token. |

## Tested and rejected: the no-anchor intrusion fallback

When no list word is credited, the R9 recall span has no anchor and the frozen
recipe counts 0 intrusions. Thirteen held-out clips are genuine wrong-word
lists that the humans scored and the machine did not, which motivated a
fallback: count intrusion tokens over the whole recording when fewer than K
list words are credited. It was rejected on development evidence.

- Alone, the fallback lowers the development intrusion ICC from 0.711 to 0.322
  (bootstrap difference -0.389, 95 percent CI [-0.562, -0.051]); 24 development
  recordings are affected, 12 toward and 10 away from the RA, because
  development no-anchor recordings are dominated by bystander and off-task
  speech.
- Combined with the loop guard and the narration rule it is a statistical tie
  on development (ICC 0.723 versus 0.735 without it; 27 changes, 21 toward
  and 5 away, against 10 changes, 9 toward and 0 away).
- On the 40 held-out clips with no credited word the humans average 0.98
  intrusions. The frozen recipe reads 0 there (MAE 0.97, bias -0.97); the
  three-rule version over-counts (MAE 2.35, bias +2.25), because the away clips
  are research-assistant conversation, bystanders, Whisper noise and short
  give-ups in one- to three-token clauses. Over all 585 clips the three-rule
  version reaches ICC 0.254 against pass A but with MAE 0.67 and bias +0.30,
  worse than the frozen recipe on both.
- A whole-recording non-task detector would be the natural next step and was
  not part of this work.

The fallback is not in the code; the tests assert that no fallback constant,
column or token note exists under v2.

## Which decisions were informed by Study 2 inspection

1. The existence of the failure modes: Whisper trailing duplicate segments,
   multi-chunk repeats and single-word stutters inflating repetitions (25
   clips, 181 of 399 machine repetition tokens); genuine intrusions dropped in
   clips with no credited list word (13 clips); participant narration inside
   the span (16 clips). This is what motivated looking for these patterns.
2. The decision to make the loop rule segmentation-independent (token blocks,
   not whole segments) came from the 2 multi-chunk Study 2 repeats; the choice
   between the block rule and a segment-only rule was then made on development
   RA scores (0.726 versus 0.654).
3. The decision to consider single-token blocks at all came from the 3 Study 2
   stutters; the block minimum was chosen on development (see table).
4. The decision to test a no-anchor fallback and a narration rule at all came
   from the held-out inspection; the content-token minimum, the pronoun list
   and the post-credit form were chosen on development, and the fallback was
   rejected on development evidence before its Study 2 numbers were read as
   anything but descriptive.
5. The lead's loop flag (`outputs/study2_prelim/diag_recall_rep_int.csv`) was
   used only to count catches after the rule was fixed; the one uncaught clip
   and the touched non-loop clips are reported, not tuned away.
6. Sequence disclosure: pre-classification variants of the narration rule were
   evaluated on development, then on Study 2, before the post-credit form was
   evaluated on development and then Study 2. The post-credit form was
   motivated by a development finding (lost near-miss credit), but the Study 2
   numbers of the earlier form already existed when it was chosen. The
   three-rule combination including the fallback was also run on Study 2
   before the fallback was dropped; the drop rests on the development numbers
   above.

No threshold, list entry or rule variant was selected by comparing Study 2
numbers.

## Development numbers (RA hand scores, locked participants excluded)

ICC(2,1), MAE and bias (machine minus RA), 602 recordings (601 for intrusions).
Changes are recordings whose machine value moved, and whether toward or away
from the RA.

| Measure | Frozen ICC / MAE / bias | v2 ICC / MAE / bias | Changed | Toward | Away |
|---|---|---|---|---|---|
| Correct | 0.983 / 0.17 / +0.02 | 0.983 / 0.17 / +0.02 | 0 | 0 | 0 |
| Repetitions | 0.592 / 0.71 / +0.60 | 0.726 / 0.49 / +0.37 | 31 | 30 | 1 |
| Intrusions | 0.711 / 0.41 / +0.10 | 0.735 / 0.38 / +0.07 | 10 | 9 | 0 |

The one intrusion recording not counted as toward or away moved without
changing its distance to the RA. The one repetition recording that moves away
has RA 9 repetitions against a 4-word recall that both decodes heard once.

Corpus-wide reach (2,858 development recordings; 585 locked): the loop guard
touches 164 development recordings (862 tokens) and 31 locked (184 tokens); a
narration clause is present in 154 development and 48 locked (1,999 and 483
tokens not counted). `n_correct` changes in 0 recordings anywhere; repetitions
change in 157 development and 30 locked; intrusions in 89 development and 20
locked. Machine repetition tokens over the development corpus fall from 2,614
to 1,788.

## Study 2 numbers, frozen versus v2 (both passes, evaluated once)

From `outputs/study2/study2_score_agreement.csv` and
`outputs/study2_v2/study2_score_agreement.csv`. ICC(2,1) / MAE / bias
(machine minus human). Human ceilings (pass A versus pass B) are unchanged:
repetitions 0.781 / 0.04, intrusions 0.891 / 0.15.

| Measure | Comparison | n | Frozen | v2 |
|---|---|---|---|---|
| Recall correct | vs pass A | 585 | 0.971 / 0.21 / +0.01 | 0.971 / 0.21 / +0.01 |
| Recall correct | vs pass B | 583 | 0.968 / 0.21 / -0.00 | 0.968 / 0.21 / -0.00 |
| Recall correct | vs mean | 583 | 0.973 / 0.21 / +0.00 | 0.973 / 0.21 / +0.00 |
| Recall repetitions | vs pass A | 585 | 0.079 / 0.58 / +0.56 | 0.185 / 0.26 / +0.25 |
| Recall repetitions | vs pass B | 583 | 0.086 / 0.58 / +0.57 | 0.203 / 0.26 / +0.25 |
| Recall repetitions | vs mean | 583 | 0.083 / 0.58 / +0.57 | 0.197 / 0.27 / +0.25 |
| Recall intrusions | vs pass A | 585 | 0.175 / 0.66 / +0.17 | 0.210 / 0.58 / +0.09 |
| Recall intrusions | vs pass B | 583 | 0.192 / 0.65 / +0.15 | 0.214 / 0.58 / +0.06 |
| Recall intrusions | vs mean | 583 | 0.186 / 0.65 / +0.16 | 0.215 / 0.57 / +0.07 |
| Primacy | vs pass A / B / mean | 585 / 583 / 583 | 0.963 / 0.968 / 0.969 | identical |
| Middle | vs pass A / B / mean | 585 / 583 / 583 | 0.954 / 0.950 / 0.959 | identical |
| Recency | vs pass A / B / mean | 585 / 583 / 583 | 0.974 / 0.973 / 0.977 | identical |

Exact-match and within-1 rates: repetitions 0.83 / 0.89 frozen to 0.88 / 0.94
v2 (both passes); intrusions 0.69 to 0.70 / 0.87 to 0.88 frozen to 0.71 / 0.89
v2. Serial position and transcript WER tables are identical because
`n_correct` and the transcripts are unchanged.

Reach on the held-out clips (descriptive):

- Loop guard: 24 of the lead's 25 loop clips are touched; the miss is a
  five-fold stutter that the word decode independently supports (kept by
  design). 7 non-loop clips are touched (6 of them change their repetition
  count, all toward the humans; 3 are single-token drops). Machine repetition
  tokens fall from 399 to 217 (humans: 70 and 69); on the 25 loop clips from
  181 to 11 (humans 4), MAE 7.08 to 0.28, 24 clips changed and all toward the
  human mean; on the 560 non-loop clips from 218 to 206, MAE 0.29 to 0.26.
  `n_correct` changes in 0 clips.
- Narration rule: 20 clips change their intrusion count (17 toward the human
  mean, 1 away, 2 at equal distance). A narration clause is present in 48
  clips; on those MAE falls from 1.66 to 0.95 and bias from +1.14 to +0.41.
  The 40 clips with no credited word stay at 0 as in the frozen recipe (MAE
  0.97, bias -0.97); they are the fallback's territory and are left for a
  non-task detector.

Reading: the loop guard is the one change with unambiguous support on both
corpora. The narration rule moves in the same direction on both corpora with
a small effect and no away movement on development.

## The repetition convention: evidence for the group

Decision D2 (10 September 2026) says a repeated intrusion counts as both a
repetition and an intrusion, and the typed laboratory scoring instructions
say the same. The scanned scoring-binder page defines repetitions as repeated
List A words. The frozen implementation of D2 counts every repeated off-list
word as a repetition, whether or not the word was counted as an intrusion;
on the held-out sample that is 106 of the machine's 399 repetition tokens,
almost all conversational words ("to", "this", "you", "yeah"), and 69 of the
87 outside the looped recordings fall outside the recall span. That is an
inconsistency in the implementation rather than a convention the scorers
follow.

What the scorers did, measured on their own transcripts (a repeated off-list
word found by the same classifier; the scorer's repetition count compared
with the list-word-only and the D2 value):

| Scorers | Recordings with a repeated off-list word (tokens) | Count equals list words only | Count equals D2 | Neither |
|---|---|---|---|---|
| Study 2 pass A | 23 (40) | 20 | 1 | 2 |
| Study 2 pass B | 20 (28) | 16 | 2 | 2 |
| Development RAs (594 transcripts) | 35 (59) | 18 | 3 | 14 |

The repeated off-list words in the scorers' transcripts are the same
conversational words ("this", "to", "not", "is"); genuine repeated wrong
answers ("sink", "wall", "struggle") are a handful per pass. An earlier
version of this page reported 0 of 17, 1 of 14 and 1 of 10; those
denominators were recordings in which the machine transcript had a repeated
off-list word and the scorer's transcript contained the same word twice,
which counted repeated function words, and they are superseded by the table
above.

Re-counting the machine repetitions as repeated list words only (no code
change to the frozen recipe; a column-level recount from the token table)
gives:

| Version and convention | Development (602) | Study 2 vs pass A (585) | Study 2 vs pass B (583) | Study 2 vs mean (583) |
|---|---|---|---|---|
| Frozen, D2 as implemented (as reported) | 0.592 / 0.71 / +0.60 | 0.079 / 0.58 / +0.56 | 0.086 / 0.58 / +0.57 | 0.083 / 0.58 / +0.57 |
| Frozen, list words only | 0.634 / 0.59 / +0.46 | 0.110 / 0.39 / +0.38 | 0.115 / 0.40 / +0.38 | 0.113 / 0.40 / +0.38 |
| v2, D2 as implemented | 0.726 / 0.49 / +0.37 | 0.185 / 0.26 / +0.25 | 0.203 / 0.26 / +0.25 | 0.197 / 0.27 / +0.25 |
| v2, list words only | 0.778 / 0.39 / +0.24 | 0.449 / 0.12 / +0.11 | 0.473 / 0.12 / +0.11 | 0.474 / 0.12 / +0.11 |
| Human ceiling (pass A vs pass B) | | | | 0.781 / 0.04 / -0.00 |

Exact agreement with either pass rises from 0.83 (frozen, D2) to 0.93 (v2,
list words only). The group can keep D2 (then the implementation should
count only repeated words that were themselves counted as intrusions, and
the human passes should be re-scored under it, or the comparison flagged) or
adopt the list-word convention the scorers use and the binder states (then
the repeated-intrusion term is dropped from the repetition column and the
change is reported as a convention change, not a tuning). Either way the
machine intrusion count is unaffected.

The list-word repetitions themselves (131 machine tokens outside the looped
recordings against 66 in pass A): 88 are also repeated in the scorer's
transcript, and the scorers counted most of those (the difference being
self-corrections and words spoken under the breath, which the typed
instructions exclude); 43 are words Whisper wrote twice that the scorer
heard once.

## Caveats

- Everything here is exploratory for the reason stated at the top; a
  confirmatory test needs a fresh held-out sample.
- ICC on Study 2 repetitions is intrinsically unstable (90 percent of human
  values are zero; an always-zero predictor has MAE 0.12); MAE, bias and exact
  agreement carry the information.
- The loop guard's support test is position blind: when a repeated pair is
  supported once, the guard may delete a supported middle copy instead of the
  unsupported trailing copy; counts are the same, the token table may show
  the other occurrence.
- When both decodes agree on a stutter the guard keeps it; whether those are
  speech cannot be settled without audio.
- The remaining repetition bias on development under v2 (+0.37 under D2,
  +0.24 under the list-word convention) is genuine machine over-counting that
  the loop guard does not address.

## Reproduction

    python3 -m pytest -q
    python3 tools/pipeline/validate_recipe.py --check          # default version, must pass
    SWC_RECIPE_VERSION=2026-09-v2 SWC_RECIPE_CONFIG=ra_informed \
        python3 src/score_wordrecall.py --out-dir outputs/v2
    SWC_RECIPE_VERSION=2026-09-v2 SWC_RECIPE_CONFIG=ra_informed \
        python3 src/score_serial.py --out-dir outputs/v2
    python3 tools/pipeline/dev_agreement_v2.py --scores outputs/v2/scores_word_recall.csv
    python3 tools/pipeline/study2_reliability.py --pass-a <passA.xlsx> --pass-b <passB.xlsx> \
        --scores-recall outputs/v2/scores_word_recall.csv \
        --scores-serial outputs/v2/scores_serial.csv --out-dir outputs/study2_v2
