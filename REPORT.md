# Drug Label Retrieval — Evaluation Report

_Generated 2026-09-14_

A retrieval system over official FDA drug labelling. Answers medication
questions with a citation on every factual claim, and refuses when the labelling
does not support an answer.

**Scope.** An information retrieval tool over published labelling, for
healthcare professionals. It reports what the label says. It does not give
patient-specific advice, does not recommend treatment, does not diagnose, and is
not a medical device.

## Headline

recall@10 improved from **0.000** (fixed / 800, dense) to **0.771** (section / 1000, dense+lexical, rerank) across 22 measured configurations.

## Corpus and evaluation set

- **Source**: openFDA drug labelling, snapshot `2026-09-09`
- **Partitions used**: 2 of 14
- **Filter**: human prescription drugs with indications, contraindications and dosage populated; one label per molecule
- **Evaluation set**: 35 answerable questions, 40 unanswerable questions, all written by hand

Data courtesy of the U.S. Food and Drug Administration. The FDA does not endorse this tool. Public domain.

## Ablation

Each row is one configuration, measured against the same question set. Changes
were made one at a time so that each row's contribution is attributable.

| Configuration | recall@10 | recall@50 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| baseline: fixed 800 chunks, dense only | 0.000 | 0.000 | 0.000 | 0.000 |
| baseline: fixed 800 chunks, dense only | 0.250 | 0.500 | 0.250 | 0.250 |
| baseline: fixed 800 chunks, dense only | 0.400 | 0.600 | 0.133 | 0.195 |
| baseline: fixed 800, dense only, 1 label per drug | 0.700 | 0.850 | 0.354 | 0.438 |
| baseline: fixed 800, dense only, 35 questions | 0.629 | 0.829 | 0.337 | 0.408 |
| section chunks 1000, dense only | 0.429 | 0.543 | 0.208 | 0.263 |
| section chunks 1000, dense only (labels re-resolved) | 0.429 | 0.543 | 0.284 | 0.321 |
| section chunks 1000, dense only, questions shortened | 0.343 | 0.429 | 0.170 | 0.215 |
| section chunks 1000, dense only, questions shortened | 0.343 | 0.429 | 0.170 | 0.215 |
| baseline: fixed 800, shortened questions | 0.343 | 0.429 | 0.170 | 0.215 |
| section 1000, ENRICHED, dense only | 0.314 | 0.400 | 0.160 | 0.197 |
| section 1000, enriched, dense only | 0.543 | 0.657 | 0.265 | 0.330 |
| section 1000, NO enrichment, dense only | 0.571 | 0.743 | 0.390 | 0.433 |
| fixed 800, dense only | 0.388 | 0.543 | 0.256 | 0.268 |
| section 1000, dense only | 0.564 | 0.743 | 0.410 | 0.454 |
| section 1000, lexical only | 0.057 | 0.086 | 0.034 | 0.040 |
| section 1000, dense only | 0.564 | 0.743 | 0.410 | 0.454 |
| section 1000, lexical only | 0.057 | 0.086 | 0.034 | 0.040 |
| section 1000, hybrid RRF | 0.536 | 0.743 | 0.405 | 0.440 |
| section 1000, lexical only (OR terms) | 0.557 | 0.657 | 0.328 | 0.380 |
| section 1000, hybrid RRF (fixed lexical) | 0.664 | 0.757 | 0.493 | 0.523 |
| section 1000, hybrid + cross-encoder rerank | 0.771 | 0.800 | 0.634 | 0.669 |

## Where the gains came from

| Question type | recall@10 | nDCG@10 |
|---|---|---|
| **ALL** | 0.771 | 0.669 |
| direct_lookup | 0.800 | 0.763 |
| named_entity | 0.750 | 0.470 |
| paraphrase | 0.714 | 0.627 |

| Question type | baseline: fixed 800 chunks, dense  | section 1000, hybrid RRF | change |
|---|---|---|---|
| ALL | 0.000 | 0.536 | +0.536 |
| direct_lookup | 0.000 | 0.537 | +0.537 |

## Abstention

Threshold metric: **top score**  
Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`  
Sets: 35 answerable, 40 unanswerable

| Threshold | Correctly refuses unanswerable | Wrongly refuses answerable |
|---|---|---|
| -10.455 | 0% | 0% |
| -9.939 | 3% | 0% |
| -9.422 | 5% | 0% |
| -8.906 | 15% | 0% |
| -8.390 | 22% | 0% |
| -7.873 | 25% | 0% |
| -7.357 | 30% | 0% |
| -6.841 | 32% | 0% |
| -6.324 | 38% | 0% |
| -5.808 | 43% | 0% |
| -5.292 | 48% | 0% |
| -4.775 | 50% | 0% |
| -4.259 | 52% | 0% |
| -3.743 | 60% | 0% |
| -3.226 | 65% | 0% |
| -2.710 | 70% | 0% |
| -2.193 | 72% | 0% |
| -1.677 | 75% | 0% |
| -1.161 | 82% | 0% |
| -0.644 | 85% | 3% |
| -0.128 | 90% | 3% |
| +0.388 | 92% | 3% |
| +0.905 | 98% | 3% |
| +1.421 | 100% | 3% **&larr; chosen** |
| +1.937 | 100% | 6% |
| +3.486 | 100% | 11% |
| +4.519 | 100% | 17% |
| +5.035 | 100% | 20% |
| +5.552 | 100% | 31% |
| +6.068 | 100% | 34% |
| +6.584 | 100% | 43% |
| +7.101 | 100% | 63% |
| +7.617 | 100% | 77% |
| +8.134 | 100% | 89% |
| +8.650 | 100% | 94% |
| +9.166 | 100% | 97% |

**Operating point +1.421** — correctly refuses 100% of unanswerable questions while wrongly refusing 3% of answerable ones.

Both numbers, always together. Correct abstention alone is meaningless: a system that refuses everything scores 100%.

## Citation integrity

Every factual sentence in a generated answer must carry a marker resolving to a
passage that was actually supplied. This is checked in code, not requested in
the prompt: the answer is parsed, markers are resolved against the supplied
passages, and any factual sentence without one rejects the whole answer. The
system regenerates once with specific feedback, then abstains.

**This verifies citation integrity, not accuracy.** A marker can resolve
correctly to a passage that does not in fact support the claim. Measuring that
requires human review, which has not been done — see Limitations.

## Limitations

- **Faithfulness has not been measured.** Citations are verified to resolve;
  whether each cited passage actually supports its claim has not been checked by
  hand. One observed case: an answer about breastfeeding cited a passage about
  sun exposure alongside a correct one. The marker resolved; the support did not
  exist.
- **The evaluation set is small.** Differences below roughly 0.10 on 35
  questions are not distinguishable from noise, and no confidence intervals are
  reported.
- **The unanswerable set covers one failure mode.** All questions are
  "drug not in corpus". Questions about a section a label lacks, or about
  information labelling never contains (cost, comparative efficacy), are likely
  harder and are not represented.
- **Several unanswerable questions are meta-questions about the dataset**
  rather than clinical questions, which probably inflates the measured
  separation between the two score distributions.
- **A subset of questions is multi-clause** and fails in every configuration
  tested. These measure question phrasing rather than retrieval quality and cap
  the achievable recall.
- **Tabular content is retrieved but not usable.** Passages consisting mostly of
  numeric tables occupy candidate slots and can never serve as a citation.
- **One partition of fourteen** was ingested, and one label per molecule
  retained. Results do not generalise to the full corpus without re-measurement.

## Reproducing

```bash
docker compose up -d                          # or a local Postgres with pgvector
python -m drug_label_rag.ingest.download --limit 1
python -m drug_label_rag.db.session
python -m drug_label_rag.ingest.run --reset
python -m drug_label_rag.eval.retrieval --rerank
python -m drug_label_rag.eval.abstention
```
