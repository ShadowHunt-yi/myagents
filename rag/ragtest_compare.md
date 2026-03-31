# RAG Evaluation Comparison

## Overview

- Baseline: `rag/ragtest_report.hybrid_baseline.json` (manual hybrid weighting)
- Current: `rag/ragtest_report.json` (reranker ranking)
- Conclusion: reranker keeps recall unchanged and improves ranking quality plus answer accuracy.

## Summary Metrics

| Metric | Hybrid Baseline | Reranker | Delta |
| --- | ---: | ---: | ---: |
| Recall@K | 1.0000 | 1.0000 | +0.0000 |
| Hit@K | 1.0000 | 1.0000 | +0.0000 |
| Precision-like | 0.4394 | 0.8939 | +0.4545 |
| Top1 Source Hit Rate | 0.5455 | 1.0000 | +0.4545 |
| Source MRR | 0.7576 | 1.0000 | +0.2424 |
| Answer Accuracy | 0.3515 | 0.5848 | +0.2333 |

## Interpretation

- Recall@K and Hit@K stay at 1.0, so both versions can retrieve the target source.
- Top1 Source Hit Rate improves from 0.5455 to 1.0, so reranker consistently moves the correct source to rank 1.
- Precision-like improves from 0.4394 to 0.8939, so the returned context is much cleaner.
- Source MRR improves from 0.7576 to 1.0, so the correct source appears earlier overall.
- Answer Accuracy improves from 0.3515 to 0.5848, so retrieval improvements propagate to final answers.

## Improved Cases

| Case | Answer Acc | First Match Rank | Source Change |
| --- | ---: | ---: | --- |
| `lore_trailer_vocab` | 0.0000 -> 1.0000 | 3 -> 1 | `['2603.15602v1.pdf', '2603.15594v1.pdf', '2603.15566v1.pdf']` -> `['2603.15566v1.pdf']` |
| `openseeker_two_innovations` | 0.0000 -> 1.0000 | 2 -> 1 | `['2603.15602v1.pdf', '2603.15594v1.pdf']` -> `['2603.15594v1.pdf']` |
| `openseeker_data_scale` | 0.0000 -> 0.8333 | 2 -> 1 | `['2603.15566v1.pdf', '2603.15594v1.pdf', '2603.15602v1.pdf']` -> `['2603.15594v1.pdf']` |
| `lore_git_trailers` | 0.0000 -> 0.3333 | 1 -> 1 | `['2603.15566v1.pdf', '2603.15594v1.pdf']` -> `['2603.15566v1.pdf']` |
| `energetics_three_tasks` | 0.0000 -> 0.3333 | 2 -> 1 | `['2603.15594v1.pdf', '2603.15602v1.pdf', '2603.15566v1.pdf']` -> `['2603.15602v1.pdf', '2603.15594v1.pdf']` |
| `openseeker_browsecomp` | 0.1667 -> 0.3333 | 1 -> 1 | `['2603.15594v1.pdf', '2603.15566v1.pdf']` -> `['2603.15594v1.pdf']` |

## Regressed Cases

| Case | Answer Acc | First Match Rank | Source Change |
| --- | ---: | ---: | --- |
| `energetics_three_costs` | 0.5000 -> 0.0000 | 2 -> 1 | `['2603.15594v1.pdf', '2603.15602v1.pdf', '2603.15566v1.pdf']` -> `['2603.15602v1.pdf', '2603.15594v1.pdf', '2603.15566v1.pdf']` |
| `lore_core_thesis` | 0.8000 -> 0.4000 | 1 -> 1 | `['2603.15566v1.pdf', '2603.15594v1.pdf']` -> `['2603.15566v1.pdf']` |
| `lore_decision_shadow` | 0.4000 -> 0.2000 | 1 -> 1 | `['2603.15566v1.pdf', '2603.15594v1.pdf']` -> `['2603.15566v1.pdf']` |

## Notes

- Most cases now keep fewer `retrieved_sources`, which means less cross-document noise.
- The strongest gains are on semantic ranking cases such as `openseeker_two_innovations` and `lore_trailer_vocab`.
- The regressions do not come from worse source rank. They are more likely answer-generation or keyword-coverage issues.

## Related Files

- `rag/ragtest.py`
- `rag/rag.py`
- `rag/ragtest_report.hybrid_baseline.json`
- `rag/ragtest_report.json`
