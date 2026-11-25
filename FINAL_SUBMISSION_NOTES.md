# Final submission build

The contents of the two submission files contain no author name, student number, account handle or personal home path. Notebook sources, retained outputs and metadata were scanned; both PNG metadata blocks were checked. The notebook imports its companion script through a generic filename pattern instead of an embedded student identifier. Required external submission filenames remain unchanged.

All 14 executed code-cell outputs are retained, and the notebook remains below 3 MB. Model training and prediction code is unchanged: the Python AST excluding the module usage docstring matches the previous CSE-tested model exactly. This is an anonymity update, not a new model or a fresh CSE timing. The existing model configuration was measured at 96.87 seconds on CSE. The original model metrics and source hashes remain historical records; the content-only build change is separately documented in `experiments/submission_privacy_verification.json`.

37 automated tests passed, including the new privacy scan and standalone notebook companion-loader check.

## Last bounded weighting experiment

Only the training CSV was used for selection. The seed-42 validation partition retained every row and equal evaluation weight. Training flags represent feature/label discordance rather than confirmed incorrect labels:

- Regression: 17 fitting rows had revenue <=1000 alongside budget >=1000000 or vote count >=1000.
- Classification: 53 fitting rows had a rating/vote-average discordance relative to fitting-only class quartiles, with positive vote average and at least 50 votes. Zero vote averages were not treated as incorrect labels.

| Weight on flagged fitting rows | Val RMSLE | Val Macro F1 |
| --- | ---: | ---: |
| Existing model | 1.840734 | 0.768874 |
| 0.50 | 1.841137 | 0.769362 |
| 0.75 | 1.844718 | 0.764232 |

Neither setting improved both tasks; the 0.50 classification gain was only0.000488 on one reused development split. No weighting change was adopted, and provided-test labels were not used to select these settings. This rejects these particular rules, not all possible data-quality methods. Raw evidence is in `experiments/final_weight_results.json`.

The retained provided-test scores remain RMSLE1.941241 and Macro F1 0.738150. No new test improvement is claimed. The final download archive contains exactly the two submission files.
