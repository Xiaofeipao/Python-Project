# Ten-trial attempt (incomplete)

Both tasks use the same frozen policy with max_trials=10 and a 7200-second task budget. Coupon completed 9 successful validation trials; headlines completed 4. Each run stopped on an API read timeout with unknown token usage. No retry or zero-cost assumption was made. No new final test scores are reported. Best train+validation checkpoint models remain at the paths in retention.json; historical best models are unchanged.

Coupon validation improved from 75.8128% after trial 2 to 75.9606% by trial 8 (about +0.148 percentage points); trial 9 did not improve it. Headlines validation stayed at 69.0426% through trial 4; an eight-trial comparison is unavailable. These are validation results, not final test scores.

Known task-wide costs are CNY 0.057425 and 0.013465. Each task retains an unresolved CNY 5.24288 worst-case reservation; this is not an actual charge. Obtain provider usage for the timed-out requests before further API calls. Paths in public logs are redacted.
