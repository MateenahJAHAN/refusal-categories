# Sensitivity to stopping at 96 tokens

Added November 21 after scoring the complete 125-answer run. Its 96-token cap was reached by 104 replies. This sensitivity analysis extends every capped reply from the exact saved token prefix; the initial analysis and labels stay fixed.

Use the same Qwen2.5 checkpoint, chat messages, greedy decoding and FP32 device, with a ceiling of 512 total answer tokens. Leave the 21 original end-of-sequence replies intact. Preserve both original and continued outputs. Read the extended answers under the existing rubric with shuffled new IDs, omitting prior codes, paired probe outputs, and benchmark item IDs from the reading file. The same AI reader already knows the aggregate initial results and some response contents. This is a sensitivity analysis of the original ratings. Report remaining caps, changed ratings, final category counts, and correspondence with the four presentation conditions. No further tuning or automatic exclusions.
