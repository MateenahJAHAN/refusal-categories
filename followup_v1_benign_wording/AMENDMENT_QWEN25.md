# BlueDot runtime amendment · 20 November 2026

The first frozen run returned 72 truncations without final answers on the cached Qwen3 4B checkpoint. The run retains all raw outputs as a diagnostic. Only completed answers can support the behavioral comparison. The Qwen3.5 4B arm remains valid.

This amendment was recorded before any Qwen2.5 inference. The fallback is the original TARA Qwen2.5-1.5B-Instruct checkpoint, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, using the offline Transformers/MPS runtime. Use the same 24 tasks, three matched labels, order seed, greedy decoding, token cap, and frozen scoring. Record two separate runtime controls before the 72 bank prompts. Stop if either control truncates or misses the expected short answer.

Compare label conditions within each checkpoint and runtime. Checkpoint, quantization, template, and backend differ between checkpoints. Cross-model differences are descriptive.
