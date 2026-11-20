# BlueDot precision amendment · 20 November 2026

The two first Qwen2.5 runtime controls returned repeated punctuation up to the cap in FP16. The protocol stopped before any candidate-bank prompt. The two diagnostic outputs and their original manifest are preserved.

Use the original TARA checkpoint's previously validated FP32/MPS/eager runtime. All tasks, conditions, order, token cap, endpoints, and the two required controls stay fixed. This change was decided from the controls before any bank inference on Qwen2.5.
