# boontft v1: 11L XSA4 + EMA + TTT + Partial RoPE + LN Scale + Shared VE128

## Architecture
- 11 transformer layers, 512-dim, 8 heads (4 KV heads, GQA)
- 3x MLP expansion with relu-squared activation
- U-Net skip connections (5 encoder + 6 decoder)
- XSA (Exclusive Self-Attention) on last 4 layers
- Partial RoPE: 16/64 head dims with NTK-aware scaling
- LN Scale: layer-dependent 1/sqrt(layer+1) normalization
- Shared Value Embedding (dim=128) on layers 9-10
- SmearGate + BigramHash(2048)
- Logit softcap at 30.0, tied FP16 embeddings

## Training
- Muon optimizer (WD=0.04, momentum 0.92→0.99 over 1500 steps)
- AdamW for embeddings/scalars (WD=0.04)
- EMA (decay=0.997) — replaces SWA for 0.003 bpb improvement
- Late QAT: int6 STE fake-quantize enabled when LR scale < 0.1
- Gradient clipping at 0.3
- Batch size: 786K tokens, seq_len: 2048
- Warmdown: 3000 iterations

## Evaluation
- TTT: 3 epochs of SGD (lr=0.002, momentum=0.9) on val data, first 2 blocks frozen
- Sliding window eval with stride=64 at seq_len=2048

## Quantization
- Int6 per-row for MLP and attention weights
- Int8 per-row for embeddings
- FP16 passthrough for control tensors
- zstd-22 compression

## Key Differentiators
- Combines EMA (proven 0.003 better than SWA) with TTT (proven 0.002 additive gain)
- No prior submission has combined both EMA and TTT with full XSA+VE+PartialRoPE stack
- SDPA fallback for consumer GPU development (4090 compatible)

## Running
```bash
# 4090 quick test
RUN_ID=test ITERATIONS=500 TRAIN_BATCH_TOKENS=131072 \
torchrun --standalone --nproc_per_node=1 boontft/train_gpt.py

# 8xH100 full submission
RUN_ID=submit SEED=42 \
torchrun --standalone --nproc_per_node=8 boontft/train_gpt.py
```
