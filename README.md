# 2000-Step LLM Speedrun: Optimized Hindi-English Language Model

An optimized Transformer-based language model built from scratch under strict constraints to compress and model a mixed English and Hindi text corpus. This project successfully improves the baseline GPT model's performance, reducing validation bits-per-byte (bpb) from **2.3718 to 1.8199** in exactly **2,000 steps** on a standard CPU.

---

## 🚀 Performance Overview

| Metric | Baseline GPT | Optimized Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Dev Bits-per-Byte (BPB)** | 2.3718 | **1.8199** | **-0.5519 (23.3%)** |
| **Total Parameters** | 1,339,840 | **1,976,448** | Within 2M Cap |
| **Tokenizer Compression** | 1.00x (Byte-level) | **2.24x (Devanagari BPE)** | +1.24x |
| **Context Window Size** | 128 | **256** | 2x Longer Context |
| **Optimizer Steps** | 2,000 | **2,000** | Within 2K Cap |
| **CPU Training Time** | ~5 minutes | **~13 minutes** | Highly Optimized |

---

## 🛠️ Hard Constraints Met
- **Optimizer Steps**: Final checkpoint trained for exactly 2000 optimizer steps.
- **Parameters**: `1,976,448` parameters, strictly under the **2,000,000 parameter cap**.
- **Pure Python & PyTorch**: No pre-trained weights, custom GPU kernels, or external tokenizers (e.g., HuggingFace). Standard PyTorch and NumPy CPU execution only.
- **Data Integrity**: Trained exclusively on the provided `train_corpus.txt`.

---

## 💡 Optimization Pipeline

### 1. Custom Devanagari BPE Tokenizer (Human-Directed Feature)
Rather than a naive byte-level tokenizer (which treats each Hindi character as 3 separate tokens, destroying context density), we implemented a custom **1024-vocabulary BPE Tokenizer**.
- **Priority Rules**: Multiplies frequencies of Devanagari byte sequences by `2.0x` during BPE training, forcing the early merging of Hindi combining vowel marks (matras) and consonant conjuncts.
- **Sequence Compression**: Shrank the validation sequence length from 159KB to 71,085 tokens (**2.24x compression**), enabling the model to see more text per step.
- **Word-Level Cache**: Word tokenization is cached, reducing Python overhead during training encoding.

### 2. Architectural Optimization
- **Weight Tying**: Shares parameters between the token embedding table and the output prediction head, saving `163,840` parameters.
- **Rotary Position Embeddings (RoPE)**: Swapped learned absolute position embeddings for mathematical query/key complex rotations, saving `20,480` parameters while improving positional relationship scaling.
- **Capacity Scaling**: Reinvested all saved parameters to scale embedding width (`n_embd`) from 160 to `192` (6 attention heads, 32-dim each) and context window (`block_size`) to `256` tokens.

### 3. Hyperparameter & Optimizer Tuning
- **AdamW Optimizer**: Upgraded from standard SGD to AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.1`).
- **LR Scheduling**: Implemented a 200-step linear warmup followed by a cosine decay down to `1e-4` (peak `1e-3`).
- **Gradient Clipping**: Clipped gradients at `1.0` to stabilize step updates.

---

## 📊 Controlled Experiment History

| Run | Change Description | Parameters | Dev BPB | Status |
| :---: | :--- | :---: | :---: | :---: |
| **1** | Baseline configuration as-is | 1,339,840 | 2.3718 | Reference Baseline |
| **2** | 1024-vocab BPE tokenizer | 1,585,600 | 2.2329 | Kept |
| **3** | Enabled Weight Tying | 1,421,760 | 2.2575 | Kept |
| **4** | AdamW + Warmup & Cosine LR schedule + Grad clip | 1,421,760 | 2.1630 | Kept |
| **5** | RMSNorm + std=0.02 init scaling | 1,420,320 | 2.2322 | Reverted to LayerNorm |
| **6** | Rotary Position Embeddings (RoPE) | 1,401,280 | 1.9499 | Kept |
| **7** | Wider model: n_embd=192, n_head=6 | 1,976,448 | 1.9219 | Kept |
| **8** | Longer context: block_size=256 | 1,976,448 | 1.8292 | Kept |
| **9** | **Custom Devanagari BPE Merge Priority** | 1,976,448 | **1.8199** | **Best** |

---

## 🏃 Verification & Running Instructions

Navigate to the `starter` folder:
```powershell
cd starter
```

### 1. Evaluate Final Checkpoint (Sanity Check)
Score the final checkpoint `ckpt.pt` against the validation file:
```powershell
C:\Users\biriy\.conda\envs\notebook-env\python.exe evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
**Expected Output:**
`{"bpb": 1.8199, "n_params": 1976448, "steps": 2000, "tokens_in_eval": 71085, "tokens_scored": 71084}`

### 2. Verify Tokenizer roundtrip
Verify that tokenization is 100% lossless:
```powershell
C:\Users\biriy\.conda\envs\notebook-env\python.exe -c "import tokenizer; tok = tokenizer.load(); text = open('../data/dev_eval.txt', encoding='utf-8').read(); ids = tok.encode(text); decoded = tok.decode(ids); assert decoded == text, 'lossy!'; print('Lossless!', len(text.encode('utf-8')), 'bytes ->', len(ids), 'tokens. Compression:', len(text.encode('utf-8')) / len(ids))"
```

### 3. Verify Parameter Count
Verify the configuration contains exactly 1,976,448 parameters:
```powershell
C:\Users\biriy\.conda\envs\notebook-env\python.exe -c "import model; cfg = model.Config(); cfg.vocab_size = 1024; m = model.GPT(cfg); print('Total parameters:', m.n_params())"
```

### 4. Train from Scratch
To run training from scratch:
```powershell
C:\Users\biriy\.conda\envs\notebook-env\python.exe train.py --data ../data/train_corpus.txt --steps 2000 --out new_ckpt.pt --batch 8 --lr 1e-3
```

---

## 👥 Human/Agent Work Split
- **Human-Directed**: Conceptualized and designed the Hindi BPE merge-priority rule in `tokenizer.py`, tuned hyperparameter runs, evaluated sequence length performance, and verified final score validity.
- **Agent-Generated**: Implemented RoPE query/key rotations, weight-tying network modifications, AdamW training loop scheduling, HTML reporting dashboards, and code style sanitization.
