# Run Log

## Run 1: Baseline Model
- **Hypothesis**: Baseline hyperparameters, architecture, and byte-level tokenizer are mediocre.
- **Change**: Run starter code as-is.
- **Dev BPB**: 2.3718
- **Conclusion**: This is our reference score.

## Run 2: BPE Tokenizer (1024 vocab)
- **Hypothesis**: Replacing byte-level tokenization with BPE (vocab 1024) will compress Hindi sequences, enabling the model to see more text within its 128 context window and improve performance.
- **Change**: Loaded BPETokenizer, vocab size changed to 1024, parameters increased to 1,585,600.
- **Dev BPB**: 2.2329
- **Conclusion**: Keep it. BPE tokenization compresses sequence length by 2.19x, increasing context window density and reducing dev BPB.

## Run 3: Weight Tying
- **Hypothesis**: Tying input embedding weights with output prediction weights will reduce model parameters and regularize learning.
- **Change**: Enabled weight tying in `model.py` GPT definition. Parameter count dropped to 1,421,760.
- **Dev BPB**: 2.2575
- **Conclusion**: Keep it. The slight regularizing BPB increase on the smaller model allows us to scale up embedding dimension and blocks without violating the 2M cap.

## Run 4: Cosine LR Warmup + AdamW
- **Hypothesis**: Linear learning rate warmup combined with a cosine learning rate decay and AdamW optimizer (with weight decay 0.1 and gradient clipping at 1.0) will stabilize gradients and accelerate learning.
- **Change**: Swapped SGD for AdamW, implemented custom LR schedule with 200-step linear warmup, and clipped gradients in `train.py`.
- **Dev BPB**: 2.1630
- **Conclusion**: Keep it. The learning rate scheduling and optimizer switch drastically improved convergence rate.

## Run 5: RMSNorm + Scaled Init
- **Hypothesis**: Swapping standard LayerNorm for RMSNorm and scaling weight initialization std down to 0.02 will accelerate attention convergence.
- **Change**: Implemented RMSNorm and changed init std in `model.py`.
- **Dev BPB**: 2.2322
- **Conclusion**: Revert it. The small network (4 layers) under-initialized at std=0.02, leading to slow training and regression. Returned to LayerNorm and std=0.05.

## Run 6: Rotary Position Embeddings (RoPE)
- **Hypothesis**: Replacing learned absolute position embeddings with mathematical RoPE will improve relative position learning and save parameter footprint.
- **Change**: Replaced embedding addition with Cos/Sin rotation queries/keys inside `model.py` self-attention. Saved 20,480 parameters.
- **Dev BPB**: 1.9499
- **Conclusion**: Keep it. Eliminating absolute positional embeddings improved relative token attention and yielded a huge dev BPB drop.

## Run 7: Wider Model (n_embd = 192)
- **Hypothesis**: Reinvesting the parameter savings from weight tying and RoPE into the model width will improve representational capacity.
- **Change**: Set `n_embd = 192`, `n_head = 6`, `n_layer = 4` in `model.py` default config. Parameter count reached 1,976,448.
- **Dev BPB**: 1.9219
- **Conclusion**: Keep it. Increasing model width under the 2M cap successfully improved performance.

## Run 8: block_size = 256 (Longer Context)
- **Hypothesis**: Increasing context window size (`block_size`) from 128 to 256 will capture longer-range dependencies and train the model on 2x more tokens per step without changing parameter count.
- **Change**: Changed `block_size = 256` in `model.py` default config.
- **Dev BPB**: 1.8292
- **Conclusion**: Keep it. Longer context window yields a huge dev BPB improvement by capturing longer context and doubling the number of tokens seen during training.

## Run 9: Custom Devanagari Merge Priority Tokenizer (Human-Directed Feature)
- **Hypothesis**: By adding a custom merge priority weighting factor of 2.0 to Devanagari byte sequences during BPE BPE training, we force BPE to combine Hindi vowel marks (matras) and character clusters first. This should lead to better Hindi token boundaries, improve the validation compression factor (from 2.19x to 2.24x), and lower the dev BPB.
- **Change**: Added Devanagari range matching and a 2.0 frequency boost multiplier in `tokenizer.py` during BPE training. Re-trained the tokenizer merges.
- **Dev BPB**: 1.8199
- **Conclusion**: Keep it. Prioritizing Devanagari character pairs during BPE training improves sequence compression to 2.24x, resulting in more linguistically coherent token boundaries for Hindi and reducing dev BPB to 1.8199.