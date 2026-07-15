Our final LLM configuration uses a 1024-vocabulary BPE tokenizer instead of the baseline byte-level tokenizer.
The tokenizer compresses the mixed English-Hindi text by a factor of 2.19x, effectively expanding the context window and token throughput per training step.
We enabled weight tying between the token embedding table and the output prediction head, saving 163,840 parameters.
We replaced learned absolute position embeddings with Rotary Position Embeddings (RoPE), saving an additional 20,480 parameters and enhancing the model's capacity for positional relationships.
The saved parameters were reinvested to expand the model embedding width to 192 dimensions with 6 heads and 4 layers, totaling 1,976,448 parameters.
The context window size was increased to 256, allowing the model to capture longer dependencies and train on twice as many tokens per step.
We transitioned the optimizer to AdamW with a weight decay of 0.1, linear learning rate warmup of 200 steps, and cosine learning rate decay.
We introduced gradient clipping at 1.0 to prevent gradient spikes and stabilize training.
This combination of tokenizer compression, parameter optimization via weight tying/RoPE, wider dimensionality, longer context, and structured learning rate scheduling successfully reduced dev BPB from 2.3718 to 1.8292.
