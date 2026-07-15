Our final LLM configuration uses a 1024-vocabulary custom BPE tokenizer with prioritized Devanagari merge rules.
During tokenizer training, Devanagari byte pairs receive a 2.0x frequency boost to force early merging of vowel matras and consonant clusters.
This custom tokenizer increases validation sequence compression to 2.24x, resulting in more coherent subword representations for Hindi text.
We enabled weight tying between the token embedding table and the output prediction head to save 163,840 parameters.
Learned absolute positional embeddings were replaced with Rotary Position Embeddings (RoPE), saving 20,480 parameters and enhancing relative positional attention.
The parameter savings were reinvested to expand embedding dimensionality to 192, with 6 heads and 4 layers, totaling 1,976,448 parameters.
We increased the context window size to 256, allowing the model to capture longer dependencies and double the training tokens per step.
The training loop utilizes the AdamW optimizer with a weight decay of 0.1, a 200-step linear learning rate warmup, and cosine learning rate decay.
Gradient clipping is set to 1.0 to stabilize learning.
This combination of Devanagari-aware tokenization, architecture optimization, larger context, and structured scheduling reduced dev BPB from 2.3718 to 1.8199.
