# English-to-German Neural Machine Translation: Experiment Log

**Dataset:** Multi30k (~29,000 training pairs) | **Tokenizer:** Custom BPE (12,000 shared vocabulary) | **Decoding:** Beam Search (`beam_width=3`, `length_penalty=0.7`)

## Summary Progression

| Experiment | Architecture | Key Hyperparameters & Regularization | Best Epoch | Val Loss | Val Acc | Test BLEU |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Exp 1** | 1-Layer GRU | `hidden=256`, no dropout | 5 / 10 | 3.0980 | 45.76% | **13.18** |
| **Exp 2** | 2-Layer GRU | `hidden=512`, `dropout=0.3` | 6 / 11 | 2.9389 | 49.36% | **18.11** |
| **Exp 3** | 2-Layer GRU | `hidden=512`, `dropout=0.3`, label smoothing, weight decay, weight tying | 12 / 17 | 3.5392 | 56.57% | **0.01** |
| **Exp 3a** | 2-Layer GRU | Removed weight tying, `dropout=0.4`, label smoothing (`0.1`), weight decay (`1e-5`) | 14 / 19 | 3.5368 | 56.50% | **24.19** |
| **Exp 4** | 2-Layer BiLSTM | `hidden=512`, `dropout=0.4`, `tanh` state bridges | 20 / 25 | 3.6601 | 53.28% | **21.41** |
| **Exp 5** | 2-Layer BiLSTM + Attention | Bahdanau Attention, linear state bridges, `dropout=0.4` | 11 / 16 | 3.1647 | **62.83%** | **31.50** |

---

## Detailed Experiment Notes

### Experiment 1: Baseline Vanilla GRU
* **Architecture:** Single-layer Encoder-Decoder GRU (`hidden_size=256`), no dropout.
* **Training Dynamics:** Early stopping triggered at Epoch 10.
  * **Best Epoch (5):** Train Loss: `2.3050` | Val Loss: `3.0980` | Val Acc: `45.76%` | LR: `0.001000`
  * **Last Epoch (10):** Train Loss: `1.3461` | Val Loss: `3.2077` | Val Acc: `46.50%` | LR: `0.000500`
* **Test Corpus BLEU Score:** `12.86` (jumped to **`13.18`** after adding regex detokenization in `predict.py`).
* **Conclusion:** Textbook overfitting after Epoch 5, combined with limited representational capacity.
* **Next Steps:** Increase model capacity (width and depth) and add dropout regularization, aiming for ~20 BLEU without Attention.

### Experiment 2: Deeper & Wider GRU with Dropout
* **Changes:** Scaled to a 2-layer GRU (`hidden_size=512`) and added `dropout=0.3` on embeddings and between recurrent layers.
* **Aim:** Increase representational capacity while reducing overfitting.
* **Training Dynamics:** Early stopping triggered at Epoch 11.
  * **Best Epoch (6):** Train Loss: `2.0927` | Val Loss: `2.9389` | Val Acc: `49.36%` | LR: `0.001000`
  * **Last Epoch (11):** Train Loss: `1.4733` | Val Loss: `2.9574` | Val Acc: `50.54%` | LR: `0.000500`
* **Test Corpus BLEU Score:** **`18.11`** (+4.93 BLEU gain).
* **Conclusion:** Increasing capacity significantly improved translation quality, though the model still began overfitting after Epoch 6.
* **Next Steps:** Add label smoothing (`0.1`), weight decay (`1e-5`), and weight tying (`self.fc.weight = self.embedding.weight`) to improve generalization.

### Experiment 3: Regularization & Weight Tying (Failed Run)
* **Changes:** Added label smoothing (`0.1`), weight decay (`1e-5`), and weight tying between `self.embedding` and `self.fc`.
* **Aim:** Further reduce overfitting.
* **Training Dynamics:** Early stopping triggered at Epoch 17.
  * **Best Epoch (12):** Train Loss: `2.0616` | Val Loss: `3.5392` | Val Acc: `56.57%` | LR: `0.000500`
  * **Last Epoch (17):** Train Loss: `1.7910` | Val Loss: `3.5754` | Val Acc: `56.29%` | LR: `0.000250`
* **Test Corpus BLEU Score:** **`0.01`**
* **Conclusion:** Weight tying caused a destructive feedback loop during autoregressive generation—sharing the exact same matrix between `self.embedding` and `self.fc` (without residual connections or LayerNorm) amplified high-frequency token biases and locked beam search into repeating loops. Label smoothing and weight decay remain useful, but weight tying must be removed.
* **Next Steps:** Remove weight tying and increase `dropout` to `0.4`.

### Experiment 3a: Regularized 2-Layer GRU (No Weight Tying)
* **Changes:** Removed weight tying and increased `dropout` from `0.3` to `0.4` (keeping label smoothing `0.1` and weight decay `1e-5`).
* **Aim:** Fix the autoregressive generation collapse from Experiment 3 and surpass the 20 BLEU target without Attention.
* **Training Dynamics:** Early stopping triggered at Epoch 19.
  * **Best Epoch (14):** Train Loss: `2.1258` | Val Loss: `3.5368` | Val Acc: `56.50%` | LR: `0.000500`
  * **Last Epoch (19):** Train Loss: `1.8740` | Val Loss: `3.5420` | Val Acc: `56.82%` | LR: `0.000250`
* **Test Corpus BLEU Score:** **`24.19`** (+6.08 BLEU over Exp 2).
* **Conclusion:** Removing weight tying resolved the decoding loop and pushed the model to the upper performance ceiling for non-attention RNNs on Multi30k. Qualitative inspection shows fluent short phrases, though the model still confuses rare nouns and loses detail on longer sentences due to the single-vector bottleneck.
* **Next Steps:** Benchmark a 2-layer Bidirectional LSTM before introducing Attention.

### Experiment 4: Bidirectional LSTM (No Attention)
* **Changes:** Replaced the 2-layer GRU with a 2-layer Bidirectional LSTM encoder (`hidden_size=512`, `dropout=0.4`) and `tanh`-activated linear state bridges (`1024 → 512`) feeding a unidirectional LSTM decoder.
* **Aim:** Compare BiLSTM against the regularized GRU baseline prior to adding Attention.
* **Training Dynamics:** Early stopping triggered at Epoch 25.
  * **Best Epoch (20):** Train Loss: `2.2559` | Val Loss: `3.6601` | Val Acc: `53.28%` | LR: `0.000500`
  * **Last Epoch (25):** Train Loss: `2.0069` | Val Loss: `3.6978` | Val Acc: `53.47%` | LR: `0.000250`
* **Test Corpus BLEU Score:** **`21.41`** (-2.78 BLEU compared to Exp 3a).
* **Conclusion:** Two primary factors explain the drop relative to the 2-layer GRU:
  1. **Higher Parameter Count on Small Data:** Extra LSTM gate matrices and bridge layers increase total parameters, making the model harder to generalize on Multi30k's 29k training pairs when constrained by a single-vector bottleneck.
  2. **Bridge Compression & Activation Loss:** Compressing `1024` concatenated bidirectional features down to `512` loses fine-grained source detail. Furthermore, applying `tanh` to the cell state (`c_n`) artificially squashes an unbounded memory accumulator (whereas keeping `bridge_c` purely linear preserves unbounded cell memory, or widening the decoder to `1024` avoids compression altogether).
* **Next Steps:** Add **Bahdanau (Additive) Attention** and switch to pure linear state bridges.

### Experiment 5: BiLSTM with Bahdanau Attention
* **Changes:** Added Bahdanau (Additive) Attention over all encoder time steps (`encoder_outputs`), switched state bridges to pure linear projections, and activated source padding masks (`attention_mask`).
* **Aim:** Eliminate the single-vector information bottleneck and break past **30+ BLEU**.
* **Training Dynamics:** Early stopping triggered at Epoch 16.
  * **Best Epoch (11):** Train Loss: `1.7759` | Val Loss: `3.1647` | Val Acc: `62.83%` | LR: `0.000500`
  * **Last Epoch (16):** Train Loss: `1.5624` | Val Loss: `3.2247` | Val Acc: `62.63%` | LR: `0.000250`
* **Test Corpus BLEU Score:** **`31.50`** (+10.09 BLEU over Exp 4; +18.32 BLEU over Exp 1 baseline).
* **Conclusion:** Bahdanau Attention dramatically improved translation fluency and long-sentence retention by allowing the decoder to query relevant source tokens dynamically at every step.

---

## Qualitative Analysis: Beam Search Attention Heatmap
* **Observations:** Visualizing `attn_weights` from `decode_step()` confirms strong diagonal alignment on common words (e.g., `weißen → white`, `Zaun → fence`) as well as grammatical phrase reordering between English and German.
* **Remaining Limitations (Rare Nouns):** Despite attending to the exact right source positions (e.g., nearly `1.0` attention weight on *"Boston Terrier"*), the model still substitutes visually similar nouns for rare entities. This stems from two factors:
  1. **Data Scarcity:** Rare nouns appear only a few times in Multi30k's 29k sentences.
  2. **Randomly Initialized Embeddings:** Training `nn.Embedding(12000, 512)` from scratch leaves rare tokens with too few gradient updates to form rich semantic representations.

### Future Roadmap
1. **Attention Visualization:** Plot beam search attention alignments via `decode_step()` *(Completed)*.
2. **Pre-Trained Embeddings:** Initialize `nn.Embedding` with aligned pre-trained subword/word vectors (e.g., BPEmb or fastText) to resolve rare-noun representations.
3. **Transformer Architecture:** Build an Encoder-Decoder Transformer from scratch on the same tokenized pipeline to compare self-attention against Bahdanau RNN attention.
