# Deep-Learning
This project builds a small, working GPT-style language model, using only
torch tensor operations for the core math (no nn.TransformerEncoder, no nn.MultiheadAttention).

## 1. What is a Language Model?

A **Language Model (LM)** is a system that predicts the next token given a sequence of
previous tokens:

$$P(x_t \mid x_1, x_2, \dots, x_{t-1})$$

A **Large Language Model (LLM)** is s a system that predicts the next token given a sequence of
previous tokens but for more parameters, more data, more
computation.

## 2. Tokens & Tokenization

**Token** is a basic unit of text an LLM operates on. It is not always a word, it can be a
character, a sub-word piece, or a whole word, depending on the tokenizer.

- Real LLMs such as GPT-4, Llama, etc, use **Byte Pair Encoding (BPE)** or similar sub-word schemes, which
  balance vocabulary size against sequence length by merging frequently co-occurring character pairs.
- For this project we use the simplest possible tokenizer i.e **character-level** , so we can verify
  every step without a large vocabulary getting in the way. The concepts suchas embedding lookup, attention,
  etc are identical regardless of tokenizer granularity.

**Tokenization** is the process of converting raw text into a sequence of integer IDs the model can
consume, using a fixed vocabulary.

## 3. Embeddings

Token IDs are just arbitrary integers i.e the number `5` is not "closer" in meaning to `6` than to
`50`. So the model's first job is to convert each token ID into a **dense vector** for e.g. 32 or 4096
numbers,where it can actually learn meaningful relationships over. This lookup table is the
**embedding layer** i.e a matrix of shape (vocab_size, embedding_dim) where row `i` is the vector for
token `i`. These vectors start random and are learned during training, so that tokens used in similar
contexts end up with similar vectors.

### Positional Embeddings
Attention  has no built-in notion of word order as it treats the input as a *set*, not a
*sequence*. So we add a second embedding table that encodes **position** (0, 1, 2, ...) and add it
elementwise to the token embedding as depicted below. This is how the model learns that "dog bites man" differs from
"man bites dog".

## 4. Self-Attention (the Core Idea)

Self-attention lets each token look at every other token in the sequence and decide "how much to pay
attention to it" when building its own updated representation. For every token we compute three
vectors, learned via linear projections of its embedding:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain?"
- **Value (V)** — "what do I actually offer if you attend to me?"

Attention scores are the dot product of a token's Query with every token's Key (scaled by
$\sqrt{d_k}$ for numerical stability), turned into a probability distribution with softmax, then used
to take a weighted sum of the Value vectors:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

## 5. Causal Masking

An LLM generates text left-to-right . When predicting token `t`, it must not be allowed to see tokens
`t+1, t+2, ...` because it would be cheating and impossible at generation time, since they don't exist
yet. We enforce this with a **causal mask**: It is a lower-triangular matrix that blocks attention to future
positions by setting their score to $-\infty$ before the softmax, so their attention weight becomes 0.

## 6. Multi-Head Attention

A single attention head can only learn one type of relationship at a time for e.g. only syntax, or only
long-range topic. **Multi-head attention** runs several attention heads in parallel, each with its own
learned Q/K/V projections into a smaller sub-space, then concatenates their outputs and projects back
to the full embedding dimension. This lets the model attend to different kinds of relationships
simultaneously.

## 7. Feed-Forward Network, LayerNorm, and Residual Connections

Each transformer block also contains:

- **Feed-Forward Network (FFN)** : It is a small two-layer MLP applied independently to every position,
  giving the model extra capacity to transform the attended representation for e.g as attention mixes
  information across tokens.
- **LayerNorm** :  It normalizes activations to stabilize and speed up training.
- **Residual /skip connections** : Using `x = x + sublayer(x)` instead of `x = sublayer(x)`. This lets
  gradients flow directly through the network during backpropagation, which is essential for training
  deep stacks of layers.

  ## 8. The Transformer Block

Combining everything above with residual connections and LayerNorm gives one **transformer block**:


x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))


(This "pre-norm" ordering i.e normalizing before each sublayer, is what modern LLMs like GPT-2/3 and
Llama use, as it trains more stably than the original "post-norm" transformer.)

## 9. Assembling the Full GPT Model

 We Stack several transformer blocks, add a final LayerNorm, and project back to vocabulary size to get
unnormalized scores over the next-token prediction. This is a minimal but depends on complete GPT-style
architecture.

## 10. Training on a Toy Dataset

Training an LLM is **next-token prediction** i.e for every position in the training sequence, the target
is simply the next token. We slice our tiny corpus into overlapping ( such as context, target) chunks and
minimize cross-entropy loss with Adam.

## 11. Text Generation

At inference time we start from a short prompt and repeatedly run the model, take the logits or unnormalized scores for the
last position, turn them into a probability distribution, and sample the next token. `temperature`
controls randomness (lower = more deterministic, higher = more diverse), and `top_k` restricts sampling
to the k most likely tokens to avoid picking very unlikely ones.


## Notes :

This project is a minimal, from-scratch skeleton of LLMs, but  for real LLMs  we add many refinements on top of exactly
these building blocks. For example:

- **Tokenization**: swap the character-level tokenizer for BPE (e.g. `tiktoken`) or a trained
  SentencePiece/WordPiece model so that sub-word tokenizers give much better coverage and shorter sequences.
- **Positional encoding**: modern models often use **RoPE** (Rotary Position Embeddings) instead of
  learned absolute position embeddings.
- **Normalization**: **RMSNorm** is a common, cheaper alternative to `LayerNorm`.
- **Activation**: **SwiGLU** feed-forward variants outperform plain GELU MLPs at scale.
- **Attention efficiency**: **KV-caching** for fast generation, **Flash Attention** for memory-efficient
  training, **Grouped-Query Attention** to reduce inference memory.
- **Scale**: real LLMs use billions of parameters, trillions of training tokens, and distributed
  training across many GPUs/TPUs.
- **Alignment**: after pretraining (what we did here), production LLMs go through instruction-tuning
  and RLHF/DPO to become helpful, harmless assistants rather than raw text completers.

**References** : Andrej Karpathy's [`nanoGPT`](https://github.com/karpathy/nanoGPT) and the Hugging Face `transformers` library.

