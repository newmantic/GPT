# GPT


GPT is based on the Transformer architecture, which is composed of layers of self-attention and feed-forward neural networks. The architecture allows the model to process input sequences in parallel, making it highly efficient.

The input to GPT is a sequence of tokens (words or subwords), which are represented as vectors through an embedding layer.

Given an input sequence:
X = [x_1, x_2, ..., x_n]
where x_i represents the token ID of the i-th token in the sequence.

Each token x_i is mapped to a dense vector representation:
E(x_i) = W_e * x_i
Where:
E(x_i) is the embedding of the token x_i.
W_e is the learned embedding matrix.

Since the Transformer does not inherently understand the order of tokens, positional encoding is added to the token embeddings:
PE(i, 2k) = sin(i / 10000^(2k/d_model))
PE(i, 2k+1) = cos(i / 10000^(2k/d_model))
Where:
i is the position of the token in the sequence.
k is the dimension index.
d_model is the dimensionality of the model.
The final input embedding is:
Z_i = E(x_i) + PE(i)

Self-attention allows the model to focus on different parts of the input sequence when generating the next token. The self-attention mechanism computes three vectors for each token: Query (Q), Key (K), and Value (V).

Given the input embeddings Z, the Query, Key, and Value matrices are computed as:
Q = Z * W_Q
K = Z * W_K
V = Z * W_V
Where:
W_Q, W_K, and W_V are learned weight matrices.
The attention scores are computed using the dot product of the query and key matrices, scaled by the square root of the key dimension:
Attention(Q, K, V) = softmax((Q * K^T) / sqrt(d_k)) * V
Where:
d_k is the dimension of the key vectors.
softmax is the softmax function that normalizes the attention scores.

GPT uses multi-head attention, where multiple sets of Q, K, and V matrices are computed, and attention is applied independently across each head. The results are then concatenated and projected:
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O
Where:
head_i = Attention(Q_i, K_i, V_i) for each head i.
W_O is a learned output projection matrix.

Each position's representation is passed through a position-wise feed-forward network consisting of two linear layers with a ReLU activation function in between:
FFN(Z) = max(0, Z * W_1 + b_1) * W_2 + b_2
Where:
W_1, W_2 are learned weight matrices.
b_1, b_2 are learned biases.
max(0, x) is the ReLU activation function.

Layer normalization is applied after the self-attention and feed-forward networks, with residual connections to add the input back to the output:
Z' = LayerNorm(Z + MultiHead(Q, K, V))
Output = LayerNorm(Z' + FFN(Z'))

Unlike models that have both encoder and decoder stacks, GPT is a decoder-only model. This means it generates text in an autoregressive manner, predicting the next token in the sequence based on the tokens that have come before it.

GPT generates text by predicting the next token in the sequence one at a time, using the previous tokens as context. Given the input sequence X, the probability of generating the next token x_(n+1) is given by:
P(x_(n+1) | x_1, x_2, ..., x_n) = softmax(W_o * h_n + b_o)
Where:
h_n is the hidden state at position n.
W_o and b_o are the output projection matrix and bias, respectively.

GPT is pretrained on a large corpus of text in a self-supervised manner, where it learns to predict the next token in a sequence. After pretraining, it can be fine-tuned on specific tasks by providing task-specific data and adjusting the model's parameters.
