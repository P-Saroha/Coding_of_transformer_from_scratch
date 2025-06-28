import torch 
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_size, vocab_size):
        super().__init__()
        self.d_size =  d_size
        self.vocab_size = vocab_size
        self.embedding  = nn.Embedding(vocab_size, d_size)

    def forward(self, x): # forward() method: This defines how data flows through those layers — the actual computation of the model.
        return self.embedding(x) * math.sqrt(self.d_size)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_size, seq_len, dropout):
        super().__init__()
        self.d_size = d_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## create a matrix of positional encodings of shape (seq_len, d_size)
        pe = torch.zeros(seq_len, d_size)

        ## create a vactor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) ## unsqueeze(1) adds a new dimension to the tensor, making it of shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_size, 2).float() * (-math.log(10000.0) / d_size))## this creates a vector of shape (d_size/2,) 
        ## calculate the positional encodings using sine and cosine functions
        pe[:, 0::2] = torch.sin(position * div_term) ## even indices    
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) ## unsqueeze(0) adds a new dimension to the tensor, making it of shape (1, seq_len, d_size)
        self.register_buffer('pe', pe) ## register_buffer() is used to register a buffer that is not a parameter, but should be part of the module's state.

        def forword(self, x):
            x = x + self.pe[:, :x.size(1)] ## add positional encodings to the input tensor x and slice it to match the input sequence length
            return self.dropout(x) 
        
class LayerNormalization(nn.Module):
    def __init__(self, d_size, eps = 10**-6):
        super().__init__()
        self.d_size = d_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_size))
        self.beta = nn.Parameter(torch.zeros(d_size))

        def forword(self, x):
            mean = x.mean(dim = -1, keepdim=True) ## calculate the mean of the input tensor x along the last dimension
            std = x.std(dim = -1, keepdim=True) ## calculate the standard deviation 
            return self.gamma * (x - mean) / (std + self.eps) + self.beta ## apply layer normalization to the input tensor x using the learned parameters gamma and beta


class FeedForward(nn.Module):
    def __init__(self,d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) ## first linear layer w1 , b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) ## second linear layer w2, b2
        self.activation = nn.ReLU() ## activation function

        
    def forword(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x)))) ## apply the feedforward network to the input tensor x, passing it through the two linear layers with dropout and activation in between


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_k = d_model // n_heads # dimension of each head 
        self.w_q = nn.Linear(d_model, d_model)  # weight matrix for queries
        self.w_k = nn.Linear(d_model, d_model)  # weight matrix for keys    
        self.w_v = nn.Linear(d_model, d_model)  # weight matrix for values
        self.w_o = nn.Linear(d_model, d_model)  # weight matrix for output 
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        query = self.w_q(q)  # shape: (batch_size, seq_len, d_model)
        key = self.w_k(k)      # shape: (batch_size, seq_len, d_model)
        value = self.w_v(v)    # shape: (batch_size, seq_len, d_model) 

        @staticmethod
        def attention(query, key, value, mask=None):
            d_k = query.size(-1)        # dimension of each head
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)    # apply mask to the scores  
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)  # apply dropout to attention weights
            return torch.matmul(attn_weights, value)
        # Calculate attention scores

        # Reshape and transpose to get (batch_size, n_heads, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k)       




