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
        def attention(query, key, value, mask, dropout: nn.Dropout):
            d_k = query.size(-1)        # dimension of each head
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)    # apply mask to the scores  
            if dropout is not None:
                attn_weights = self.dropout(attn_weights) # apply dropout to the attention weights
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, value)  # shape: (batch_size, seq_len, d_model)
            return output, attn_weights 
        

        # Reshape and transpose to get (batch_size, n_heads, seq_len, d_k) 
        query = query.view(query.size(0), query.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k)
        key = key.view(key.size(0), key.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k)
        value = value.view(value.size(0), value.size(1), self.n_heads, self.d_k).transpose(1, 2) # shape: (batch_size, n_heads, seq_len, d_k) 

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # apply attention to the reshaped query, key and value tensors 
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1 , self.n_heads*self.d_k) # shape: (batch_size, seq_len, d_model) 

        return self.w_o(x)  # shape: (batch_size, seq_len, d_model)    


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))  # apply residual connection and layer normalization to the input tensor x and the output of the sublayer

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual_attention = ResidualConnection(dropout)
        self.residual_feed_forward = ResidualConnection(dropout)

    def forward(self, x, mask=None):
        x = self.residual_attention(x, lambda x: self.attention(x, x, x, mask))  # apply multi-head attention with residual connection
        return self.residual_feed_forward(x, self.feed_forward)  # apply feedforward network with residual connection
    
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers): # number of layers in the encoder and other hyperparameters which define the architecture of the encoder
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])  # create a list of encoder blocks and store them in a ModuleList for easy iteration and parameter management

    def forward(self, x, mask=None): # forward() method: This defines how data flows through those layers — the actual computation of the model.
        for layer in self.layers:
            x = layer(x, mask)  # apply each encoder block to the input tensor x
        return x  # return the final output tensor after passing through all encoder blocks
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: nn.Dropout):
        super().__init__()
        self.self_attention = ResidualConnection(dropout)
        self.cross_attention = ResidualConnection(dropout)
        self.feed_forward = ResidualConnection(dropout)
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        x = self.self_attention(x, lambda x: self.self_attention_block(x, x, x, self_attention_mask))   # apply self-attention with residual connection 
        x = self.cross_attention(x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, cross_attention_mask))  # apply cross-attention with residual connection
        return self.feed_forward(x, self.feed_forward_block)        
    
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(
            MultiHeadAttention(d_model, n_heads, dropout),
            MultiHeadAttention(d_model, n_heads, dropout),
            FeedForward(d_model, d_ff, dropout),
            nn.Dropout(dropout)
        ) for _ in range(n_layers)])  # create a list of decoder blocks and store them in a ModuleList for easy iteration and parameter management

    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, self_attention_mask, cross_attention_mask)  # apply each decoder block to the input tensor x and the encoder output
        return x  # return the final output tensor after passing through all decoder blocks
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)  # linear layer to project the output of the decoder to the vocabulary size

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1 ) # apply the linear layer to the input tensor x and return the output tensor

def 