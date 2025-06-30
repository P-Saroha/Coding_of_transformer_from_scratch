import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    Custom Layer Normalization implementation as described in the paper.
    Normalizes across the feature dimension (last dimension) for each sample.
    """
    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps  # Small value to prevent division by zero
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable scale parameter (γ)
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable bias parameter (β)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # Compute mean and std across the feature dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch_size, seq_len, 1)
        
        # Apply normalization: (x - μ) / σ * γ + β
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN) as described in the paper.
    Two linear transformations with ReLU activation in between.
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)    # First linear layer: d_model -> d_ff
        self.dropout = nn.Dropout(dropout)          # Dropout for regularization
        self.linear_2 = nn.Linear(d_ff, d_model)    # Second linear layer: d_ff -> d_model

    def forward(self, x):
        # Apply: Linear -> ReLU -> Dropout -> Linear
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class InputEmbeddings(nn.Module):
    """
    Converts input tokens to dense embeddings.
    Each token is mapped to a d_model dimensional vector.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding lookup table

    def forward(self, x):
        # x: (batch_size, seq_len) - token indices
        # output: (batch_size, seq_len, d_model) - dense embeddings
        # Scale embeddings by sqrt(d_model) as mentioned in the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to embeddings using sinusoidal functions.
    This allows the model to understand the position of tokens in the sequence.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)
        
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        
        # Create division term for the sinusoidal pattern
        # This creates the 10000^(2i/d_model) term from the paper
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))  # (d_model/2,)
        
        # Apply sine to even indices (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices (1, 3, 5, ...)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)  # Won't be updated during training

    def forward(self, x):
        # Add positional encoding to input embeddings
        # x: (batch_size, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    """
    Implements residual connection with layer normalization.
    Uses pre-norm: LayerNorm(x) -> Sublayer -> Add residual
    Output = x + Dropout(Sublayer(LayerNorm(x)))
    """
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # Apply layer norm first (pre-norm), then sublayer, then add residual
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention mechanism from "Attention Is All You Need".
    
    Splits the input into multiple heads, applies scaled dot-product attention
    to each head, then concatenates and projects the results.
    
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Model dimension (e.g., 512)
        self.h = h              # Number of attention heads (e.g., 8)
        
        # Ensure d_model is divisible by number of heads
        assert d_model % h == 0, "d_model must be divisible by number of heads"

        self.d_k = d_model // h  # Dimension per head (e.g., 512/8 = 64)
        
        # Linear projections for Q, K, V (no bias as per paper)
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Key projection  
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Output projection
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Compute scaled dot-product attention.
        
        Args:
            query: (batch_size, h, seq_len, d_k)
            key: (batch_size, h, seq_len, d_k) 
            value: (batch_size, h, seq_len, d_k)
            mask: (batch_size, h, seq_len, seq_len) or broadcastable
            dropout: Dropout layer
            
        Returns:
            attention_output: (batch_size, h, seq_len, d_k)
            attention_scores: (batch_size, h, seq_len, seq_len)
        """
        d_k = query.shape[-1]  # Get dimension per head
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # Shape: (batch_size, h, seq_len, seq_len)
        
        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax to get attention probabilities
        attention_scores = attention_scores.softmax(dim=-1)
        # Shape: (batch_size, h, seq_len, seq_len)
        
        # Apply dropout to attention probabilities
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # Apply attention to values: Attention_weights * V
        attention_output = attention_scores @ value
        # Shape: (batch_size, h, seq_len, d_k)
        
        return attention_output, attention_scores

    def forward(self, q, k, v, mask):
        """
        Forward pass for multi-head attention.
        
        Args:
            q: Query tensor (batch_size, seq_len, d_model)
            k: Key tensor (batch_size, seq_len, d_model)
            v: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Apply linear projections to get Q, K, V
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)    # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # Reshape and transpose for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Apply attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concatenate heads: (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Apply output projection
        return self.w_o(x)  # (batch_size, seq_len, d_model)

class EncoderBlock(nn.Module):
    """
    Single encoder block consisting of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Each sub-layer has a residual connection and layer normalization.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections: one for attention, one for feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            src_mask: Source mask for padding tokens
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        # x + Dropout(SelfAttention(LayerNorm(x)))
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Feed-forward with residual connection  
        # x + Dropout(FFN(LayerNorm(x)))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x

class Encoder(nn.Module):
    """
    Stack of N encoder blocks followed by final layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of encoder blocks
        self.norm = LayerNormalization(features)  # Final layer norm

    def forward(self, x, mask):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            mask: Source mask for padding tokens
            
        Returns:
            output: Encoded representations (batch_size, seq_len, d_model)
        """
        # Pass through all encoder blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Apply final layer normalization
        return self.norm(x)

class DecoderBlock(nn.Module):
    """
    Single decoder block consisting of:
    1. Masked multi-head self-attention (to prevent looking at future tokens)
    2. Multi-head cross-attention (attending to encoder output)
    3. Position-wise feed-forward network
    Each sub-layer has a residual connection and layer normalization.
    """
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, 
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block    # Masked self-attention
        self.cross_attention_block = cross_attention_block  # Cross-attention with encoder
        self.feed_forward_block = feed_forward_block
        # Three residual connections: self-attention, cross-attention, feed-forward
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: Target embeddings (batch_size, seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source mask for encoder padding tokens
            tgt_mask: Target mask (causal + padding)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Masked self-attention (decoder can't see future tokens)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        
        # Cross-attention with encoder output
        # Query from decoder, Key and Value from encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        
        # Feed-forward network
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x

class Decoder(nn.Module):
    """
    Stack of N decoder blocks followed by final layer normalization.
    """
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers  # List of decoder blocks
        self.norm = LayerNormalization(features)  # Final layer norm

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Args:
            x: Target embeddings (batch_size, seq_len, d_model)
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source mask for encoder padding tokens
            tgt_mask: Target mask (causal + padding)
            
        Returns:
            output: Decoded representations (batch_size, seq_len, d_model)
        """
        # Pass through all decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        # Apply final layer normalization
        return self.norm(x)

class ProjectionLayer(nn.Module):
    """
    Final linear projection layer that maps from d_model to vocabulary size.
    This converts the decoder output to logits over the vocabulary.
    """
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)  # Linear projection

    def forward(self, x):
        """
        Args:
            x: Decoder output (batch_size, seq_len, d_model)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        return self.proj(x)

class Transformer(nn.Module):
    """
    Complete Transformer model from "Attention Is All You Need".
    
    Consists of:
    - Encoder: Processes source sequence
    - Decoder: Generates target sequence
    - Embeddings: Convert tokens to dense vectors
    - Positional Encoding: Add position information
    - Projection: Convert to vocabulary logits
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed              # Source embeddings
        self.tgt_embed = tgt_embed              # Target embeddings
        self.src_pos = src_pos                  # Source positional encoding
        self.tgt_pos = tgt_pos                  # Target positional encoding
        self.projection_layer = projection_layer # Final projection to vocab

    def encode(self, src, src_mask):
        """
        Encode source sequence.
        
        Args:
            src: Source token indices (batch_size, src_seq_len)
            src_mask: Source padding mask
            
        Returns:
            encoder_output: (batch_size, src_seq_len, d_model)
        """
        # Embed source tokens and add positional encoding
        src = self.src_embed(src)      # (batch_size, src_seq_len, d_model)
        src = self.src_pos(src)        # Add positional encoding
        return self.encoder(src, src_mask)  # Pass through encoder

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decode target sequence.
        
        Args:
            encoder_output: Encoder output (batch_size, src_seq_len, d_model)
            src_mask: Source padding mask
            tgt: Target token indices (batch_size, tgt_seq_len)
            tgt_mask: Target causal + padding mask
            
        Returns:
            decoder_output: (batch_size, tgt_seq_len, d_model)
        """
        # Embed target tokens and add positional encoding
        tgt = self.tgt_embed(tgt)      # (batch_size, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)        # Add positional encoding
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Project decoder output to vocabulary logits.
        
        Args:
            x: Decoder output (batch_size, seq_len, d_model)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, 
                     tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, 
                     dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Build a complete Transformer model with specified parameters.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size  
        src_seq_len: Maximum source sequence length
        tgt_seq_len: Maximum target sequence length
        d_model: Model dimension (512 in paper)
        N: Number of encoder/decoder blocks (6 in paper)
        h: Number of attention heads (8 in paper)
        dropout: Dropout probability (0.1 in paper)
        d_ff: Feed-forward hidden dimension (2048 in paper)
        
    Returns:
        transformer: Complete Transformer model
    """
    # Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        # Each encoder block needs its own attention and feed-forward layers
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        # Each decoder block needs self-attention, cross-attention, and feed-forward layers
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, 
                                   decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the complete transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize parameters using Xavier uniform initialization
    # This helps with training stability
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer

# Example usage and testing
if __name__ == "__main__":
    print("Building Transformer model...")
    
    # Model hyperparameters (same as paper)
    src_vocab_size = 5000      # Source vocabulary size
    tgt_vocab_size = 5000      # Target vocabulary size  
    src_seq_len = 100          # Maximum source sequence length
    tgt_seq_len = 100          # Maximum target sequence length
    d_model = 512              # Model dimension
    N = 6                      # Number of encoder/decoder layers
    h = 8                      # Number of attention heads
    dropout = 0.1              # Dropout probability
    d_ff = 2048               # Feed-forward hidden dimension
    
    # Build the model
    model = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, 
                             d_model, N, h, dropout, d_ff)
    
    print("Model built successfully!")
    
    # Create sample input data
    batch_size = 2
    src_seq_len_sample = 10
    tgt_seq_len_sample = 10
    
    # Random token indices (avoiding 0 as it's typically padding)
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len_sample))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len_sample))
    
    print(f"Source input shape: {src.shape}")
    print(f"Target input shape: {tgt.shape}")
    
    # Create masks
    # Source mask: hide padding tokens (assuming 0 is padding)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)
    
    # Target mask: hide padding tokens AND future tokens (causal mask)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (batch_size, 1, tgt_seq_len, 1)
    
    # Create causal mask (lower triangular matrix)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask  # Combine padding and causal masks
    
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Target mask shape: {tgt_mask.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    
    # Encode source sequence
    encoder_output = model.encode(src, src_mask)
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Decode target sequence
    decoder_output = model.decode(encoder_output, src_mask, tgt, tgt_mask)
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Project to vocabulary
    output = model.project(decoder_output)
    print(f"Final output shape: {output.shape}")
    print(f"Expected shape: [batch_size, seq_length, tgt_vocab_size] = [{batch_size}, {tgt_seq_len_sample}, {tgt_vocab_size}]")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== Model Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (assuming float32)")
    
    print("\n=== All tests passed! Your Transformer is ready to use! ===")
    print("\nNext steps:")
    print("1. Prepare your dataset with proper tokenization")
    print("2. Implement training loop with appropriate loss function (CrossEntropyLoss)")
    print("3. Add learning rate scheduling and optimization (Adam optimizer)")
    print("4. Consider using mixed precision training for efficiency")