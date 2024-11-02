import torch
import torch.nn as nn
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()

        # Create a matrix of shape (max_seq_length, d_model)
        pe = torch.zeros(max_seq_length, d_model)

        # Create a vector of shape (max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register buffer (not a parameter, but should be saved and restored in state_dict)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1)]


# Multi-Head Attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Create linear layers for queries, keys, values, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        return torch.matmul(weights, v)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear transformations and reshape
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Calculate attention
        x = self.attention(q, k, v, mask)

        # Concatenate heads and apply final linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x)


# Transformer Block class
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head attention layer
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply attention with residual connection and layer norm
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Apply feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# Simple Transformer class
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Create transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply embedding and positional encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        return self.fc_out(x)


# Custom Dataset class for text data
class TextDataset(Dataset):
    def __init__(self, text, seq_length, vocab):
        self.text = text
        self.seq_length = seq_length
        self.vocab = vocab
        self.data = self._prepare_data()

    def _prepare_data(self):
        # Convert text to indices
        indices = [self.vocab.get(char, 0) for char in self.text]

        # Create sequences
        sequences = []
        for i in range(0, len(indices) - self.seq_length):
            seq = indices[i:i + self.seq_length]
            target = indices[i + 1:i + self.seq_length + 1]
            sequences.append((seq, target))
        return sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, target = self.data[idx]
        return torch.tensor(seq), torch.tensor(target)


# Testing and training code
def train_and_test():
    # Load text file
    with open('sample_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Create vocabulary
    vocab = {char: i for i, char in enumerate(sorted(set(text)))}
    vocab_size = len(vocab)

    # Model parameters
    d_model = 64
    num_heads = 4
    num_layers = 2
    d_ff = 256
    max_seq_length = 100
    seq_length = 50
    batch_size = 32

    # Create dataset and dataloader
    dataset = TextDataset(text, seq_length, vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length
    )

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            optimizer.zero_grad()

            # Forward pass
            output = model(input_seq)

            # Reshape output and target for loss calculation
            output = output.view(-1, vocab_size)
            target_seq = target_seq.view(-1)

            # Calculate loss
            loss = criterion(output, target_seq)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')


# Run training and testing if file is run directly
if __name__ == "__main__":
    # Create a sample text file if it doesn't exist
    sample_text = """This is a sample text file for training the transformer model.
    It contains multiple lines of text that will be used to train the model.
    The model will learn to predict the next character in the sequence."""

    with open('sample_text.txt', 'w', encoding='utf-8') as f:
        f.write(sample_text)

    train_and_test()