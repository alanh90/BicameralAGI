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


def generate_text(model, input_text, vocab, inv_vocab, max_length=50, temperature=0.7):
    """
    Generate text based on input prompt with improved sampling
    """
    model.eval()

    # Convert input text to indices
    input_indices = [vocab.get(char, 0) for char in input_text]
    input_tensor = torch.tensor([input_indices], dtype=torch.long)

    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            # Get model output
            output = model(input_tensor)

            # Get the last token's predictions
            last_token_logits = output[0, -1, :] / temperature

            # Filter out unlikely tokens
            top_k = 40
            top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)

            # Apply softmax to get probabilities
            probabilities = torch.softmax(top_k_logits, dim=0)

            # Sample from the filtered distribution
            next_token_idx = torch.multinomial(probabilities, 1).item()
            next_token = top_k_indices[next_token_idx].item()

            # Append the new token
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]])], dim=1)

            # Convert to character
            next_char = inv_vocab[next_token]
            input_text += next_char

            # Stop if we generate a period or newline
            if next_char in ['.', '\n']:
                break

    return input_text


def train_model(text_data):
    """Train the model with improved parameters"""
    # Create vocabulary
    vocab = {char: i for i, char in enumerate(sorted(set(text_data)))}
    inv_vocab = {i: char for char, i in vocab.items()}
    vocab_size = len(vocab)

    # Improved model parameters
    d_model = 128  # Increased from 64
    num_heads = 8  # Increased from 4
    num_layers = 3  # Increased from 2
    d_ff = 512  # Increased from 256
    max_seq_length = 100
    seq_length = 50
    batch_size = 16  # Reduced for better training

    # Create dataset and dataloader
    dataset = TextDataset(text_data, seq_length, vocab)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Added learning rate

    # Training loop with more epochs
    num_epochs = 10  # Increased from 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (input_seq, target_seq) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(input_seq)
            output = output.view(-1, vocab_size)
            target_seq = target_seq.view(-1)
            loss = criterion(output, target_seq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Added gradient clipping
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch} completed, Average Loss: {avg_loss:.4f}')

    return model, vocab, inv_vocab


def interactive_session():
    """Run an interactive session with improved training data"""
    # Much more training data with better structure
    training_data = """
    Hello! How are you today? I'm doing well, thank you for asking.
    It's a beautiful day outside. The sun is shining and the birds are singing.
    I enjoy learning about artificial intelligence and machine learning.
    Python is a powerful programming language used in many applications.
    The quick brown fox jumps over the lazy dog.
    Good morning! Would you like some coffee or tea?
    The temperature today is perfect for a walk in the park.
    I love reading books in my free time. Fiction is my favorite genre.
    The cats sleep peacefully in the warm sunshine.
    Thank you for your help! I really appreciate it.
    Could you please help me with this question?
    The meeting starts at 10 AM tomorrow morning.
    Have a great day! See you later.
    What time would you like to meet for lunch?
    The garden is full of beautiful flowers and butterflies.
    I'm working on an interesting project right now.
    The music playing in the background is very relaxing.
    Let's meet at the coffee shop on Main Street.
    The children are playing happily in the playground.
    Would you like to join us for dinner tonight?
    """

    print("Training the model... This may take a few minutes...")
    model, vocab, inv_vocab = train_model(training_data)
    print("\nModel training completed!")

    while True:
        user_input = input("\nEnter some text to complete (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        completed_text = generate_text(model, user_input, vocab, inv_vocab)
        print("\nCompleted text:")
        print(completed_text)


if __name__ == "__main__":
    interactive_session()
