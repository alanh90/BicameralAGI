import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from torch.nn.utils.rnn import pad_sequence
import os
from pathlib import Path

# Global variables
char_to_idx = {}
idx_to_char = {}


class ThoughtPatternConfig:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 128
        self.n_embd = 256
        self.n_head = 8
        self.n_layer = 6
        self.dropout = 0.2
        self.learning_rate = 3e-4
        self.max_iters = 5000
        self.vocab_size = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ThoughtPatternDataset(torch.utils.data.Dataset):
    def __init__(self, patterns, block_size):
        self.patterns = patterns
        self.block_size = block_size

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        # Convert pattern to tensor of indices
        input_ids = torch.tensor([char_to_idx[c] for c in pattern], dtype=torch.long)
        # Trim if too long
        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
        return input_ids


def collate_batch(batch):
    # Pad sequences to the length of the longest sequence in the batch
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    # Create targets (shifted by 1)
    targets = torch.roll(padded_batch, shifts=-1, dims=1)
    targets[:, -1] = 0  # Last target should be padding
    return padded_batch, targets


class ThoughtPatternValidator:
    def __init__(self, reference_file):
        with open(reference_file, 'r', encoding='utf-8') as f:
            self.reference = f.read()
        self.parse_reference()

    def parse_reference(self):
        # Extract valid symbols and rules from reference file
        self.core_states = re.findall(r'<[^>]+>|\[-\]|\[+\]|{@}|{~}|<!>|<\?>', self.reference)
        self.operators = ['>>>', '<<<', '==>', '^^^', 'vvv', '|||', '...']
        self.cognitive_functions = ['#', '&', '^', 'v', '*', '$', '%', '=', '→', '∑', '∆']
        self.process_markers = ['[...]', '{|}', '+++', '---', '<|>', ':::', '(?)']

    def validate_pattern(self, pattern):
        # Check if pattern follows basic rules
        if not pattern.startswith(tuple(self.core_states)):
            return False, "Pattern must start with a core state"

        if not any(pattern.endswith(state) for state in self.core_states + ['+++', '---']):
            return False, "Pattern must end with a state marker or expansion/reduction"

        # Check for valid operators between elements
        parts = pattern.split('>>>')
        if len(parts) < 2:
            return False, "Pattern must contain at least one operator"

        return True, "Valid pattern"


class TextToPatternConverter:
    def __init__(self):
        self.emotion_map = {
            'happy': '[+]',
            'sad': '[-]',
            'confused': '{~}',
            'curious': '<?>',
            'focused': '{@}',
            'neutral': '<♢>',
            'insight': '<!>'
        }

        self.activity_map = {
            'think': '{@}',
            'problem': '<?>',
            'learn': '<♢>',
            'create': '{@}',
            'decide': '<♢>',
            'analyze': '{@}',
            'understand': '<♢>'
        }

    def convert_text_to_pattern_start(self, text):
        text = text.lower()

        # First check for emotions
        for emotion, symbol in self.emotion_map.items():
            if emotion in text:
                return symbol

        # Then check for activities
        for activity, symbol in self.activity_map.items():
            if activity in text:
                return symbol

        # Default to neutral if no matches
        return '<♢>'


class ThoughtPatternGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.size()

        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding[:, :t, :]
        x = self.drop(tok_emb + pos_emb)

        # Create padding mask
        padding_mask = (idx == 0).to(self.config.device)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, src_key_padding_mask=padding_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            # Only compute loss on non-padding tokens
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            # Create mask for non-padding tokens
            non_pad_mask = (targets != 0)
            # Calculate loss only on non-padding tokens
            loss = F.cross_entropy(logits[non_pad_mask], targets[non_pad_mask])

        return logits, loss

    def generate(self, context, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context_cond = context[:, -self.config.block_size:]
            logits, _ = self(context_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

            # Stop if we generate an end token
            if next_token.item() == char_to_idx.get('>'):
                break

        return context


def load_training_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        patterns = []
        for line in f:
            line = line.strip()
            if line and not line.startswith(('//', '#')):
                patterns.append(line)
    return patterns


def train_model(config, training_data):
    model = ThoughtPatternGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    dataset = ThoughtPatternDataset(training_data, config.block_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_batch
    )

    print("Training model...")
    model.train()

    for epoch in range(config.max_iters):
        total_loss = 0
        for batch, targets in dataloader:
            batch = batch.to(config.device)
            targets = targets.to(config.device)

            logits, loss = model(batch, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

    return model


def explain_pattern(pattern):
    # Split pattern into components
    components = pattern.split('>>>')

    # Component explanations
    explanations = {
        '<♢>': 'Starting from a neutral state',
        '[-]': 'Experiencing negative emotions',
        '[+]': 'Experiencing positive emotions',
        '{@}': 'Entering deep focus',
        '{~}': 'Experiencing confusion',
        '<!>': 'Achieving insight',
        '<?>': 'Questioning',
        '#': 'Recognizing patterns',
        '&': 'Making connections',
        '^': 'Abstracting concepts',
        '*': 'Accessing memory',
        '$': 'Assessing value',
        '%': 'Estimating probability',
        '=': 'Making comparisons',
        '[...]': 'Processing',
        '+++': 'Building/expanding',
        '---': 'Reducing/simplifying',
        '<|>': 'At a decision point',
        '(?)': 'Investigating'
    }

    # Build explanation
    explanation_parts = []
    for comp in components:
        for symbol, meaning in explanations.items():
            if symbol in comp:
                explanation_parts.append(meaning)
                break
        else:
            # If no exact match found, try to interpret compound symbols
            if any(c in comp for c in '#&^v*$%=→∑∆'):
                explanation_parts.append('Cognitive processing')

    return ' → '.join(explanation_parts)


def save_model(model, config, char_to_idx, idx_to_char, save_dir='models'):
    # Create models directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)

    # Save model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char
    }, f'{save_dir}/thought_pattern_model.pt')
    print(f"Model saved to {save_dir}/thought_pattern_model.pt")


def load_model(save_dir='models'):
    model_path = f'{save_dir}/thought_pattern_model.pt'
    if not os.path.exists(model_path):
        return None, None, None, None

    # Load the saved state
    checkpoint = torch.load(model_path)

    # Reconstruct config
    config = ThoughtPatternConfig()
    for key, value in checkpoint['config'].items():
        setattr(config, key, value)

    # Create and load model
    model = ThoughtPatternGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    return model, config, checkpoint['char_to_idx'], checkpoint['idx_to_char']


def main():
    global char_to_idx, idx_to_char

    # First try to load existing model
    model, loaded_config, loaded_char_to_idx, loaded_idx_to_char = load_model()

    if model is None:
        # No saved model found, train new one
        # Initialize components
        config = ThoughtPatternConfig()
        validator = ThoughtPatternValidator('thought_pattern_reference.txt')
        converter = TextToPatternConverter()

        # Load and process training data
        print("Loading training data...")
        training_data = load_training_data('thought_pattern_data.txt')

        # Create vocabulary
        print("Creating vocabulary...")
        all_text = ''.join(training_data)
        chars = sorted(list(set(all_text)))
        char_to_idx.clear()
        idx_to_char.clear()
        char_to_idx.update({ch: i for i, ch in enumerate(chars)})
        idx_to_char.update({i: ch for ch, i in char_to_idx.items()})
        config.vocab_size = len(chars)

        print(f"Vocabulary size: {config.vocab_size}")
        print(f"Number of training patterns: {len(training_data)}")

        # Train model
        model = train_model(config, training_data)

        # Save the trained model
        save_model(model, config, char_to_idx, idx_to_char)
    else:
        print("Loaded existing model from disk")
        config = loaded_config
        char_to_idx.clear()
        idx_to_char.clear()
        char_to_idx.update(loaded_char_to_idx)
        idx_to_char.update(loaded_idx_to_char)

    validator = ThoughtPatternValidator('thought_pattern_reference.txt')
    converter = TextToPatternConverter()

    print("\nThought Pattern Generator")
    print("------------------------")
    print("1. Enter a thought pattern symbol (e.g., '<♢>', '[-]', '<?>')")
    print("2. Or enter a text description of a thought/situation")
    print("Enter 'quit' to exit\n")

    # [Rest of the main function remains the same...]

    while True:
        user_input = input("Enter input: ").strip()
        if user_input.lower() == 'quit':
            break

        try:
            # Determine if input is a symbol or text
            if user_input.startswith(('<', '[', '{')):
                start_pattern = user_input
            else:
                start_pattern = converter.convert_text_to_pattern_start(user_input)
                print(f"Converted to starting pattern: {start_pattern}")

            # Generate pattern
            context = torch.tensor([[char_to_idx[c] for c in start_pattern]], dtype=torch.long).to(config.device)
            generated = model.generate(context, max_new_tokens=50, temperature=0.8)
            pattern = ''.join([idx_to_char[int(i)] for i in generated[0]])

            # Validate pattern
            is_valid, message = validator.validate_pattern(pattern)

            print("\nGenerated Pattern:", pattern)
            print("Validation:", message)
            print("Pattern Explanation:", explain_pattern(pattern))

        except Exception as e:
            print(f"Error: {e}")
            print("Try another input or check if the symbol is valid.")


if __name__ == "__main__":
    main()
