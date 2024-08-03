import json
import os
from datetime import datetime
import random
import re
from typing import Dict, List, Any


class BicaUtilities:
    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def save_json_file(data: Dict[str, Any], file_path: str) -> None:
        """Save data to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

    @staticmethod
    def generate_timestamp() -> str:
        """Generate a timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text by removing extra whitespace and converting to lowercase."""
        return ' '.join(text.lower().split())

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts (placeholder implementation)."""
        # This is a very basic implementation. In practice, you might want to use
        # more sophisticated methods like cosine similarity with word embeddings.
        words1 = set(BicaUtilities.normalize_text(text1).split())
        words2 = set(BicaUtilities.normalize_text(text2).split())
        return len(words1.intersection(words2)) / len(words1.union(words2))

    @staticmethod
    def generate_random_float(min_value: float, max_value: float) -> float:
        """Generate a random float between min_value and max_value."""
        return random.uniform(min_value, max_value)

    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
        """Extract key words from a given text."""
        # This is a basic implementation. You might want to use NLP libraries for better results.
        words = re.findall(r'\b\w+\b', BicaUtilities.normalize_text(text))
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]

    @staticmethod
    def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two dictionaries, with values from dict2 overwriting those from dict1 if keys clash."""
        return {**dict1, **dict2}

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        """Clamp a value between min_value and max_value."""
        return max(min(value, max_value), min_value)

    @staticmethod
    def format_list_as_string(items: List[str], conjunction: str = "and") -> str:
        """Format a list of items as a string, e.g., ['a', 'b', 'c'] -> 'a, b, and c'."""
        if len(items) == 0:
            return ""
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        else:
            return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"
