import json
import os
from datetime import datetime
import random
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

# Explicitly load the .env file located in the parent directory of the project
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path)


class BicaUtilities:
    @staticmethod
    def get_environment_variable(var_name: str) -> str:
        """Get an environment variable value."""
        var_value = os.getenv(var_name)
        if var_value is None:
            raise EnvironmentError(f"{var_name} not found in environment. Ensure the .env file is set up correctly.")
        return var_value

    @staticmethod
    def read_file(path: str) -> List[str]:
        """Read the contents of a file into a list of lines."""
        with open(path, 'r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]

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

    @staticmethod
    def normalize_weights(items: List[Dict[str, Any]], weight_key: str = "weight") -> List[Dict[str, Any]]:
        """Normalize the weights of a list of dictionaries."""
        total_weight = sum(item[weight_key] for item in items)
        if total_weight > 0:
            for item in items:
                item[weight_key] /= total_weight
        return items


def main():
    # Example of loading and saving a JSON file
    test_data = {"name": "Test", "value": 123}
    test_file = "test.json"
    BicaUtilities.save_json_file(test_data, test_file)
    print(f"Saved JSON: {test_data}")

    loaded_data = BicaUtilities.load_json_file(test_file)
    print(f"Loaded JSON: {loaded_data}")
    os.remove(test_file)

    # Generate and print a timestamp
    timestamp = BicaUtilities.generate_timestamp()
    print(f"Generated timestamp: {timestamp}")

    # Normalize text
    text = "   Hello    WORLD  "
    normalized_text = BicaUtilities.normalize_text(text)
    print(f"Normalized text: '{normalized_text}'")

    # Calculate similarity
    similarity = BicaUtilities.calculate_similarity("Hello World", "Hello")
    print(f"Similarity: {similarity}")

    # Generate a random float
    random_float = BicaUtilities.generate_random_float(1.0, 2.0)
    print(f"Random float between 1.0 and 2.0: {random_float}")

    # Extract keywords from text
    text = "This is a test sentence for keyword extraction. Keyword extraction is useful."
    keywords = BicaUtilities.extract_keywords(text, max_keywords=3)
    print(f"Extracted keywords: {keywords}")

    # Merge two dictionaries
    dict1 = {"a": 1, "b": 2}
    dict2 = {"b": 3, "c": 4}
    merged_dict = BicaUtilities.merge_dicts(dict1, dict2)
    print(f"Merged dictionary: {merged_dict}")

    # Clamp a value
    clamped_value = BicaUtilities.clamp(5.0, 1.0, 10.0)
    print(f"Clamped value (5.0 clamped between 1.0 and 10.0): {clamped_value}")

    clamped_value = BicaUtilities.clamp(-1.0, 1.0, 10.0)
    print(f"Clamped value (-1.0 clamped between 1.0 and 10.0): {clamped_value}")

    clamped_value = BicaUtilities.clamp(11.0, 1.0, 10.0)
    print(f"Clamped value (11.0 clamped between 1.0 and 10.0): {clamped_value}")

    # Format a list as a string
    items = ["apple", "banana", "cherry"]
    formatted_string = BicaUtilities.format_list_as_string(items)
    print(f"Formatted list: {formatted_string}")

    items = ["apple", "banana"]
    formatted_string = BicaUtilities.format_list_as_string(items)
    print(f"Formatted list: {formatted_string}")

    items = ["apple"]
    formatted_string = BicaUtilities.format_list_as_string(items)
    print(f"Formatted list: {formatted_string}")

    # Test environment variable retrieval
    try:
        api_key = BicaUtilities.get_environment_variable('OPENAI_API_KEY')
        print(f"Retrieved API_KEY: {api_key}")
    except EnvironmentError as e:
        print(e)

    # Test reading a file
    test_file = "test_read_file.txt"
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")
    file_contents = BicaUtilities.read_file(test_file)
    print(f"Read file contents: {file_contents}")
    os.remove(test_file)


if __name__ == "__main__":
    main()
