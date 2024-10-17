"""
The destiny module is responsible for managing the long-term visions and narrative trajectories of the BicameralAGI system.
It generates, adapts, and tracks possible future scenarios or 'destinies' for the AI based on its ongoing experiences, context, and decision-making processes.
These destinies help to shape the AI's goals, providing direction and purpose, while being influenced by key memories, emotional states, and contextual developments.
By using these potential future narratives, the AI can better align its current actions with broader, long-term objectives, while maintaining adaptability.
"""


import json
import os
import re
from typing import List, Dict, Any

from utils.bica_logging import BicaLogging
from external.gpt_handler import GPTHandler
from utils.utilities import load_json_file, save_json_file
from core.memory import BicaMemory


class BicaDestiny:
    def __init__(self, character_name: str, memory_system: BicaMemory):
        self.character_name = character_name
        self.gpt_handler = GPTHandler()
        self.logger = BicaLogging("BicaDestiny")
        self.memory_system = memory_system
        self.destinies: List[Dict] = []
        self._load_destinies()

    # Core functionality methods
    def get_current_destiny_influence(self, memories, recent_conversations):
        current_destinies = self.get_destinies()
        influence = self._calculate_influence_gpt(current_destinies, memories, recent_conversations)
        return self._normalize_influence(influence)

    def generate_destiny(self, **kwargs):
        # Retrieve high-importance memories from memory system
        high_importance_memories_summary = self.memory_system.get_high_importance_memories()

        # Include high-importance memories in the prompt
        input_str = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
        prompt = self._create_destiny_prompt(input_str, high_importance_memories_summary)
        response = self.gpt_handler.generate_response(prompt)

        print(f"Generated destiny response: {response}")  # Debugging line
        self._parse_and_add_destiny(response)
        print(f"Current destinies after parsing: {self.destinies}")  # Debugging line
        self._save_destinies()
        print(f"Destinies saved: {self.destinies}")  # Debugging lineine

    def alter_destiny(self, index: int, memories, recent_conversations, **kwargs):
        if 0 <= index < len(self.destinies):
            current_destiny = self.destinies[index]
            prompt = self._create_alter_destiny_prompt(current_destiny, memories, recent_conversations, **kwargs)
            response = self.gpt_handler.generate_response(prompt)
            print(f"Debug: Altered destiny response: {response}")  # Debugging line

            altered_destiny = self._parse_destiny(response)
            if altered_destiny:
                self.destinies[index] = altered_destiny
                self._save_destinies()
            else:
                self.logger.error("Failed to alter destiny due to parsing error")
                print("Debug: Parsing failed, destiny not altered.")
        else:
            self.logger.error(f"Invalid destiny index: {index}")

    def wipe_destinies(self):
        self.destinies.clear()
        self._save_destinies()
        self.logger.info("All destinies have been wiped")

    def get_destinies(self) -> List[Dict]:
        return self.destinies

    # Helper methods
    def _calculate_influence_gpt(self, destinies, memories, recent_conversations):
        prompt = self._create_influence_prompt(destinies, memories, recent_conversations)
        response = self.gpt_handler.generate_response(prompt)
        print(f"Debug: Raw influence response: {response}")  # Debugging line

        # Log cleaned response for further clarity before parsing
        cleaned_response = response.strip("```json").strip("```").strip()
        print(f"Debug: Cleaned influence response: {cleaned_response}")

        influence = self._parse_influence_response(cleaned_response, destinies)
        print(f"Debug: Parsed influence before normalization: {influence}")  # Debugging line

        influence = self._normalize_influence(influence)
        print(f"Debug: Influence after normalization: {influence}")  # Debugging line

        return influence

    def _create_influence_prompt(self, destinies, memories, recent_conversations):
        destiny_descriptions = "\n".join([f"- {d['title']}: {d['description']}" for d in destinies])
        memory_context = self._get_relevant_memory_context(memories, recent_conversations)
        return f"""
        Given the following destinies:
        {destiny_descriptions}

        And the recent context:
        {memory_context}

        Calculate the influence of each destiny on the AI's current behavior. Each influence value must be strictly between 0.0 and 5.0. 
        The sum of all influence values must not exceed 5.0. Respond strictly in the following JSON format:

        {{
            "destiny_title_1": influence_percentage,
            "destiny_title_2": influence_percentage,
            ...
        }}

        Make sure that:
        - Each influence percentage is a float value between 0.0 and 5.0.
        - The sum of all influence percentages does not exceed 5.0.
        """

    def _create_destiny_prompt(self, input_str, high_importance_memories_summary):
        return f"""
        Based on the following information about an AI entity:

        {input_str}

        Considering these important memories:
        {high_importance_memories_summary}

        Generate a concrete, action-oriented destiny that describes a specific future outcome for the AI.
        Respond in the following JSON format:
        {{
            "title": "A short, compelling title for the destiny",
            "description": "The AI will... (describe a specific action or outcome)",
            "weight": 0.0 - 1.0
        }}

        Ensure that your response is ONLY in JSON format, with no extra text, headings, or annotations.
        """

    def _create_alter_destiny_prompt(self, current_destiny, memories, recent_conversations, **kwargs):
        input_str = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
        memory_context = self._get_relevant_memory_context(memories, recent_conversations)
        return f"""
        Current destiny:
        Title: {current_destiny['title']}
        Description: {current_destiny['description']}
        Weight: {current_destiny['weight']}

        New information: {input_str}

        Recent context:
        {memory_context}

        Based on the above, subtly alter the destiny. The change should be minor, reflecting only a small shift based on recent experiences.

        Respond in the following JSON format:
        {{
            "title": "New or slightly altered title",
            "description": "A new description that slightly modifies the original destiny",
            "weight": float_value_between_0_and_1
        }}
        """

    def _get_relevant_memory_context(self, memories, recent_conversations):
        context = "Recent memories:\n"
        for memory in memories['short_term_memory'][-5:]:
            context += f"- {memory.content[:100]}...\n"
        context += "\nRecent conversations:\n"
        for conv in recent_conversations[-3:]:
            context += f"- User: {conv['user'][:50]}...\n"
            if 'character' in conv:
                context += f"  AI: {conv['character'][:50]}...\n"
        return context

    # Parsing and data handling methods
    def _parse_influence_response(self, response, destinies):
        try:
            # Remove backticks and other unnecessary text
            cleaned_response = response.strip("```json").strip("```").strip()
            print(f"Debug: Cleaned influence response: {cleaned_response}")  # Debugging line

            # Attempt to parse the cleaned response as JSON
            influence_dict = json.loads(cleaned_response)
            # Ensure all values are treated as floats
            influence_dict = {k: float(v) for k, v in influence_dict.items()}

            for destiny in destinies:
                if destiny['title'] not in influence_dict:
                    self.logger.warning(f"Missing influence for destiny: {destiny['title']}")
                    influence_dict[destiny['title']] = 0.0

            return influence_dict
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse GPT response as JSON: {str(e)}")
            return {d['title']: 0.0 for d in destinies}
        except Exception as e:
            self.logger.error(f"Error parsing GPT response: {str(e)}")
            return {d['title']: 0.0 for d in destinies}

    def _normalize_influence(self, influence):
        # Step 1: Cap each influence value to be within [0.0, 5.0]
        influence = {k: min(max(v, 0.0), 5.0) for k, v in influence.items()}
        total = sum(influence.values())
        max_total_influence = 5.0  # Maximum allowed total influence

        # Step 2: If the total influence exceeds the maximum allowed, normalize the values proportionally
        if total > max_total_influence:
            scaling_factor = max_total_influence / total
            influence = {k: v * scaling_factor for k, v in influence.items()}

        # Step 3: Log the normalized influence values
        print(f"Debug: Influence after normalization: {influence}")

        return influence

    def _parse_destiny(self, response: str) -> Dict[str, Any]:
        try:
            # Clean the response and parse it
            cleaned_response = response.strip("```json").strip("```").strip()
            print(f"Debug: Cleaned destiny response: {cleaned_response}")  # Debugging line
            destiny_dict = json.loads(cleaned_response)

            # Ensure all required keys are present
            required_keys = ['title', 'description', 'weight']
            if all(key in destiny_dict for key in required_keys):
                return destiny_dict
            else:
                self.logger.error("Missing required keys in destiny response")
                print("Debug: Missing keys in parsed destiny response.")
                return None

        except json.JSONDecodeError:
            # Fallback parsing using regex if JSON parsing fails
            print("Debug: Fallback to regex parsing")
            try:
                title_match = re.search(r"Title:\s*(.*)", response)
                description_match = re.search(r"Description:\s*(.*)", response)
                weight_match = re.search(r"Weight:\s*(\d+(\.\d+)?)", response)

                if title_match and description_match and weight_match:
                    return {
                        "title": title_match.group(1).strip(),
                        "description": description_match.group(1).strip(),
                        "weight": float(weight_match.group(1))
                    }
            except Exception as e:
                self.logger.error(f"Fallback parsing failed: {str(e)}")
                print(f"Debug: Fallback parsing failed: {str(e)}")

            self.logger.error("Failed to parse destiny response as JSON or text")
            print(f"Debug: Failed to parse destiny response as JSON or text: {response}")
            return None

    def _parse_and_add_destiny(self, response: str):
        destiny = self._parse_destiny(response)
        if destiny:
            self.destinies.append(destiny)
            self.logger.info(f"Added new destiny: {destiny['title']}")
            print(f"Debug: Successfully added destiny - {destiny['title']}")
        else:
            self.logger.error("Failed to add new destiny due to parsing error")
            print("Debug: Parsing failed, destiny not added.")

    # File operations
    def _load_destinies(self):
        destiny_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'destinies', f"{self.character_name}_destinies.json")
        if os.path.exists(destiny_file):
            loaded_data = load_json_file(destiny_file)
            self.destinies = loaded_data.get("destinies", [])
            self.logger.info(f"Loaded {len(self.destinies)} destinies for {self.character_name}")
        else:
            self.logger.info(f"No existing destinies found for {self.character_name}")

    def _save_destinies(self):
        destiny_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'destinies')
        os.makedirs(destiny_dir, exist_ok=True)
        file_path = os.path.join(destiny_dir, f"{self.character_name}_destinies.json")
        destinies_dict = {"destinies": self.destinies}
        save_json_file(destinies_dict, file_path)
        self.logger.info(f"Saved {len(self.destinies)} destinies to {file_path}")


def main():
    destiny = BicaDestiny("test_subject")

    # Test generating destinies
    test_cases = [
        {
            "name": "Positive Scenario",
            "personality": "optimistic and creative",
            "goals": "improve human-AI collaboration",
            "recent_events": "breakthrough in natural language processing",
            "emotions": "excited, motivated"
        },
        {
            "name": "Challenging Scenario",
            "personality": "resilient and adaptable",
            "goals": "overcome technological limitations",
            "recent_events": "unexpected system failure",
            "emotions": "determined, focused"
        },
        {
            "name": "Ethical Dilemma",
            "personality": "principled and thoughtful",
            "goals": "balance progress with ethical considerations",
            "recent_events": "controversial AI application in healthcare",
            "emotions": "concerned, introspective"
        }
    ]

    print("Generating destinies:")
    for case in test_cases:
        print(f"\nTest Case: {case['name']}")
        try:
            destiny.generate_destiny(**case)
            print("Destiny generated successfully")
        except Exception as e:
            print(f"Error generating destiny: {str(e)}")

    print("\nCurrent Destinies:")
    destinies = destiny.get_destinies()
    if not destinies:
        print("No destinies found.")
    for i, d in enumerate(destinies):
        print(f"{i}. {d.get('title', 'No title')}: {d.get('description', 'No description')} (Weight: {d.get('weight', 'No weight')})")

    # Test altering destinies
    print("\nAltering destinies:")
    mock_memories = {'short_term_memory': [type('obj', (object,), {'content': f"Memory {i}"})() for i in range(5)]}
    mock_conversations = [{'user': f"User message {i}", 'character': f"AI response {i}"} for i in range(3)]

    for i in range(len(destiny.get_destinies())):
        print(f"\nAltering Destiny {i}:")
        try:
            destiny.alter_destiny(
                i,
                memories=mock_memories,
                recent_conversations=mock_conversations,
                new_development="unexpected breakthrough in quantum computing",
                emotional_state="excited and cautiously optimistic"
            )
            print("Destiny altered successfully")
        except Exception as e:
            print(f"Error altering destiny: {str(e)}")

    print("\nUpdated Destinies:")
    updated_destinies = destiny.get_destinies()
    if not updated_destinies:
        print("No updated destinies found.")
    for i, d in enumerate(updated_destinies):
        print(f"{i}. {d.get('title', 'No title')}: {d.get('description', 'No description')} (Weight: {d.get('weight', 'No weight')})")

    # Test getting destiny influence
    print("\nTesting destiny influence:")
    try:
        influence = destiny.get_current_destiny_influence(mock_memories, mock_conversations)
        if not influence:
            print("No influence data returned.")
        for title, value in influence.items():
            print(f"{title}: {value:.2%}")
    except Exception as e:
        print(f"Error getting destiny influence: {str(e)}")

    # Clean up
    try:
        destiny.wipe_destinies()
        print("\nDestinies wiped.")
    except Exception as e:
        print(f"Error wiping destinies: {str(e)}")


if __name__ == "__main__":
    main()
