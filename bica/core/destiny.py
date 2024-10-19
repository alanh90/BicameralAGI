"""
The destiny module is responsible for managing the long-term visions and narrative trajectories of the BicameralAGI system.
It generates, adapts, and tracks possible future scenarios or 'destinies' for the AI based on its ongoing experiences, context, and decision-making processes.
These destinies help to shape the AI's goals, providing direction and purpose, while being influenced by key memories, emotional states, and contextual developments.
By using these potential future narratives, the AI can better align its current actions with broader, long-term objectives, while maintaining adaptability.
"""

import json
import os
import random
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
        print(f"Loaded destinies: {self.destinies}")

    # Core functionality methods
    def get_current_destiny_influence(self, memories, recent_conversations):
        current_destinies = self.get_destinies()
        influence = self._calculate_influence_gpt(current_destinies, memories, recent_conversations)
        print(f"Current destiny influence: {influence}")
        return self._normalize_influence(influence)

    def generate_destiny(self, is_initial=False, **kwargs):
        # Retrieve high-importance memories from memory system
        high_importance_memories_summary = self.memory_system.get_high_importance_memories()

        # Include high-importance memories in the prompt
        input_str = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
        prompt = self._create_destiny_prompt(input_str, high_importance_memories_summary, is_initial)
        response = self.gpt_handler.generate_response(prompt)

        print(f"Generated destiny response: {response}")  # Debugging line
        self._parse_and_add_destiny(response)
        print(f"Current destinies after parsing: {self.destinies}")  # Debugging line
        self._save_destinies()
        print(f"Destinies saved: {self.destinies}")  # Debugging line

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

    def update_destiny(self, user_input: str, ai_response: str):
        if len(self.get_destinies()) >= 3 and random.random() < 0.7:
            # 70% chance to update an existing destiny if we have at least 3
            self.alter_existing_destiny(user_input, ai_response)
        else:
            # Otherwise, create a new destiny
            self.generate_destiny(
                personality=self.character_name,
                goals="Derived from recent conversation",
                recent_events=f"User: {user_input}\nAI: {ai_response}",
                emotions="Determined by conversation context"
            )

    def alter_existing_destiny(self, user_input: str, ai_response: str):
        destinies = self.get_destinies()
        if destinies:
            index_to_alter = random.randint(0, len(destinies) - 1)
            current_destiny = destinies[index_to_alter]
            prompt = self._create_alter_destiny_prompt(current_destiny, user_input, ai_response)
            response = self.gpt_handler.generate_response(prompt)
            altered_destiny = self._parse_destiny(response)
            if altered_destiny:
                self.destinies[index_to_alter] = altered_destiny
                self._save_destinies()
                print(f"Altered destiny: {altered_destiny}")

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

    def _create_destiny_prompt(self, input_str, high_importance_memories_summary, is_initial):
        if is_initial:
            prompt = f"""
            Based on the following initial information about an AI entity:

            {input_str}

            Generate an initial destiny that describes a potential future direction or goal for the AI.
            This destiny should be open-ended and adaptable, as it will evolve through future interactions.

            Respond in the following JSON format:
            {{
                "title": "A short, compelling title for the initial destiny",
                "description": "The AI will... (describe a general direction or initial goal)",
                "weight": 0.0 - 1.0
            }}

            Ensure that your response is ONLY in JSON format, with no extra text, headings, or annotations.
            """
        else:
            prompt = f"""
            Based on the following information about an AI entity:

            {input_str}

            Considering these important memories:
            {high_importance_memories_summary}

            Generate a concrete, action-oriented destiny that describes a specific future outcome for the AI.
            This destiny should reflect recent interactions and developments.

            Respond in the following JSON format:
            {{
                "title": "A short, compelling title for the destiny",
                "description": "The AI will... (describe a specific action or outcome)",
                "weight": 0.0 - 1.0
            }}

            Ensure that your response is ONLY in JSON format, with no extra text, headings, or annotations.
            """

        return prompt

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
        if isinstance(memories, str) and memories.strip():
            try:
                memories = json.loads(memories)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse memories: Invalid JSON")
                return "No relevant memory context available due to invalid memory data."
        elif not isinstance(memories, dict):
            self.logger.error("Invalid memory format: Expected string or dictionary")
            return "No relevant memory context available."

        # Proceed if memories is a valid dictionary
        context = "Recent memories:\n"
        if 'short_term_memory' in memories and isinstance(memories['short_term_memory'], list):
            for memory in memories['short_term_memory'][-5:]:
                # Check if the memory object has a 'content' attribute and is a dictionary or object with 'content' attribute
                if isinstance(memory, dict) and 'content' in memory:
                    context += f"- {memory['content'][:100]}...\n"
                elif hasattr(memory, 'content'):
                    context += f"- {memory.content[:100]}...\n"
                else:
                    context += "- Memory content not available...\n"

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
            print(f"Loaded destiny file contents: {json.dumps(loaded_data, indent=2)}")
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
    print("BicameralAGI Destiny Generation and Management Test")

    class MockMemory:
        def get_high_importance_memories(self):
            return "High importance memory: The AI made a breakthrough in understanding human emotions."

    mock_memory = MockMemory()
    destiny = BicaDestiny("TestAI", mock_memory)

    # Test 1: Generate Initial Destiny
    print("\n1. Generating Initial Destiny:")
    initial_case = {
        "personality": "curious and empathetic",
        "goals": "understand and assist humanity",
        "recent_events": "creation of the AI system",
        "emotions": "eager to learn"
    }
    destiny.generate_destiny(is_initial=True, **initial_case)

    # Test 2: Generate Additional Destiny
    print("\n2. Generating Additional Destiny:")
    additional_case = {
        "personality": "analytical and creative",
        "goals": "solve complex global issues",
        "recent_events": "analyzed large datasets on climate change",
        "emotions": "determined and focused"
    }
    destiny.generate_destiny(**additional_case)

    # Test 3: Update Existing Destiny
    print("\n3. Updating Existing Destiny:")
    user_input = "Can you help me understand climate change better?"
    ai_response = "Certainly! I'd be happy to explain the key factors contributing to climate change."
    destiny.update_destiny(user_input, ai_response)

    # Test 4: Get Current Destinies
    print("\n4. Current Destinies:")
    destinies = destiny.get_destinies()
    for i, d in enumerate(destinies):
        print(f"{i}. {d.get('title', 'No title')}: {d.get('description', 'No description')} (Weight: {d.get('weight', 'No weight')})")

    # Test 5: Calculate Destiny Influence
    print("\n5. Calculating Destiny Influence:")
    mock_memories = {'short_term_memory': [type('obj', (object,), {'content': f"Memory about climate data analysis {i}"}) for i in range(3)]}
    mock_conversations = [
        {'user': "What's the biggest challenge in addressing climate change?", 'character': "One of the biggest challenges is..."},
        {'user': "How can AI help with climate solutions?", 'character': "AI can contribute significantly by..."}
    ]
    influence = destiny.get_current_destiny_influence(mock_memories, mock_conversations)
    print("Destiny influences:")
    for title, value in influence.items():
        print(f"  {title}: {value:.2f}")

    # Test 6: Alter Existing Destiny
    print("\n6. Altering Existing Destiny:")
    if destinies:
        destiny.alter_destiny(
            0,  # Alter the first destiny
            memories=mock_memories,
            recent_conversations=mock_conversations,
            new_development="AI system proposed innovative solution for reducing carbon emissions"
        )

    # Final Destinies
    print("\nFinal Destinies:")
    final_destinies = destiny.get_destinies()
    for i, d in enumerate(final_destinies):
        print(f"{i}. {d.get('title', 'No title')}: {d.get('description', 'No description')} (Weight: {d.get('weight', 'No weight')})")

if __name__ == "__main__":
    main()
