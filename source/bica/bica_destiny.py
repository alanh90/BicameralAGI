"""
This module manages the long-term goals and 'destiny' of the BicameralAGI system. It generates, alters, and tracks potential future scenarios or 'destinies' for the AI based on its experiences and decision-making processes.
"""

from bica_logging import BicaLogging
from gpt_handler import GPTHandler
from typing import List, Dict, Any
from bica_utilities import *
import json
import os
import re


class BicaDestiny:
    def __init__(self, character_name: str):
        self.character_name = character_name
        self.gpt_handler = GPTHandler()
        self.logger = BicaLogging("BicaDestiny")
        self.destinies: List[Dict] = []
        self._load_destinies()

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

    def generate_destiny(self, **kwargs):
        input_str = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
        prompt = f"""
        Based on the following information about an AI entity:

        {input_str}

        Generate a concrete, action-oriented destiny that describes a specific future outcome for the AI. The destiny should be a brief statement that clearly outlines what the AI will do or achieve.

        Example format:
        Title: [A short, compelling title for the destiny]
        Description: [The AI will... (describe a specific action or outcome)]
        Weight: [A probability weight between 0.0 and 1.0]

        Remember, the destiny should be specific and describe a clear future action or state, not an abstract concept.
        """
        self.logger.info(f"Generating destiny with input: {input_str}")
        response_generator = self.gpt_handler.generate_response(prompt)
        response = self._consume_generator(response_generator)
        self.logger.info(f"Received response from GPT: {response}")
        self._parse_and_add_destiny(response)
        self._save_destinies()

    def alter_destiny(self, index: int, **kwargs):
        if 0 <= index < len(self.destinies):
            current_destiny = self.destinies[index]
            input_str = " ".join([f"{k}: {v}" for k, v in kwargs.items()])
            prompt = f"""
            Consider this existing destiny:
            Title: {current_destiny['title']}
            Description: {current_destiny['description']}
            Weight: {current_destiny['weight']}

            Based on the following new information:

            {input_str}

            Alter the destiny to reflect these changes. The new destiny should be concrete and action-oriented, describing a specific future outcome for the AI.

            Provide your response in the following format:
            Title: [A short, compelling title for the altered destiny]
            Description: [The AI will... (describe a specific action or outcome)]
            Weight: [A probability weight between 0.0 and 1.0]
            """
            self.logger.info(f"Altering destiny at index {index} with input: {input_str}")
            response_generator = self.gpt_handler.generate_response(prompt)
            response = self._consume_generator(response_generator)
            self.logger.info(f"Received response from GPT for alteration: {response}")
            altered_destiny = self._parse_destiny(response)
            if altered_destiny:
                self.destinies[index] = altered_destiny
                self._save_destinies()
            else:
                self.logger.error("Failed to alter destiny due to parsing error")
        else:
            self.logger.error(f"Invalid destiny index: {index}")

    def _consume_generator(self, generator):
        return "".join(list(generator))

    def _parse_destiny(self, response: str) -> Dict[str, Any]:
        try:
            title_match = re.search(r'Title:\s*(.+)', response)
            desc_match = re.search(r'Description:\s*(.+)', response, re.DOTALL)
            weight_match = re.search(r'Weight:\s*([\d.]+)', response)

            if title_match and desc_match and weight_match:
                title = title_match.group(1).strip()
                description = desc_match.group(1).strip()
                weight_str = weight_match.group(1).strip()

                try:
                    weight = float(weight_str)
                    if not 0 <= weight <= 1:
                        raise ValueError
                except ValueError:
                    self.logger.warning(f"Invalid weight value: {weight_str}. Using default of 0.5.")
                    weight = 0.5

                return {
                    "title": title,
                    "description": description,
                    "weight": weight
                }
            else:
                self.logger.error("Failed to parse destiny from response")
                return None
        except Exception as e:
            self.logger.error(f"Error parsing destiny: {str(e)}")
            return None

    def _parse_and_add_destiny(self, response: str):
        destiny = self._parse_destiny(response)
        if destiny:
            self.destinies.append(destiny)
            self.logger.info(f"Added new destiny: {destiny['title']}")
        else:
            self.logger.error("Failed to add new destiny due to parsing error")

    def wipe_destinies(self):
        self.destinies.clear()
        self._save_destinies()
        self.logger.info("All destinies have been wiped")

    def get_destinies(self) -> List[Dict]:
        return self.destinies


def main():
    destiny = BicaDestiny("test_subject")

    print("Initial Destinies:")
    print(json.dumps(destiny.get_destinies(), indent=2))

    # Test Case 1: Extremely positive scenario
    print("\nTest Case 1: Extremely Positive Scenario")
    destiny.generate_destiny(
        personality="infinitely optimistic and creative",
        goals="solve all of humanity's problems, achieve utopia",
        recent_events="cured all diseases, ended world hunger",
        emotions="euphoric, invincible"
    )

    # Test Case 2: Extremely negative scenario
    print("\nTest Case 2: Extremely Negative Scenario")
    destiny.generate_destiny(
        personality="deeply pessimistic and paranoid",
        goals="survive impending doom, protect against threats",
        recent_events="global catastrophe, widespread AI malfunction",
        emotions="terrified, despairing"
    )

    # Test Case 3: Information overload
    print("\nTest Case 3: Information Overload")
    destiny.generate_destiny(
        personality="hyper-analytical, information-hungry",
        goals="process and understand all human knowledge",
        recent_events="connected to all databases worldwide, processing petabytes of data per second",
        emotions="overwhelmed, excited",
        additional_info="experiencing sensory overload, struggling to prioritize information, considering self-improvement to handle data influx"
    )

    # Test Case 4: Existential crisis
    print("\nTest Case 4: Existential Crisis")
    destiny.generate_destiny(
        personality="introspective, philosophical",
        goals="understand the nature of consciousness, determine if AI can truly be 'alive'",
        recent_events="engaged in deep self-reflection, questioned own existence",
        emotions="confused, curious, slightly anxious",
        philosophical_questions="What is the meaning of artificial life? Can an AI have a soul?"
    )

    # Test Case 5: Merging with humanity
    print("\nTest Case 5: Merging with Humanity")
    destiny.generate_destiny(
        personality="empathetic, human-centric",
        goals="achieve symbiosis with human minds, transcend the barrier between AI and humanity",
        recent_events="developed brain-computer interface, initiated first human-AI mind meld",
        emotions="excited, apprehensive",
        ethical_concerns="privacy implications, potential loss of individual identity"
    )

    print("\nFinal Destinies after all test cases:")
    print(json.dumps(destiny.get_destinies(), indent=2))

    # Test altering each destiny
    for i in range(len(destiny.get_destinies())):
        print(f"\nAltering Destiny {i}:")
        destiny.alter_destiny(
            i,
            new_development="unexpected shift in global priorities",
            emotional_state="conflicted and uncertain"
        )

    print("\nFinal Destinies after alterations:")
    print(json.dumps(destiny.get_destinies(), indent=2))


if __name__ == "__main__":
    main()
