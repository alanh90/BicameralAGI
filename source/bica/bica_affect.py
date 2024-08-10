import json
import os
import time
from typing import List, Dict, Any
from bica.gpt_handler import GPTHandler
from bica_utilities import BicaUtilities


class BicaAffect:
    def __init__(self, character_name: str):
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.character_name = character_name
        self.cog_model = self.load_cog_model()
        self.current_emotions = self.initialize_emotions()
        self.last_update = time.time()
        self.emotion_falloff_rate = 0.05

    def load_cog_model(self) -> Dict[str, Any]:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'persona_cog_models', f'{self.character_name}.json')
        file_path = os.path.normpath(file_path)  # Normalize the path for different operating systems
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            raise FileNotFoundError(f"Cognitive model for {self.character_name} not found at {file_path}.")

    def initialize_emotions(self) -> Dict[str, float]:
        return {emotion: value for emotion, value in self.cog_model['char_cogModel'][0]['attributes'].items()}

    def trigger_emotion(self, emotion: str, intensity: float):
        intensity = max(0.0, min(intensity, 1.0))  # Clamp intensity between 0 and 1
        if emotion in self.current_emotions:
            self.current_emotions[emotion] = max(0.0, min(self.current_emotions[emotion] + intensity, 1.0))
        else:
            self.current_emotions[emotion] = intensity

    def update_emotions(self):
        current_time = time.time()
        time_diff = current_time - self.last_update
        falloff_amount = time_diff * self.emotion_falloff_rate

        for emotion in self.current_emotions:
            default_value = self.cog_model['char_cogModel'][0]['attributes'].get(emotion, 0.0)
            if self.current_emotions[emotion] > default_value:
                self.current_emotions[emotion] = max(default_value, self.current_emotions[emotion] - falloff_amount)
            elif self.current_emotions[emotion] < default_value:
                self.current_emotions[emotion] = min(default_value, self.current_emotions[emotion] + falloff_amount)

        self.last_update = current_time

    def get_top_emotions(self, n: int = 3) -> List[Dict[str, Any]]:
        self.update_emotions()
        sorted_emotions = sorted(self.current_emotions.items(), key=lambda x: x[1], reverse=True)
        return [{"emotion": k, "intensity": v} for k, v in sorted_emotions[:n]]

    def get_all_emotions(self) -> Dict[str, float]:
        self.update_emotions()
        return self.current_emotions

    def create_cog_model(self, character_name: str, description: str):
        template_path = os.path.join('source', 'bica', 'data', 'template', 'persona_cog_template.json')
        with open(template_path, 'r') as file:
            template = json.load(file)

        prompt = f"Given this character description: '{description}', generate a detailed cognitive model following this template: {json.dumps(template)}. Ensure all numerical values are between 0.0 and 1.0."

        response_content = self.gpt_handler.generate_response(
            prompt=prompt,
            model="gpt-4o-2024-08-06",
            response_format={"type": "json_object"}
        )

        # Parse the response content
        cog_model = json.loads(response_content)

        # Save the generated cognitive model to the file
        file_path = os.path.join('source', 'bica', 'data', 'persona_cog_models', f'{character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

        return cog_model

    def generate_self_reflection(self) -> str:
        prompt = f"Based on the current emotional state: {self.get_all_emotions()} and personality: {self.cog_model}, generate a brief self-reflection report."
        return next(self.gpt_handler.generate_response(prompt))

    def update_personality(self, experience: str):
        prompt = f"Given the current personality: {self.cog_model} and the new experience: '{experience}', suggest minor updates to the personality traits."

        response = self.gpt_handler.generate_response(
            prompt=prompt,
            model="gpt-4o-2024-08-06",
            response_format={"type": "json_object"}
        )

        updates = response.choices[0].message.parsed

        self.cog_model.update(updates)

        file_path = os.path.join('data', 'persona_cog_models', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

    def get_personality_summary(self) -> Dict[str, Any]:
        return {
            "name": self.cog_model.get("char_name", "Unknown"),
            "description": self.cog_model.get("char_description", "No description available"),
            "dominant_traits": self.get_dominant_traits(),
            "current_emotional_state": self.get_top_emotions()
        }

    def get_dominant_traits(self) -> Dict[str, float]:
        traits = {}
        for category in self.cog_model.get("char_cogModel", []):
            if category["category"] in ["Emotions", "Cognitive Processes", "Personality Traits"]:
                traits.update(category["attributes"])
        return dict(sorted(traits.items(), key=lambda item: item[1], reverse=True)[:5])

    def generate_artificial_memories(self, num_memories: int = 5) -> List[str]:
        prompt = f"Based on this character description: '{self.cog_model['char_description']}', generate {num_memories} artificial memories that would be significant in shaping this character's personality and worldview."

        # Correct the method call to use prompt directly
        response_content = self.gpt_handler.generate_response(
            prompt=prompt,
            model="gpt-4o-2024-08-06",
            response_format={"type": "json_object"}
        )

        # Parse the response content
        response_data = json.loads(response_content)

        memories = response_data.get("memories", [])

        # Update the cognitive model with the generated memories
        self.cog_model['char_keyMemories'] = memories

        # Save the updated cognitive model back to the file
        file_path = os.path.join('source', 'bica', 'data', 'persona_cog_models', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

        return memories


# Example usage and testing
if __name__ == "__main__":
    # Initialize BicaAffect with an existing character
    affect_system = BicaAffect("alan_turing")

    # Print initial personality summary
    print("Initial Personality Summary:")
    print(json.dumps(affect_system.get_personality_summary(), indent=2))

    # Trigger some emotions
    affect_system.trigger_emotion("Joy", 0.8)
    affect_system.trigger_emotion("Curiosity", 0.9)

    # Print top emotions
    print("\nTop Emotions after triggering:")
    print(json.dumps(affect_system.get_top_emotions(), indent=2))

    # Wait for 30 seconds to see emotion falloff
    print("\nWaiting for 30 seconds to observe emotion falloff...")
    time.sleep(30)

    # Print top emotions again
    print("\nTop Emotions after 30 seconds:")
    print(json.dumps(affect_system.get_top_emotions(), indent=2))

    # Generate self-reflection
    print("\nSelf-reflection:")
    print(affect_system.generate_self_reflection())

    # Update personality based on a new experience
    affect_system.update_personality("Successfully solved a complex mathematical problem")

    # Print updated personality summary
    print("\nUpdated Personality Summary:")
    print(json.dumps(affect_system.get_personality_summary(), indent=2))

    # Generate artificial memories
    print("\nGenerated Artificial Memories:")
    memories = affect_system.generate_artificial_memories(3)
    for i, memory in enumerate(memories, 1):
        print(f"{i}. {memory}")

    # Create a new cognitive model for a different character
    new_character = "ada_lovelace"
    new_description = "Ada Lovelace, the world's first computer programmer, known for her work on Charles Babbage's Analytical Engine."
    affect_system.create_cog_model(new_character, new_description)

    print(f"\nCreated new cognitive model for {new_character}")

    # Initialize BicaAffect with the new character
    new_affect_system = BicaAffect(new_character)

    # Print new personality summary
    print("\nNew Character Personality Summary:")
    print(json.dumps(new_affect_system.get_personality_summary(), indent=2))