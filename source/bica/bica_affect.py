import json
import os
import random
import time
from typing import List, Dict, Any
from bica.gpt_handler import GPTHandler
from bica_utilities import BicaUtilities
from bica_logging import BicaLogging


class BicaAffect:
    def __init__(self, character_name: str):
        self.negative_emotions = None
        self.positive_emotions = None
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.logger = BicaLogging("BicaAffect")
        self.character_name = character_name
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.cog_model = self.load_or_create_cog_model()
        self.runtime_emotions = self.initialize_emotions()
        self.emotion_falloff_rate = 0.05
        self.last_update = time.time()
        self.emotion_decay_time = 90  # 90 seconds for emotion decay
        self.personality_summary = ""
        self.emotional_stability = self.calculate_emotional_stability()
        self.emotional_intelligence = random.uniform(0.3, 0.8)  # Initialize with a random value

    def calculate_emotional_stability(self):
        try:
            conscientiousness = self.cog_model['char_cogModel'][1]['attributes'].get('Conscientiousness', 0.5)
            neuroticism = self.cog_model['char_cogModel'][1]['attributes'].get('Neuroticism', 0.5)
            return (conscientiousness + (1 - neuroticism)) / 2
        except (IndexError, KeyError):
            self.logger.warning("Failed to calculate emotional stability. Using default value.")
            return 0.5

    def load_or_create_cog_model(self) -> Dict[str, Any]:
        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')

        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    model = json.load(file)
                return model
            else:
                raise FileNotFoundError
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.info(f"Cognitive model not found or invalid. Creating a new one for {self.character_name}.")
            return self.create_cog_model(self.character_name, f"Default cognitive model for {self.character_name}")

    def initialize_emotions(self) -> Dict[str, float]:
        try:
            emotions = self.cog_model.get('char_cogModel', [{}])[0].get('attributes', {})
            return {emotion: value for emotion, value in emotions.items() if isinstance(value, (int, float))}
        except IndexError:
            self.logger.warning("Failed to initialize emotions. Using default emotions.")
            return {"Joy": 0.5, "Sadness": 0.5, "Anger": 0.5, "Fear": 0.5, "Surprise": 0.5}

    def trigger_emotion(self, emotion: str, intensity: float, stimulate: bool = False):
        intensity = max(0.0, min(intensity, 1.0))
        if emotion in self.runtime_emotions:
            if stimulate:
                self.runtime_emotions[emotion] = max(0.0, min(self.runtime_emotions[emotion] + intensity, 1.0))
            else:
                self.runtime_emotions[emotion] = intensity
        else:
            self.logger.warning(f"Triggering unknown emotion: {emotion}")
            self.runtime_emotions[emotion] = intensity
        self.logger.info(f"{'Stimulated' if stimulate else 'Set'} emotion {emotion} with intensity {intensity}")

    def update_emotions(self):
        current_time = time.time()
        time_diff = current_time - self.last_update
        decay_factor = time_diff / max(self.emotion_decay_time, 0.1)  # Avoid division by zero

        for emotion in list(self.runtime_emotions.keys()):
            default_value = self.cog_model['char_cogModel'][0]['attributes'].get(emotion, 0.5)
            current_value = self.runtime_emotions[emotion]

            if current_value > default_value:
                decay_amount = decay_factor * (1 - abs(current_value - default_value))
                self.runtime_emotions[emotion] = max(default_value, current_value - decay_amount)
            elif current_value < default_value:
                self.runtime_emotions[emotion] = min(default_value, current_value + decay_factor)

        self.last_update = current_time

    def get_top_emotions(self, n: int = 3) -> List[Dict[str, Any]]:
        self.update_emotions()
        sorted_emotions = sorted(self.runtime_emotions.items(), key=lambda x: x[1], reverse=True)
        return [{"emotion": k, "intensity": v} for k, v in sorted_emotions[:min(n, len(sorted_emotions))]]

    def get_all_emotions(self):
        self.update_emotions()
        return self.runtime_emotions

    def categorize_emotions(self):
        # Load these from a configuration file in a real-world scenario
        positive_emotions = ['Joy', 'Excitement', 'Contentment']
        negative_emotions = ['Sadness', 'Anger', 'Fear']
        self.positive_emotions = {e: v for e, v in self.runtime_emotions.items() if e in positive_emotions}
        self.negative_emotions = {e: v for e, v in self.runtime_emotions.items() if e in negative_emotions}

    def calculate_mood(self):
        self.categorize_emotions()
        positive_score = sum(self.positive_emotions.values())
        negative_score = sum(self.negative_emotions.values())
        total_emotions = len(self.runtime_emotions)

        if total_emotions == 0:
            self.logger.warning("No emotions present. Returning neutral mood.")
            return 0.5

        base_mood = (positive_score - negative_score) / total_emotions
        optimism = self.cog_model['char_cogModel'][1]['attributes'].get('Optimism', 0.5)
        return (base_mood + optimism) / 2

    def simulate_emotional_contagion(self, external_emotion: str, intensity: float):
        contagion_factor = max(0.1, min(random.uniform(0.1, 0.5), 1.0))  # Ensure factor is between 0.1 and 1.0
        self.trigger_emotion(external_emotion, intensity * contagion_factor)

    def create_cog_model(self, character_name: str, description: str):
        template_path = os.path.join(self.base_path, 'data', 'template', 'persona_cog_template.json')
        try:
            with open(template_path, 'r') as file:
                template = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.error(f"Failed to load cognitive model template from {template_path}")
            return self._create_default_cog_model(character_name, description)

        prompt = f"Given this character description: '{description}', generate a detailed cognitive model following this template: {json.dumps(template)}. Ensure all numerical values are between 0.0 and 1.0."

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with creating cognitive models for characters."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.gpt_handler.generate_response(
                messages=messages,
                functions=[{
                    "name": "create_cognitive_model",
                    "description": "Create a cognitive model based on the template",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "char_name": {"type": "string"},
                            "char_description": {"type": "string"},
                            "char_cogModel": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "category": {"type": "string"},
                                        "attributes": {
                                            "type": "object",
                                            "additionalProperties": {"type": "number"}
                                        }
                                    }
                                }
                            },
                            "char_keyMemories": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["char_name", "char_description", "char_cogModel"]
                    }
                }],
                function_call={"name": "create_cognitive_model"}
            )

            if isinstance(response, dict) and "function_call" in response:
                cog_model = json.loads(response["function_call"]["arguments"])
            else:
                raise ValueError("Unexpected response format from GPT")

        except Exception as e:
            self.logger.error(f"Failed to generate cognitive model: {str(e)}")
            return self._create_default_cog_model(character_name, description)

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{character_name}.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

        self.logger.info(f"Created cognitive model for {character_name}")
        return cog_model

    def update_personality(self, experience: str):
        print(f"Updating personality based on experience: {experience}")
        prompt = f"Given the current personality: {self.cog_model['char_cogModel']} and the new experience: '{experience}', suggest minor updates to the personality traits. Provide numerical values between 0 and 1 for each trait."

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with updating personality traits based on new experiences."},
            {"role": "user", "content": prompt}
        ]

        print("Sending request to GPT model")
        response = self.gpt_handler.generate_response(messages)

        print(f"Raw response from GPT: {response}")

        updates = self._parse_trait_updates(response)

        if not updates:
            print("No traits returned, generating traits from experience")
            self.logger.warning("No traits returned, generating traits from experience")
            updates = self._generate_traits_from_experience(experience)

        print(f"Final updates: {updates}")

        for category in self.cog_model["char_cogModel"]:
            if category["category"] == "Personality Traits":
                for trait, value in updates.items():
                    if trait in category["attributes"]:
                        category["attributes"][trait] = (category["attributes"][trait] + value) / 2
                    else:
                        category["attributes"][trait] = value
                break
        else:
            print("No 'Personality Traits' category found. Creating a new one.")
            self.cog_model["char_cogModel"].append({
                "category": "Personality Traits",
                "attributes": updates
            })

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

        print(f"Updated cognitive model: {self.cog_model}")
        self.logger.info(f"Updated personality traits for {self.character_name}")

    def _apply_personality_updates(self, updates):
        for category in self.cog_model["char_cogModel"]:
            if category["category"] == "Personality Traits":
                for trait, value in updates.items():
                    if trait in category["attributes"]:
                        category["attributes"][trait] = (category["attributes"][trait] + value) / 2
                    else:
                        category["attributes"][trait] = value
                break
        else:
            self.logger.info("No 'Personality Traits' category found. Creating a new one.")
            self.cog_model["char_cogModel"].append({
                "category": "Personality Traits",
                "attributes": updates
            })

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

        self.logger.info(f"Updated personality traits for {self.character_name}")


    def update_personality_from_emotion(self):
        for emotion, intensity in self.runtime_emotions.items():
            if intensity > 0.7:
                self.alter_traits_from_emotion(emotion, intensity * 0.1)


    def alter_traits_from_emotion(self, emotion: str, intensity: float):
        # This should ideally be loaded from a configuration file
        trait_changes = {
            'Joy': {'Optimism': 0.05, 'Extraversion': 0.03},
            'Sadness': {'Neuroticism': 0.05, 'Optimism': -0.03},
            'Anger': {'Agreeableness': -0.05, 'Neuroticism': 0.03},
            'Fear': {'Neuroticism': 0.05, 'Openness': -0.03}
        }

        if emotion in trait_changes:
            for trait, change in trait_changes[emotion].items():
                for category in self.cog_model['char_cogModel']:
                    if category['category'] == 'Personality Traits':
                        current_value = category['attributes'].get(trait, 0.5)
                        new_value = max(0, min(1, current_value + change * intensity))
                        category['attributes'][trait] = new_value
                        break

    def _parse_trait_updates(self, response: str) -> Dict[str, float]:
        updates = {}
        lines = response.split('\n')
        for line in lines:
            parts = line.split(':')
            if len(parts) == 2:
                trait = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    if 0 <= value <= 1:
                        updates[trait] = value
                except ValueError:
                    self.logger.warning(f"Invalid trait value: {parts[1].strip()}")
        return updates

    def alter_traits_from_experience(self, experience: str, intensity: float) -> Dict[str, float]:
        intensity = max(0, min(intensity, 1))  # Ensure intensity is between 0 and 1
        prompt = f"Given the experience: '{experience}' with an intensity of {intensity} (0-1 scale), suggest updates to 5 personality traits. Format: Trait: value"

        try:
            response = self.gpt_handler.generate_response([{"role": "user", "content": prompt}])
            trait_updates = self._parse_trait_updates(response)

            # Apply intensity to trait changes
            for trait in trait_updates:
                trait_updates[trait] *= intensity

            return trait_updates
        except Exception as e:
            self.logger.error(f"Failed to alter traits from experience: {str(e)}")
            return {}

    def generate_personality_summary(self):
        prompt = f"Summarize the personality of a character with these traits: {self.cog_model['char_cogModel']} and current emotions: {self.get_top_emotions()}. Provide a concise paragraph."

        messages = [
            {"role": "system", "content": "You are an AI assistant that summarizes character personalities."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.gpt_handler.generate_response(messages)
            self.personality_summary = response.strip()
        except Exception as e:
            self.logger.error(f"Failed to generate personality summary: {str(e)}")
            self.personality_summary = "Unable to generate personality summary."


    def get_personality_summary(self) -> str:
        if not self.personality_summary:
            self.generate_personality_summary()
        return self.personality_summary

    def generate_artificial_memories(self, num_memories: int = 5) -> List[str]:
        prompt = f"Based on this character description: '{self.cog_model['char_description']}', generate {num_memories} detailed, vivid memories that shaped this character. Each memory should be a paragraph long."

        messages = [
            {"role": "system", "content": "You are an AI that creates artificial memories for characters based on their background."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.gpt_handler.generate_response(messages)
            memories = [memory.strip() for memory in response.split('\n\n') if memory.strip()]

            self.cog_model['char_keyMemories'] = memories[:num_memories]

            file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

            self.logger.info(f"Generated {len(memories)} artificial memories for {self.character_name}")
            return memories[:num_memories]
        except Exception as e:
            self.logger.error(f"Failed to generate artificial memories: {str(e)}")
            return []


# Comprehensive testing
if __name__ == "__main__":
    def run_test(test_func):
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")

    def test_initialization():
        affect = BicaAffect("TestCharacter")
        assert affect.character_name == "TestCharacter"
        assert len(affect.runtime_emotions) > 0
        assert 0 <= affect.emotional_stability <= 1
        assert 0.3 <= affect.emotional_intelligence <= 0.8

    def test_load_or_create_cog_model():
        affect = BicaAffect("TestCharacter")
        assert affect.cog_model is not None
        assert "char_name" in affect.cog_model
        assert "char_cogModel" in affect.cog_model

    def test_initialize_emotions():
        affect = BicaAffect("TestCharacter")
        assert len(affect.runtime_emotions) > 0
        for emotion, intensity in affect.runtime_emotions.items():
            assert 0 <= intensity <= 1

    def test_trigger_emotion():
        affect = BicaAffect("TestCharacter")
        affect.trigger_emotion("Joy", 0.8)
        assert "Joy" in affect.runtime_emotions
        assert affect.runtime_emotions["Joy"] == 0.8
        affect.trigger_emotion("Anger", 1.5)  # Should be clamped to 1.0
        assert affect.runtime_emotions["Anger"] == 1.0

    def test_update_emotions():
        affect = BicaAffect("TestCharacter")
        affect.trigger_emotion("Sadness", 1.0)
        initial_sadness = affect.runtime_emotions["Sadness"]
        time.sleep(2)  # Wait for 2 seconds
        affect.update_emotions()
        assert affect.runtime_emotions["Sadness"] < initial_sadness

    def test_get_top_emotions():
        affect = BicaAffect("TestCharacter")
        affect.trigger_emotion("Joy", 0.9)
        affect.trigger_emotion("Anger", 0.7)
        affect.trigger_emotion("Surprise", 0.5)
        top_emotions = affect.get_top_emotions(2)
        assert len(top_emotions) == 2
        assert top_emotions[0]["emotion"] == "Joy"
        assert top_emotions[1]["emotion"] == "Anger"

    def test_get_all_emotions():
        affect = BicaAffect("TestCharacter")
        all_emotions = affect.get_all_emotions()
        assert len(all_emotions) > 0
        for emotion, intensity in all_emotions.items():
            assert 0 <= intensity <= 1

    def test_categorize_emotions():
        affect = BicaAffect("TestCharacter")
        affect.trigger_emotion("Joy", 0.8)
        affect.trigger_emotion("Sadness", 0.6)
        affect.categorize_emotions()
        assert "Joy" in affect.positive_emotions
        assert "Sadness" in affect.negative_emotions

    def test_calculate_mood():
        affect = BicaAffect("TestCharacter")
        affect.trigger_emotion("Joy", 0.8)
        affect.trigger_emotion("Sadness", 0.2)
        mood = affect.calculate_mood()
        assert 0 <= mood <= 1

    def test_simulate_emotional_contagion():
        affect = BicaAffect("TestCharacter")
        affect.simulate_emotional_contagion("Excitement", 0.7)
        assert "Excitement" in affect.runtime_emotions
        assert 0 < affect.runtime_emotions["Excitement"] <= 0.7

    def test_update_personality():
        affect = BicaAffect("TestCharacter")
        initial_traits = affect.cog_model["char_cogModel"][1]["attributes"].copy()
        affect.update_personality("Overcame a significant challenge")
        updated_traits = affect.cog_model["char_cogModel"][1]["attributes"]
        assert initial_traits != updated_traits

    def test_alter_traits_from_emotion():
        affect = BicaAffect("TestCharacter")
        initial_neuroticism = affect.cog_model["char_cogModel"][1]["attributes"].get("Neuroticism", 0.5)
        affect.alter_traits_from_emotion("Fear", 0.8)
        updated_neuroticism = affect.cog_model["char_cogModel"][1]["attributes"].get("Neuroticism", 0.5)
        assert updated_neuroticism > initial_neuroticism

    def test_alter_traits_from_experience():
        affect = BicaAffect("TestCharacter")
        trait_updates = affect.alter_traits_from_experience("Won a prestigious award", 0.9)
        assert len(trait_updates) > 0
        for trait, value in trait_updates.items():
            assert 0 <= value <= 1

    def test_generate_personality_summary():
        affect = BicaAffect("TestCharacter")
        affect.generate_personality_summary()
        assert affect.personality_summary != ""

    def test_generate_artificial_memories():
        affect = BicaAffect("TestCharacter")
        memories = affect.generate_artificial_memories(3)
        assert len(memories) == 3
        for memory in memories:
            assert len(memory) > 0

    # Run all tests
    run_test(test_initialization)
    run_test(test_load_or_create_cog_model)
    run_test(test_initialize_emotions)
    run_test(test_trigger_emotion)
    run_test(test_update_emotions)
    run_test(test_get_top_emotions)
    run_test(test_get_all_emotions)
    run_test(test_categorize_emotions)
    run_test(test_calculate_mood)
    run_test(test_simulate_emotional_contagion)
    run_test(test_update_personality)
    run_test(test_alter_traits_from_emotion)
    run_test(test_alter_traits_from_experience)
    run_test(test_generate_personality_summary)
    run_test(test_generate_artificial_memories)

    print("All tests completed.")