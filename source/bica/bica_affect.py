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
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.logger = BicaLogging("BicaAffect")
        self.character_name = character_name
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.cog_model = self.load_or_create_cog_model()
        self.runtime_emotions = self.initialize_emotions()
        self.emotion_falloff_rate = 0.05
        self.last_update = time.time()

        self.negative_emotions = None
        self.positive_emotions = None
        self.emotion_decay_time = 90  # 90 seconds for emotion decay
        self.personality_summary = ""
        self.emotional_stability = self.calculate_emotional_stability()
        self.emotional_intelligence = random.uniform(0.3, 0.8)  # Initialize with a random value

        self.logger.info(f"Initialized BicaAffect for character: {character_name}")
        self.logger.info(f"Base path set to: {self.base_path}")
        self.logger.info(f"Initial runtime emotions: {json.dumps(self.runtime_emotions)}")
        self.logger.info(f"Emotional stability: {self.emotional_stability:.2f}")
        self.logger.info(f"Emotional intelligence: {self.emotional_intelligence:.2f}")

    def calculate_emotional_stability(self):
        self.logger.info(f"Calculating emotional stability for {self.character_name}")

        try:
            conscientiousness = self.cog_model['char_cogModel'][1]['attributes'].get('Conscientiousness', 0.5)
            neuroticism = self.cog_model['char_cogModel'][1]['attributes'].get('Neuroticism', 0.5)
            es = (conscientiousness + (1 - neuroticism)) / 2
            self.logger.info(f"Emotional stability calculated: {es:.2f}")
            return es
        except (IndexError, KeyError):
            self.logger.warning("Failed to calculate emotional stability. Using default value.")
            return 0.5

    def load_or_create_cog_model(self) -> Dict[str, Any]:
        self.logger.info(f"Attempting to load cognitive model for {self.character_name}")
        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')

        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    self.logger.info(f"Cognitive model {'loaded' if os.path.exists(file_path) else 'created'} for {self.character_name}")
                    return json.load(file)
            else:
                raise FileNotFoundError
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.info(f"Cognitive model not found or invalid. Creating a new one for {self.character_name}.")
            return self.create_cog_model(self.character_name, f"Default cognitive model for {self.character_name}")

    def initialize_emotions(self) -> Dict[str, float]:
        self.logger.info(f"Initializing emotions for {self.character_name}")
        try:
            emotions = self.cog_model.get('char_cogModel', [{}])[0].get('attributes', {})
            self.logger.info(f"Emotions initialized: {json.dumps(emotions)}")
            return {emotion: value for emotion, value in emotions.items() if isinstance(value, (int, float))}
        except IndexError:
            self.logger.warning("Failed to initialize emotions. Using default emotions.")
            return {"Joy": 0.5, "Sadness": 0.5, "Anger": 0.5, "Fear": 0.5, "Surprise": 0.5}

    def trigger_emotion(self, emotion: str, intensity: float, stimulate: bool = False):
        self.logger.info(f"Triggering emotion for {self.character_name}: {emotion} with intensity {intensity:.2f}")

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
        self.logger.info(f"Updating emotions for {self.character_name}")
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
        self.logger.info(f"Emotions updated: {json.dumps(self.runtime_emotions)}")

    def get_top_emotions(self, n: int = 3) -> List[Dict[str, Any]]:
        self.update_emotions()
        sorted_emotions = sorted(self.runtime_emotions.items(), key=lambda x: x[1], reverse=True)
        return [{"emotion": k, "intensity": v} for k, v in sorted_emotions[:min(n, len(sorted_emotions))]]

    def get_all_emotions(self):
        self.update_emotions()
        return self.runtime_emotions

    def categorize_emotions(self):
        self.logger.info(f"Categorizing emotions for {self.character_name}")
        positive_emotions = ['Joy', 'Excitement', 'Contentment']
        negative_emotions = ['Sadness', 'Anger', 'Fear']
        self.positive_emotions = {e: v for e, v in self.runtime_emotions.items() if e in positive_emotions}
        self.negative_emotions = {e: v for e, v in self.runtime_emotions.items() if e in negative_emotions}
        self.logger.info(f"Positive emotions: {json.dumps(self.positive_emotions)}")
        self.logger.info(f"Negative emotions: {json.dumps(self.negative_emotions)}")

    def calculate_mood(self):
        self.logger.info(f"Calculating mood for {self.character_name}")
        self.categorize_emotions()
        positive_score = sum(self.positive_emotions.values())
        negative_score = sum(self.negative_emotions.values())
        total_emotions = len(self.runtime_emotions)

        if total_emotions == 0:
            self.logger.warning("No emotions present. Returning neutral mood.")
            return 0.5

        base_mood = (positive_score - negative_score) / total_emotions
        optimism = self.cog_model['char_cogModel'][1]['attributes'].get('Optimism', 0.5)
        mood = (base_mood + optimism) / 2
        self.logger.info(f"Calculated mood: {mood:.2f}")
        return mood

    def simulate_emotional_contagion(self, external_emotion: str, intensity: float):
        self.logger.info(f"Simulating emotional contagion for {self.character_name}: {external_emotion} with intensity {intensity:.2f}")
        contagion_factor = max(0.1, min(random.uniform(0.1, 0.5), 1.0))  # Ensure factor is between 0.1 and 1.0
        self.trigger_emotion(external_emotion, intensity * contagion_factor)
        self.logger.info(f"Emotional contagion result: {external_emotion} triggered with intensity {intensity * contagion_factor:.2f}")

    def create_cog_model(self, character_name: str, description: str) -> Dict[str, Any]:
        self.logger.info(f"Creating cognitive model for {character_name}")
        self.logger.info(f"Character description: {description[:100]}...")

        template_path = os.path.join(self.base_path, 'data', 'template', 'persona_cog_template.json')
        self.logger.info(f"Using template from: {template_path}")
        with open(template_path, 'r') as file:
            template = json.load(file)

        prompt = f"""
            Create a detailed cognitive model for {character_name} based on this description: '{description}'.
            Follow these specific instructions:
            1. char_name: Use "{character_name}" exactly.
            2. char_description: Provide a concise description based on the given information.
            3. char_keyMemories: Generate exactly 8 key memories that shaped the character's life and personality.
            4. char_cogModel: Include all categories and attributes from the template, with values between 0.0 and 1.0.
               Ensure all emotion values sum up to approximately 6.0.
            5. situationalConversations: For each situation, provide an intensity value between 0.0 and 1.0, and exactly 3 example conversations.
            6. styleGuideValues: Include all style guide values as in the template, using the exact same keys and value formats.

            Ensure that your response exactly matches the structure of the provided template, including all required fields and proper nesting of objects and arrays.
            """

        try:
            response = self.gpt_handler.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with creating cognitive models for characters."},
                    {"role": "user", "content": prompt}
                ],
                functions=[{
                    "name": "create_cognitive_model",
                    "description": "Create a cognitive model based on the template",
                    "parameters": {
                        "type": "object",
                        "properties": template
                    }
                }],
                function_call={"name": "create_cognitive_model"}
            )

            if isinstance(response, dict) and "function_call" in response:
                cog_model = json.loads(response["function_call"]["arguments"])
                self._validate_cog_model(cog_model, template)
            else:
                raise ValueError("Unexpected response format from GPT")

        except Exception as e:
            self.logger.error(f"Failed to create cognitive model for {character_name}: {str(e)}")
            return self._create_default_cog_model(character_name, description)

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

        self.logger.info(f"Created cognitive model for {character_name}")
        return cog_model

    def _validate_cog_model(self, cog_model: Dict[str, Any], template: Dict[str, Any]) -> None:
        def validate_structure(model, temp, path=""):
            for key, value in temp.items():
                if key not in model:
                    raise ValueError(f"Missing key: {path + key}")
                if isinstance(value, dict):
                    if not isinstance(model[key], dict):
                        raise ValueError(f"Expected dict for key: {path + key}")
                    validate_structure(model[key], value, path + key + ".")
                elif isinstance(value, list):
                    if not isinstance(model[key], list):
                        raise ValueError(f"Expected list for key: {path + key}")
                    if key == "char_keyMemories" and len(model[key]) != 8:
                        raise ValueError(f"Expected exactly 8 key memories, got {len(model[key])}")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            validate_structure(model[key][i], item, f"{path + key}[{i}].")

        validate_structure(cog_model, template)

        # Additional validations
        emotions_sum = sum(cog_model['char_cogModel'][0]['attributes'].values())
        if not 5.9 <= emotions_sum <= 6.1:
            raise ValueError(f"Sum of emotion values should be approximately 6.0, got {emotions_sum}")

        for situation in cog_model['situationalConversations'].values():
            if len(situation['examples']) != 3:
                raise ValueError(f"Expected exactly 3 examples for each situation, got {len(situation['examples'])}")

        self.logger.info("Cognitive model structure validated successfully")

    def compare_with_reference(self, generated_model: Dict[str, Any], reference_model: Dict[str, Any]) -> None:
        def compare_structure(gen, ref, path=""):
            for key, value in ref.items():
                if key not in gen:
                    self.logger.warning(f"Missing key in generated model: {path + key}")
                elif isinstance(value, dict):
                    if not isinstance(gen[key], dict):
                        self.logger.warning(f"Expected dict for key in generated model: {path + key}")
                    else:
                        compare_structure(gen[key], value, path + key + ".")
                elif isinstance(value, list):
                    if not isinstance(gen[key], list):
                        self.logger.warning(f"Expected list for key in generated model: {path + key}")
                    elif len(gen[key]) != len(value):
                        self.logger.warning(f"List length mismatch for key: {path + key}")
                elif isinstance(value, (int, float)) and isinstance(gen[key], (int, float)):
                    if abs(gen[key] - value) > 0.1:
                        self.logger.warning(f"Significant value difference for key {path + key}: Reference {value}, Generated {gen[key]}")

        compare_structure(generated_model, reference_model)
        self.logger.info("Comparison with reference model completed")

    def update_personality(self, experience: str):
        self.logger.info(f"Updating personality for {self.character_name} based on experience: {experience}")

        prompt = f"""
        <instruction>
        Given the current personality: 
        <personality>
        {json.dumps(self.cog_model['char_cogModel'], indent=2)}
        </personality>
        And the new experience: '{experience}', suggest minor updates to the personality traits.
        </instruction>

        <output_format>
        Provide your response as a JSON object with trait names as keys and updated values (between 0 and 1) as values:
        {{
            "trait1": 0.7,
            "trait2": 0.8,
            ...
        }}
        </output_format>
        """

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with updating personality traits based on new experiences."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.gpt_handler.generate_response(messages)
            updates = json.loads(response)

            for category in self.cog_model["char_cogModel"]:
                if category["category"] == "Personality Traits":
                    for trait, value in updates.items():
                        if trait in category["attributes"]:
                            category["attributes"][trait] = (category["attributes"][trait] + value) / 2
                        else:
                            category["attributes"][trait] = value
                    break
            else:
                self.cog_model["char_cogModel"].append({
                    "category": "Personality Traits",
                    "attributes": updates
                })

            file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

            self.logger.info(f"Personality updated successfully for {self.character_name}")
            self.logger.info(f"Updated traits: {json.dumps(updates, indent=2)}")
        except Exception as e:
            self.logger.error(f"Failed to update personality for {self.character_name}: {str(e)}")

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

    def _create_default_cog_model(self, character_name: str, description: str):
        self.logger.warning(f"Creating default cognitive model for {character_name}")
        default_model = {
            "char_name": character_name,
            "char_description": description,
            "char_keyMemories": [
                "Default memory 1",
                "Default memory 2",
                "Default memory 3",
                "Default memory 4",
                "Default memory 5",
                "Default memory 6",
                "Default memory 7",
                "Default memory 8"
            ],
            "char_cogModel": [
                {
                    "category": "Emotions",
                    "attributes": {
                        "Joy": 0.5,
                        "Sadness": 0.5,
                        "Anger": 0.5,
                        "Fear": 0.5,
                        "Surprise": 0.5
                    }
                },
                {
                    "category": "Personality Traits",
                    "attributes": {
                        "Openness": 0.5,
                        "Conscientiousness": 0.5,
                        "Extraversion": 0.5,
                        "Agreeableness": 0.5,
                        "Neuroticism": 0.5
                    }
                }
            ],
            "situationalConversations": {
                "Default Situation": {
                    "intensity": 0.5,
                    "examples": [
                        "Default example 1",
                        "Default example 2",
                        "Default example 3"
                    ]
                }
            },
            "styleGuideValues": {
                "formality": "medium",
                "conciseness": "medium",
                "technicalLanguage": "medium",
                "emotionalExpression": "medium"
            }
        }

        self.logger.info(f"Created default cognitive model for {character_name}")
        return default_model

    def generate_artificial_memories(self, num_memories: int = 5) -> List[str]:
        self.logger.info(f"Generating {num_memories} artificial memories for {self.character_name}")

        prompt = f"""
        <instruction>
        Based on this character description: '{self.cog_model['char_description']}', 
        generate {num_memories} detailed, vivid memories that shaped this character. 
        Each memory should be a paragraph long.
        </instruction>

        <output_format>
        Provide your response as a JSON array of strings, where each string is a memory:
        [
            "Memory 1 description...",
            "Memory 2 description...",
            ...
        ]
        </output_format>
        """

        messages = [
            {"role": "system", "content": "You are an AI that creates artificial memories for characters based on their background."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.gpt_handler.generate_response(messages)
            memories = json.loads(response)

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


    def test_alan_turing_model():
        logger = BicaLogging("TuringTest")
        logger.info("Starting Alan Turing model test")

        turing_affect = BicaAffect("alan_turing")

        # Test creating cognitive model
        logger.info("Creating Alan Turing's cognitive model")
        turing_description = """
        Alan Turing was a brilliant British mathematician, logician, and computer scientist. 
        He is considered the father of theoretical computer science and artificial intelligence. 
        Key characteristics:
        - Highly intelligent and analytical
        - Innovative thinker
        - Socially awkward but kind-hearted
        - Passionate about mathematics and puzzles
        - Persevered through adversity
        - Played a crucial role in breaking the Enigma code during World War II
        - Faced persecution due to his homosexuality
        - Contributed significantly to the fields of computer science and artificial intelligence
        """
        turing_model = turing_affect.create_cog_model("alan_turing", turing_description)
        assert turing_model is not None, "Turing model is None"

        if turing_model["char_name"] == "alan_turing":
            # Only run these assertions if it's not the default model
            assert len(turing_model["char_keyMemories"]) == 8, f"Incorrect number of key memories: {len(turing_model['char_keyMemories'])}"
            assert len(turing_model["char_cogModel"]) > 0, "char_cogModel is empty"
            assert "situationalConversations" in turing_model, "situationalConversations missing from model"
            assert "styleGuideValues" in turing_model, "styleGuideValues missing from model"
        else:
            logger.warning("Default model was created instead of Alan Turing specific model")

        # Test generating artificial memories
        logger.info("Generating artificial memories")
        memories = turing_affect.generate_artificial_memories(3)
        assert len(memories) == 3
        logger.info(f"Generated {len(memories)} memories")

        # Test updating personality
        logger.info("Updating personality based on significant experience")
        experience = "Successfully broke the Enigma code, contributing significantly to the Allied war effort."
        initial_traits = turing_affect.cog_model["char_cogModel"][1]["attributes"].copy()
        turing_affect.update_personality(experience)
        updated_traits = turing_affect.cog_model["char_cogModel"][1]["attributes"]
        assert initial_traits != updated_traits
        logger.info("Personality updated successfully")

        # Test emotional reactions
        logger.info("Testing emotional reactions")
        turing_affect.trigger_emotion("Joy", 0.8)
        turing_affect.trigger_emotion("Curiosity", 0.9)
        top_emotions = turing_affect.get_top_emotions(2)
        assert len(top_emotions) == 2
        assert top_emotions[0]["emotion"] == "Curiosity"
        assert top_emotions[1]["emotion"] == "Joy"
        logger.info(f"Top emotions: {top_emotions}")

        # Test mood calculation
        logger.info("Calculating mood")
        mood = turing_affect.calculate_mood()
        assert 0 <= mood <= 1
        logger.info(f"Calculated mood: {mood}")

        logger.info("Alan Turing model test completed successfully")

        # Load the reference model
        reference_path = os.path.join(turing_affect.base_path, 'data', 'persona_cog_models', 'alan_turing_reference.json')
        with open(reference_path, 'r') as file:
            reference_model = json.load(file)

        # Compare the generated model with the reference
        turing_affect.compare_with_reference(turing_model, reference_model)


    # Run the new Alan Turing test
    run_test(test_alan_turing_model)

    print("All tests completed.")

    # Print some details about the Alan Turing model
    turing_affect = BicaAffect("alan_turing")
    logger = BicaLogging("MainAlanTuring")
    logger.info("Printing Alan Turing model summary")
    print("\nAlan Turing Cognitive Model Summary:")
    print(turing_affect.get_personality_summary())
    print("\nTop Emotions for Alan Turing:")
    top_emotions = turing_affect.get_top_emotions(3)
    for emotion in top_emotions:
        print(f"{emotion['emotion']}: {emotion['intensity']:.2f}")
    logger.info("Alan Turing model summary printed")
