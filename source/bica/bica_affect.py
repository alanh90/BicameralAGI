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
            return {emotion: value for emotion, value in emotions.items() if isinstance(value, (int, float))}
        except IndexError:
            self.logger.warning("Failed to initialize emotions. Using default emotions.")
            return {"Joy": 0.5, "Sadness": 0.5, "Anger": 0.5, "Fear": 0.5, "Surprise": 0.5}


    def trigger_emotion(self, emotion: str, intensity: float):
        self.logger.info(f"Triggering emotion for {self.character_name}: {emotion} with intensity {intensity:.2f}")
        intensity = max(0.0, min(intensity, 1.0))
        if emotion in self.runtime_emotions:
            self.runtime_emotions[emotion] = max(0.0, min(self.runtime_emotions[emotion] + intensity, 1.0))
        else:
            self.logger.warning(f"Triggering unknown emotion: {emotion}")
            self.runtime_emotions[emotion] = intensity

    def update_emotions(self):
        self.logger.info(f"Updating emotions for {self.character_name}")
        current_time = time.time()
        time_diff = current_time - self.last_update
        falloff_amount = time_diff * self.emotion_falloff_rate

        for emotion in list(self.runtime_emotions.keys()):
            default_value = self.cog_model['char_cogModel'][0]['attributes'].get(emotion, 0.5)
            current_value = self.runtime_emotions[emotion]

            if current_value > default_value:
                self.runtime_emotions[emotion] = max(default_value, current_value - falloff_amount)
            elif current_value < default_value:
                self.runtime_emotions[emotion] = min(default_value, current_value + falloff_amount)

        self.last_update = current_time
        self.logger.info(f"Emotions updated: {json.dumps(self.runtime_emotions)}")

    def get_top_emotions(self, n: int = 3) -> List[Dict[str, Any]]:
        self.update_emotions()
        sorted_emotions = sorted(self.runtime_emotions.items(), key=lambda x: x[1], reverse=True)
        return [{"emotion": k, "intensity": v} for k, v in sorted_emotions[:n]]

    def get_all_emotions(self):
        self.update_emotions()
        return self.runtime_emotions

    def create_cog_model(self, character_name: str, description: str) -> Dict[str, Any]:
        self.logger.info(f"Creating cognitive model for {character_name}")

        template_path = os.path.join(self.base_path, 'data', 'template', 'persona_cog_template.json')
        with open(template_path, 'r') as file:
            template = json.load(file)

        prompt = f"""
        Create a detailed cognitive model for {character_name} based on this description: '{description}'.
        Follow the structure of this template exactly:
        {json.dumps(template, indent=2)}

        Replace all placeholder values with appropriate content for {character_name}.
        Ensure that:
        1. char_name is set to "{character_name}"
        2. char_description is based on the provided description
        3. char_cogModel contains exactly two categories: "Emotions" and "Traits", with all listed attributes
        4. situationalConversations includes all 8 situations with intensity and 3 examples each
        5. styleGuideValues includes all listed style guide values
        """

        try:
            response = self.gpt_handler.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with creating cognitive models for characters."},
                    {"role": "user", "content": prompt}
                ]
            )
            cog_model = json.loads(response)

            # Ensure critical keys are present
            assert "char_keyMemories" in cog_model, "Missing 'char_keyMemories' in generated cognitive model"
            self._validate_cog_model(cog_model, template)
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
                    if len(model[key]) != len(value):
                        raise ValueError(f"List length mismatch for key: {path + key}")
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            validate_structure(model[key][i], item, f"{path + key}[{i}].")

        validate_structure(cog_model, template)

        assert len(cog_model["char_cogModel"]) == 2, "char_cogModel must contain exactly 2 categories"
        assert cog_model["char_cogModel"][0]["category"] == "Emotions", "First category in char_cogModel must be Emotions"
        assert cog_model["char_cogModel"][1]["category"] == "Traits", "Second category in char_cogModel must be Traits"

        for situation in cog_model["situationalConversations"].values():
            assert len(situation["examples"]) == 3, "Each situation must have exactly 3 examples"

        for value in cog_model["styleGuideValues"].values():
            assert value in ["low", "medium", "high"], "Style guide values must be 'low', 'medium', or 'high'"

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
        Given the current personality: 
        {json.dumps(self.cog_model['char_cogModel'], indent=2)}
        And the new experience: '{experience}', suggest minor updates to the personality traits.
        Provide your response as a JSON object with trait names as keys and updated values (between 0 and 1) as values.
        """

        try:
            response = self.gpt_handler.generate_response(
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with updating personality traits based on new experiences."},
                    {"role": "user", "content": prompt}
                ]
            )
            updates = json.loads(response)

            for category in self.cog_model["char_cogModel"]:
                if category["category"] == "Traits":
                    for trait, value in updates.items():
                        if trait in category["attributes"]:
                            category["attributes"][trait] = (category["attributes"][trait] + value) / 2

            file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

            self.logger.info(f"Personality updated successfully for {self.character_name}")
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
                    "category": "Traits",
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


    def test_alan_turing_model_generation():
        turing_affect = BicaAffect("Alan Turing")
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
        turing_model = turing_affect.create_cog_model("Alan Turing", turing_description)

        # Verify structure
        assert "char_name" in turing_model
        assert "char_description" in turing_model
        assert "char_keyMemories" in turing_model
        assert len(turing_model["char_keyMemories"]) == 8

        assert "char_cogModel" in turing_model
        assert len(turing_model["char_cogModel"]) == 2
        assert turing_model["char_cogModel"][0]["category"] == "Emotions"
        assert turing_model["char_cogModel"][1]["category"] == "Traits"

        emotions = ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise", "Love", "Trust", "Anticipation", "Curiosity", "Shame", "Pride", "Guilt", "Envy", "Gratitude", "Awe", "Contempt", "Anxiety", "Boredom", "Confusion"]
        for emotion in emotions:
            assert emotion in turing_model["char_cogModel"][0]["attributes"]

        traits = ["Openness", "Conscientiousness", "Extroversion", "Agreeableness", "Neuroticism", "Confirmation Bias", "Anchoring Bias", "Availability Heuristic", "Dunning-Kruger Effect", "Negativity Bias", "Reward Processing", "Goal-Oriented Behavior", "Intrinsic Motivation", "Extrinsic Motivation", "Fatigue Level", "Stress Level", "Pain", "Theory of Mind", "Empathy", "Social Cue Interpretation", "Facial Recognition", "Emotional Intelligence", "Alertness", "Arousal", "Self-Awareness",
                  "Metacognition", "Visual Processing", "Auditory Processing", "Proprioception", "Vestibular Processing", "Planning", "Organizing", "Time Management", "Task Initiation", "Impulse Control", "Emotional Regulation", "Cognitive Flexibility", "Self-Monitoring", "Attention", "Working Memory", "Long-term Memory", "Learning", "Decision Making", "Problem Solving", "Reasoning", "Language Processing", "Spatial Awareness", "Pattern Recognition", "Creativity"]
        for trait in traits:
            assert trait in turing_model["char_cogModel"][1]["attributes"]

        assert "situationalConversations" in turing_model
        situations = ["Personal/Intimate Setting", "Social Gathering", "Professional Environment", "Educational/Academic Setting", "Public Space", "Online/Digital Interaction", "Formal Event", "Emergency or High-Stress Situation"]
        for situation in situations:
            assert situation in turing_model["situationalConversations"]
            assert "intensity" in turing_model["situationalConversations"][situation]
            assert "examples" in turing_model["situationalConversations"][situation]
            assert len(turing_model["situationalConversations"][situation]["examples"]) == 3

        assert "styleGuideValues" in turing_model
        style_values = ["formality", "conciseness", "technicalLanguage", "emotionalExpression", "sentenceComplexity", "vocabularyRange", "idiomUsage", "directness", "verbalPacing", "fillerWords", "politeness", "verbVoice", "detailSpecificity", "tonality", "figurativeLanguage", "contentRelevance", "salutation", "informationDensity"]
        for value in style_values:
            assert value in turing_model["styleGuideValues"]
            assert turing_model["styleGuideValues"][value] in ["low", "medium", "high"]

        print("Alan Turing model structure verification passed.")
        return turing_model


    # Run the test
    turing_model = test_alan_turing_model_generation()

    # Print some details about the generated model
    print("\nAlan Turing Cognitive Model Summary:")
    print(f"Name: {turing_model['char_name']}")
    print(f"Description: {turing_model['char_description'][:100]}...")
    print("\nSample Key Memories:")
    for memory in turing_model['char_keyMemories'][:3]:
        print(f"- {memory}")
    print("\nSample Emotions:")
    for emotion, value in list(turing_model['char_cogModel'][0]['attributes'].items())[:5]:
        print(f"- {emotion}: {value:.2f}")
    print("\nSample Traits:")
    for trait, value in list(turing_model['char_cogModel'][1]['attributes'].items())[:5]:
        print(f"- {trait}: {value:.2f}")
    print("\nSample Situational Conversation:")
    situation = next(iter(turing_model['situationalConversations']))
    print(f"- {situation}:")
    print(f"  Intensity: {turing_model['situationalConversations'][situation]['intensity']:.2f}")
    print(f"  Example: {turing_model['situationalConversations'][situation]['examples'][0]}")
    print("\nSample Style Guide Values:")
    for style, value in list(turing_model['styleGuideValues'].items())[:5]:
        print(f"- {style}: {value}")
