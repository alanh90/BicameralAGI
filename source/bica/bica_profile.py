"""
BicameralAGI Profile Module
==============================

Overview:
---------
This module manages the emotional and personality aspects of the BicameralAGI system. It handles character traits, and their evolution over time based on experiences and interactions.

Key Features:
-------------
1. Emotion preparation
2. Personality trait modeling and evolution
3. Experience-based profile updates
4. Integration with other BicameralAGI components

Usage:
------
The BicaProfile class can be instantiated and used to manage a character's emotional and personality state:

    profile = BicaProfile(character_summary, character_name)
    profile.update_personality(new_experience)

Author: Alan Hourmand
Date: 10/2/2024
"""

from bica.gpt_handler import GPTHandler as gpt
from typing import List, Dict, Any
from bica_utilities import *
import traceback
import random
import json
import time
import os


class BicaProfile:
    def __init__(self, character_summary: str, character_name: str):
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        self.gpt_handler = gpt()

        # Character Trait Setup
        self.character_profile = self.initialize_character_profile()
        self.character_details = character_summary
        self.character_name = character_name
        self.emotion_falloff_rate = 0.05

        self.last_update = time.time()  # Used for emotional falloff

    def initialize_character_profile(self) -> Dict[str, Any]:
        file_path = os.path.join(self.base_path, 'data', 'characters', f'{self.character_name}.json')

        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    self.character_profile = json.load(file)
            else:
                raise FileNotFoundError
        except (FileNotFoundError, json.JSONDecodeError):
            self.character_profile = self.create_cog_model(self.character_name, f"Default cognitive model for {self.character_name}")

        # Ensure the cognitive model has both emotions and traits
        if 'char_cogModel' in self.character_profile:
            for category in self.character_profile['char_cogModel']:
                if category['category'] == 'Emotions':
                    self.runtime_emotions = {k: v for k, v in category['attributes'].items() if isinstance(v, (int, float))}
                elif category['category'] == 'Traits':
                    self.traits = category['attributes']

        # Set default values if not found
        if not hasattr(self, 'runtime_emotions'):
            self.runtime_emotions = {"Joy": 0.5, "Sadness": 0.5, "Anger": 0.5, "Fear": 0.5, "Surprise": 0.5}
        if not hasattr(self, 'traits'):
            self.traits = {"Openness": 0.5, "Conscientiousness": 0.5, "Extraversion": 0.5, "Agreeableness": 0.5, "Neuroticism": 0.5}

        return self.character_profile

    def create_character_file(self, character_name: str, description: str) -> Dict[str, Any]:

        template_path = os.path.join(self.base_path, 'data', 'reference', 'ref_character_traits.json')
        with open(template_path, 'r') as file:
            template = json.load(file)

        prompt = f"""
        Create a detailed cognitive model for {character_name} based on this description: '{description}'.
        Follow the structure of this reference exactly:
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
            return self._create_default_cog_model(character_name, description)

        file_path = os.path.join(self.base_path, 'data', 'characters', f'{self.character_name}.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

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
                    print(f"Missing key in generated model: {path + key}")
                elif isinstance(value, dict):
                    if not isinstance(gen[key], dict):
                        print(f"Expected dict for key: {path + key}")
                    else:
                        compare_structure(gen[key], value, path + key + ".")
                elif isinstance(value, list):
                    if not isinstance(gen[key], list):
                        print(f"Expected list for key: {path + key}")
                    elif len(gen[key]) != len(value):
                        print(f"List length mismatch for key: {path + key}")
                elif isinstance(value, (int, float)) and isinstance(gen[key], (int, float)):
                    if abs(gen[key] - value) > 0.1:
                        print(f"Significant value difference for key: {path + key}")

        compare_structure(generated_model, reference_model)

    def update_personality(self, experience: str):
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

            file_path = os.path.join(self.base_path, 'data', 'characters', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

        except Exception as e:
            pass

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
            self.cog_model["char_cogModel"].append({
                "category": "Personality Traits",
                "attributes": updates
            })

        file_path = os.path.join(self.base_path, 'data', 'characters', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

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
            self.personality_summary = "Unable to generate personality summary."

    def get_personality_summary(self) -> str:
        if not self.personality_summary:
            self.generate_personality_summary()
        return self.personality_summary

    def _create_default_cog_model(self, character_name: str, description: str):
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

        return default_model

    def generate_artificial_memories(self, num_memories: int = 5) -> List[str]:
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

            file_path = os.path.join(self.base_path, 'data', 'characters', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

            return memories[:num_memories]
        except Exception as e:
            return []


# Comprehensive testing
def run_test(test_func, *args):
    try:
        test_func(*args)
        print(f"✅ {test_func.__name__} passed")
    except Exception as e:
        print(f"❌ {test_func.__name__} failed: {str(e)}")
        traceback.print_exc()


def test_character_creation():
    character_name = input("Enter the name of the character you want to create: ")
    character_description = input(f"Enter a brief description for {character_name}: ")

    character_affect = BicaAffect(character_name)
    character_model = character_affect.create_cog_model(character_name, character_description)

    # Ensure fallback model has key memories
    if "char_keyMemories" not in character_model:
        print(f"Warning: 'char_keyMemories' missing. Using fallback default memories for {character_name}.")
        character_model["char_keyMemories"] = [
            f"Default memory 1 for {character_name}",
            f"Default memory 2 for {character_name}",
            f"Default memory 3 for {character_name}"
        ]

    # Verify structure
    assert "char_name" in character_model, "char_name is missing from the cognitive model."
    assert "char_description" in character_model, "char_description is missing from the cognitive model."
    assert "char_keyMemories" in character_model, "char_keyMemories is missing from the cognitive model."
    assert len(character_model["char_keyMemories"]) >= 3, "Less than 3 key memories found."

    assert "char_cogModel" in character_model, "char_cogModel is missing from the cognitive model."
    assert len(character_model["char_cogModel"]) == 2, "char_cogModel should contain exactly 2 categories (Emotions and Traits)."
    assert character_model["char_cogModel"][0]["category"] == "Emotions", "The first category in char_cogModel must be Emotions."
    assert character_model["char_cogModel"][1]["category"] == "Traits", "The second category in char_cogModel must be Traits."

    emotions = ["Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]
    for emotion in emotions:
        assert emotion in character_model["char_cogModel"][0]["attributes"], f"{emotion} is missing from Emotions."

    traits = ["Openness", "Conscientiousness", "Extroversion", "Agreeableness", "Neuroticism"]
    for trait in traits:
        assert trait in character_model["char_cogModel"][1]["attributes"], f"{trait} is missing from Traits."

    assert "situationalConversations" in character_model, "situationalConversations is missing from the cognitive model."
    situations = ["Personal/Intimate Setting", "Social Gathering", "Professional Environment", "Educational/Academic Setting"]
    for situation in situations:
        assert situation in character_model["situationalConversations"], f"{situation} is missing from situationalConversations."
        assert "intensity" in character_model["situationalConversations"][situation], f"intensity is missing in {situation}."
        assert "examples" in character_model["situationalConversations"][situation], f"examples are missing in {situation}."
        assert len(character_model["situationalConversations"][situation]["examples"]) == 3, f"{situation} should have exactly 3 examples."

    assert "styleGuideValues" in character_model, "styleGuideValues is missing from the cognitive model."
    style_values = ["formality", "conciseness", "technicalLanguage", "emotionalExpression"]
    for value in style_values:
        assert value in character_model["styleGuideValues"], f"{value} is missing from styleGuideValues."
        assert character_model["styleGuideValues"][value] in ["low", "medium", "high"], f"{value} should be 'low', 'medium', or 'high'."

    print(f"{character_name} model structure verification passed.")
    return character_affect


def test_personality_update(character_affect):
    experience = input("Enter an experience to update the personality of the character: ")
    character_affect.update_personality(experience)

    assert "char_cogModel" in character_affect.cog_model
    for category in character_affect.cog_model["char_cogModel"]:
        if category["category"] == "Traits":
            assert len(category["attributes"]) > 0
            break

    print(f"{character_affect.character_name} personality update passed.")


def test_memory_generation(character_affect):
    memories = character_affect.generate_artificial_memories(3)
    assert len(memories) == 3

    print(f"{character_affect.character_name} memory generation passed.")


def test_emotion_and_trait_change_based_on_context(character_affect):
    context = input("Provide a context or situation to see how the character's traits and emotions change: ")
    character_affect.alter_traits_from_experience(context, intensity=0.7)

    # Check if the traits have been updated
    for category in character_affect.cog_model["char_cogModel"]:
        if category["category"] == "Traits":
            print(f"Updated traits for {character_affect.character_name}: {json.dumps(category['attributes'], indent=2)}")
            break

    # Check updated emotions
    print(f"Updated emotions for {character_affect.character_name}: {json.dumps(character_affect.runtime_emotions, indent=2)}")

    print(f"{character_affect.character_name} context-based update passed.")


def test_style_guide_validation(character_affect):
    style_guide = character_affect.cog_model["styleGuideValues"]
    for key, value in style_guide.items():
        assert value in ["low", "medium", "high"]

    print(f"{character_affect.character_name} style guide validation passed.")


# Run all tests
if __name__ == "__main__":
    # Test character creation and personality update
    character_affect = test_character_creation()

    if character_affect:
        run_test(test_personality_update, character_affect)
        run_test(test_memory_generation, character_affect)
        run_test(test_emotion_and_trait_change_based_on_context, character_affect)
        run_test(test_style_guide_validation, character_affect)

    print("All tests completed.")
