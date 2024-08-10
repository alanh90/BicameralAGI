import json
import os
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
        self.base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.cog_model = self.load_or_create_cog_model()
        self.current_emotions = self.initialize_emotions()
        self.last_update = time.time()
        self.emotion_falloff_rate = 0.05

    def load_or_create_cog_model(self) -> Dict[str, Any]:
        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
        print(f"Attempting to load cognitive model from {file_path}")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                model = json.load(file)
                print(f"Loaded existing cognitive model: {model}")
                return model
        else:
            print(f"Cognitive model not found. Creating a new one.")
            return self.create_cog_model(self.character_name, f"Default cognitive model for {self.character_name}")


    def initialize_emotions(self) -> Dict[str, float]:
        emotions = self.cog_model.get('char_cogModel', [{}])[0].get('attributes', {})
        return {emotion: value for emotion, value in emotions.items() if isinstance(value, (int, float))}

    def trigger_emotion(self, emotion: str, intensity: float):
        intensity = max(0.0, min(intensity, 1.0))
        if emotion in self.current_emotions:
            self.current_emotions[emotion] = max(0.0, min(self.current_emotions[emotion] + intensity, 1.0))
        else:
            self.current_emotions[emotion] = intensity
        self.logger.info(f"Triggered emotion {emotion} with intensity {intensity}")

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

    def create_default_template(self):
        default_template = {
            "char_name": "Default Character",
            "char_description": "A default character template",
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
            "char_keyMemories": []
        }
        template_path = os.path.join(self.base_path, 'data', 'template', 'persona_cog_template.json')
        os.makedirs(os.path.dirname(template_path), exist_ok=True)
        with open(template_path, 'w') as file:
            json.dump(default_template, file, indent=4)
        return default_template

    def create_cog_model(self, character_name: str, description: str):
        template_path = os.path.join(self.base_path, 'data', 'template', 'persona_cog_template.json')
        if not os.path.exists(template_path):
            self.logger.warning(f"Template file not found at {template_path}. Creating a default template.")
            template = self.create_default_template()
        else:
            with open(template_path, 'r') as file:
                template = json.load(file)

        prompt = f"Given this character description: '{description}', generate a detailed cognitive model following this template: {json.dumps(template)}. Ensure all numerical values are between 0.0 and 1.0."

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with creating cognitive models for characters."},
            {"role": "user", "content": prompt}
        ]

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
            try:
                cog_model = json.loads(response["function_call"]["arguments"])
            except json.JSONDecodeError:
                self.logger.error("Failed to parse cognitive model JSON")
                raise ValueError("Failed to parse cognitive model JSON")
        else:
            self.logger.error("Failed to generate cognitive model")
            raise ValueError("Failed to generate cognitive model")

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{character_name}.json')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

        self.logger.info(f"Created cognitive model for {character_name}")
        return cog_model

    def generate_self_reflection(self) -> str:
        prompt = f"Based on the current emotional state: {self.get_all_emotions()} and personality: {self.cog_model}, generate a brief self-reflection report."
        messages = [
            {"role": "system", "content": "You are an AI assistant capable of generating self-reflections based on emotional states and personality traits."},
            {"role": "user", "content": prompt}
        ]
        response = self.gpt_handler.generate_response(messages)
        return response

    def update_personality(self, experience: str):
        print(f"Updating personality based on experience: {experience}")
        prompt = f"Given the current personality: {self.cog_model} and the new experience: '{experience}', suggest minor updates to the personality traits. Provide numerical values between 0 and 1 for each trait."

        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with updating personality traits based on new experiences."},
            {"role": "user", "content": prompt}
        ]

        print("Sending request to GPT model")
        response = self.gpt_handler.generate_response(
            messages=messages,
            functions=[{
                "name": "update_personality_traits",
                "description": "Update personality traits based on new experience",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "updated_traits": {
                            "type": "object",
                            "description": "The updated personality traits",
                            "additionalProperties": {"type": "number"}
                        }
                    },
                    "required": ["updated_traits"]
                }
            }],
            function_call={"name": "update_personality_traits"}
        )

        print(f"Raw response from GPT: {response}")

        updates = {"updated_traits": {}}
        if isinstance(response, dict) and "function_call" in response:
            try:
                arguments = response["function_call"]["arguments"]
                print(f"Function call arguments: {arguments}")
                updates = json.loads(arguments)
            except json.JSONDecodeError:
                print("Failed to parse function call arguments")
                self.logger.warning("Failed to parse function call arguments")

        if not updates.get("updated_traits"):
            print("No traits returned, generating traits from experience")
            self.logger.warning("No traits returned, generating traits from experience")
            updates["updated_traits"] = self._generate_traits_from_experience(experience)

        print(f"Final updates: {updates}")

        for category in self.cog_model["char_cogModel"]:
            if category["category"] == "Personality Traits":
                category["attributes"].update(updates["updated_traits"])
                break
        else:
            print("No 'Personality Traits' category found. Creating a new one.")
            self.cog_model["char_cogModel"].append({
                "category": "Personality Traits",
                "attributes": updates["updated_traits"]
            })

        file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
        with open(file_path, 'w') as file:
            json.dump(self.cog_model, file, indent=4)

        print(f"Updated cognitive model: {self.cog_model}")
        self.logger.info(f"Updated personality traits for {self.character_name}")
    def _generate_traits_from_experience(self, experience: str) -> Dict[str, float]:
        print(f"Generating traits from experience: {experience}")
        prompt = f"Given the experience: '{experience}', generate 5 personality traits that might be affected, with values between 0 and 1."

        response = self.gpt_handler.generate_response(
            messages=[{"role": "user", "content": prompt}],
            functions=[{
                "name": "generate_traits",
                "description": "Generate personality traits based on an experience",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "traits": {
                            "type": "object",
                            "description": "The generated personality traits",
                            "additionalProperties": {"type": "number"}
                        }
                    },
                    "required": ["traits"]
                }
            }],
            function_call={"name": "generate_traits"}
        )

        print(f"Response from trait generation: {response}")

        try:
            if isinstance(response, dict) and "function_call" in response:
                traits = json.loads(response["function_call"]["arguments"])["traits"]
                print(f"Generated traits: {traits}")
                return traits
        except (KeyError, json.JSONDecodeError, TypeError) as e:
            print(f"Error in trait generation: {e}")
            self.logger.error(f"Failed to generate traits from experience: {e}")

        print("Returning default traits")
        return {"Adaptability": 0.7, "Curiosity": 0.8, "Resilience": 0.6, "Creativity": 0.7, "Determination": 0.8}

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

        messages = [
            {"role": "system", "content": "You are an AI assistant capable of generating artificial memories for characters based on their descriptions."},
            {"role": "user", "content": prompt}
        ]

        response = self.gpt_handler.generate_response(
            messages=messages,
            functions=[{
                "name": "generate_memories",
                "description": "Generate artificial memories for a character",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of generated memories"
                        }
                    },
                    "required": ["memories"]
                }
            }],
            function_call={"name": "generate_memories"}
        )

        if isinstance(response, dict) and "function_call" in response:
            memories = json.loads(response["function_call"]["arguments"])["memories"]
            self.cog_model['char_keyMemories'] = memories

            file_path = os.path.join(self.base_path, 'data', 'persona_cog_models', f'{self.character_name}.json')
            with open(file_path, 'w') as file:
                json.dump(self.cog_model, file, indent=4)

            self.logger.info(f"Generated {len(memories)} artificial memories for {self.character_name}")
            return memories
        else:
            self.logger.error("Failed to generate artificial memories")
            raise ValueError("Failed to generate artificial memories")


# Comprehensive testing
if __name__ == "__main__":
    def run_test(test_func):
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")


    def test_initialization():
        affect_system = BicaAffect("alan_turing")
        assert affect_system.character_name == "alan_turing"
        assert isinstance(affect_system.cog_model, dict)
        assert isinstance(affect_system.current_emotions, dict)


    def test_emotion_triggering():
        affect_system = BicaAffect("alan_turing")
        initial_joy = affect_system.current_emotions["Joy"]
        affect_system.trigger_emotion("Joy", 0.5)
        assert affect_system.current_emotions["Joy"] > initial_joy


    def test_emotion_falloff():
        affect_system = BicaAffect("alan_turing")
        affect_system.trigger_emotion("Joy", 1.0)
        initial_joy = affect_system.current_emotions["Joy"]
        time.sleep(2)
        affect_system.update_emotions()
        assert affect_system.current_emotions["Joy"] < initial_joy


    def test_get_top_emotions():
        affect_system = BicaAffect("alan_turing")
        top_emotions = affect_system.get_top_emotions(3)
        assert len(top_emotions) == 3
        assert all(isinstance(e, dict) for e in top_emotions)


    def test_create_cog_model():
        affect_system = BicaAffect("test_character")
        new_model = affect_system.create_cog_model("test_character", "A test character for unit testing")
        assert isinstance(new_model, dict)
        assert "char_name" in new_model
        assert "char_description" in new_model


    def test_generate_self_reflection():
        affect_system = BicaAffect("alan_turing")
        reflection = affect_system.generate_self_reflection()
        assert isinstance(reflection, str)
        assert len(reflection) > 0


    def test_update_personality():
        print("\nStarting test_update_personality")
        affect_system = BicaAffect("alan_turing")
        initial_traits = affect_system.get_dominant_traits()
        print(f"Initial traits: {initial_traits}")
        try:
            affect_system.update_personality("Discovered a groundbreaking mathematical theorem")
            updated_traits = affect_system.get_dominant_traits()
            print(f"Updated traits: {updated_traits}")
            assert initial_traits != updated_traits, "Traits should have been updated"
        except Exception as e:
            print(f"Error in test_update_personality: {str(e)}")
            raise

    def test_generate_artificial_memories():
        affect_system = BicaAffect("alan_turing")
        memories = affect_system.generate_artificial_memories(3)
        assert isinstance(memories, list)
        assert len(memories) == 3
        assert all(isinstance(m, str) for m in memories)


    print("Running BicaAffect tests...")
    run_test(test_initialization)
    run_test(test_emotion_triggering)
    run_test(test_emotion_falloff)
    run_test(test_get_top_emotions)
    run_test(test_create_cog_model)
    run_test(test_generate_self_reflection)
    run_test(test_update_personality)
    run_test(test_generate_artificial_memories)
