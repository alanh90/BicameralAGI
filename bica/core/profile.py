import os
import json
from pydantic import BaseModel
from external.gpt_handler import GPTHandler
from typing import Dict, Any
import configparser
import shutil
import time
import re
from typing import Union


class CharacterProfileSchema(BaseModel):
    characterInfo: Dict[str, Union[str, int, Dict[str, str]]]
    cognitiveModel: Dict[str, Any]
    communicationStyle: Dict[str, str]


class BicaProfile:
    def __init__(self, character_name: str, character_summary: str, gpt_handler: GPTHandler):
        # Load configuration to determine base path
        config = configparser.ConfigParser()
        try:
            config.read(os.path.join(os.path.dirname(__file__), '../config.ini'))
            self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        except (configparser.Error, FileNotFoundError) as e:
            # Fallback to default base path if config file is missing or corrupted
            print(f"Warning: Failed to read configuration file. Using default base path. Error: {str(e)}")
            self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

        # Initialize GPT handler and character information
        self.gpt_handler = gpt_handler
        self.character_name = self.sanitize_filename(character_name)
        self.character_summary = character_summary
        self.character_profile = self.create_character_profile(character_summary)

    def get_data_path(self, *args):
        return os.path.join(self.base_path, 'data', *args)

    def get_character_path(self, character_name):
        return self.get_data_path('characters', character_name)

    def load_reference_files(self) -> Dict[str, Any]:
        traits_path = os.path.join(self.base_path, 'data', 'reference', 'ref_character_traits.json')
        styles_path = os.path.join(self.base_path, 'data', 'reference', 'ref_communication_styles.json')

        try:
            with open(traits_path, 'r') as traits_file:
                ref_character_traits = json.load(traits_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise FileNotFoundError(f"Error: Failed to load character traits reference file. Please make sure the file exists at {traits_path}. Error: {str(e)}")

        try:
            with open(styles_path, 'r') as comm_styles_file:
                ref_communication_styles = json.load(comm_styles_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise FileNotFoundError(f"Error: Failed to load communication styles reference file. Please make sure the file exists at {styles_path}. Error: {str(e)}")

        return ref_character_traits, ref_communication_styles

    def ensure_character_directory(self) -> str:
        # Ensure the directory for the character exists
        character_dir = self.get_character_path(self.character_name)
        if not os.path.exists(character_dir):
            os.makedirs(character_dir)
        return character_dir

    def sanitize_filename(self, name: str) -> str:
        # Remove invalid characters and limit the length
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
        sanitized = sanitized.strip()
        return sanitized[:50]  # Limit to 50 characters

    def create_character_profile(self, character_summary: str) -> Dict[str, Any]:
        try:
            ref_character_traits, ref_communication_styles = self.load_reference_files()
            character_dir = self.ensure_character_directory()
            profile_path = os.path.join(character_dir, f'{self.character_name}_profile.json')

            # If profile already exists, load it
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as profile_file:
                    return json.load(profile_file)

            # Create a tailored initial profile for Jane
            initial_profile = {
                "characterInfo": {
                    "name": self.character_name,
                    "description": character_summary,
                    "pastSummary": "A seasoned warrior with numerous battles and victories under her belt."
                },
                "cognitiveModel": {
                    "emotionFalloffRate": 0.05,
                    "emotions": {
                        "joy": 0.6,
                        "sadness": 0.3,
                        "anger": 0.5,
                        "fear": 0.2,
                        "disgust": 0.3,
                        "surprise": 0.5,
                        "love": 0.4,
                        "trust": 0.6,
                        "anticipation": 0.7,
                        "curiosity": 0.6,
                        "shame": 0.2,
                        "pride": 0.7,
                        "guilt": 0.3,
                        "envy": 0.2,
                        "gratitude": 0.5,
                        "awe": 0.6,
                        "contempt": 0.3,
                        "anxiety": 0.4,
                        "boredom": 0.2,
                        "confusion": 0.3
                    },
                    "traits": {
                        "openness": 0.6,
                        "conscientiousness": 0.7,
                        "extroversion": 0.6,
                        "agreeableness": 0.5,
                        "neuroticism": 0.3,
                        "confirmationBias": 0.4,
                        "anchoringBias": 0.4,
                        "availabilityHeuristic": 0.5,
                        "dunningKrugerEffect": 0.3,
                        "negativityBias": 0.3,
                        "rewardProcessing": 0.6,
                        "goalOrientedBehavior": 0.8,
                        "intrinsicMotivation": 0.7,
                        "extrinsicMotivation": 0.5,
                        "fatigueLevel": 0.3,
                        "stressLevel": 0.4,
                        "pain": 0.3,
                        "theoryOfMind": 0.6,
                        "empathy": 0.5,
                        "socialCueInterpretation": 0.6,
                        "facialRecognition": 0.6,
                        "emotionalIntelligence": 0.6,
                        "alertness": 0.7,
                        "arousal": 0.6,
                        "selfAwareness": 0.7,
                        "metacognition": 0.6,
                        "visualProcessing": 0.7,
                        "auditoryProcessing": 0.6,
                        "proprioception": 0.7,
                        "vestibularProcessing": 0.7,
                        "planning": 0.7,
                        "organizing": 0.6,
                        "timeManagement": 0.6,
                        "taskInitiation": 0.7,
                        "impulseControl": 0.6,
                        "emotionalRegulation": 0.6,
                        "cognitiveFlexibility": 0.7,
                        "selfMonitoring": 0.6,
                        "attention": 0.7,
                        "workingMemory": 0.6,
                        "longTermMemory": 0.7,
                        "learning": 0.6,
                        "decisionMaking": 0.7,
                        "problemSolving": 0.8,
                        "reasoning": 0.7,
                        "languageProcessing": 0.6,
                        "spatialAwareness": 0.7,
                        "patternRecognition": 0.7,
                        "creativity": 0.6,
                        "courage": 0.8,
                        "resilience": 0.8,
                        "leadership": 0.7
                    }
                },
                "responseTendencies": ref_character_traits["responseTendencies"],
                "communicationStyle": ref_communication_styles["styleGuide"]
            }

            # Customize responseTendencies for Jane
            initial_profile["responseTendencies"]["emotionalTriggers"]["patterns"] = [
                {"trigger": "Threat to her people", "response": "Becomes highly alert and protective, ready to take action"},
                {"trigger": "Praise for her leadership", "response": "Feels motivated to take on more responsibilities"},
                {"trigger": "Unexpected challenges", "response": "Quickly adapts and formulates tactical solutions"}
            ]

            initial_profile["responseTendencies"]["problemSolvingApproach"]["patterns"] = [
                {"situation": "Battle strategy", "approach": "Analyzes terrain and resources, formulates multi-layered plans"},
                {"situation": "Team conflicts", "approach": "Mediates with a firm but fair approach, focusing on group cohesion"},
                {"situation": "Resource scarcity", "approach": "Implements creative solutions and fair rationing"}
            ]

            with open(profile_path, 'w') as profile_file:
                json.dump(initial_profile, profile_file, indent=4)

            return initial_profile
        except FileNotFoundError as e:
            print(f"Error: {str(e)}")
            raise

    def _call_gpt_with_retry(self, messages, retries=3, delay=5):
        for attempt in range(retries):
            try:
                prompt = messages[-1]["content"]
                response = self.gpt_handler.generate_response(prompt, timeout=30)
                # Strip backticks and "json" from the response
                cleaned_response = response.strip('`').lstrip('json\n')
                return cleaned_response
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    raise

    def _validate_profile_structure(self, profile: Dict[str, Any], reference: Dict[str, Any]) -> bool:
        # Validate that the generated profile matches the expected structure of the reference
        for key in reference:
            if key not in profile:
                print(f"Missing key in profile: {key}")
                return False
            if isinstance(reference[key], dict) and isinstance(profile[key], dict):
                if not self._validate_profile_structure(profile[key], reference[key]):
                    return False
            elif isinstance(reference[key], list) and isinstance(profile[key], list):
                if len(reference[key]) != len(profile[key]):
                    print(f"List length mismatch for key: {key}")
                    return False
        return True

    def _create_backup_if_exists(self, file_path: str):
        # Create a backup of the existing profile file if it exists, with versioning to prevent overwriting
        if os.path.exists(file_path):
            backup_version = 1
            backup_path = f"{file_path}.bak.{backup_version}"
            while os.path.exists(backup_path):
                backup_version += 1
                backup_path = f"{file_path}.bak.{backup_version}"
            shutil.copy(file_path, backup_path)

    def update_personality(self, experience: str):
        prompt = f"""
        Given the current character profile:
        {json.dumps(self.character_profile, indent=2)}
        And the new experience: '{experience}', suggest updates to the entire profile.
        Provide your response as a JSON object matching the original profile structure.
        Only include fields that should be updated.
        """

        try:
            response = self._call_gpt_with_retry(
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with updating character profiles based on new experiences."},
                    {"role": "user", "content": prompt}
                ]
            )
            print(f"Cleaned GPT response: {response}")
            updates = json.loads(response)

            # Update the character profile with the new values
            self._update_nested_dict(self.character_profile, updates)

            # Save the updated profile
            profile_path = os.path.join(self.get_character_path(self.character_name.lower()), 'profile.json')
            self._create_backup_if_exists(profile_path)
            with open(profile_path, 'w') as profile_file:
                json.dump(self.character_profile, profile_file, indent=4)

        except json.JSONDecodeError as e:
            print(f"Error decoding GPT response: {str(e)}")
            print(f"Raw response: {response}")
        except Exception as e:
            print(f"Error updating character profile: {str(e)}")

    def _update_nested_dict(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d


# Example usage
if __name__ == "__main__":
    gpt_handler = GPTHandler()
    profile = BicaProfile('Jane Doe', 'A fearless warrior of the north, known for her bravery and tactical skills.', gpt_handler)

    # Initial profile output
    print("Initial Profile:")
    print(json.dumps(profile.character_profile, indent=4))

    # Update personality with a new experience and show the changes
    experience = 'Jane faced a dragon and successfully defended her village, showcasing her leadership and courage.'
    print(f"\nUpdating personality based on experience: '{experience}'")
    profile.update_personality(experience)

    # Updated profile output
    print("\nUpdated Profile:")
    print(json.dumps(profile.character_profile, indent=4))
