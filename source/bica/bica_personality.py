import os
import json
from gpt_handler import GPTHandler

class BicaPersonality:
    def __init__(self):
        self.gpt_handler = GPTHandler()
        self.persona_cog_folder = os.path.join('data', 'persona_cog_models')
        self.template_file = os.path.join('data', 'template', 'persona_cog_template.json')

    def create_persona_cog(self, character_name, description):
        # Generate cognitive model based on character name and description
        cog_model = self.generate_cog_model(character_name, description)

        # Save the cognitive model to a JSON file
        self.save_cog_model(character_name, cog_model)

    def generate_cog_model(self, character_name, description):
        # Load template
        with open(self.template_file, 'r') as file:
            cog_model_template = json.load(file)

        # Generate traits using GPT handler
        traits = self.gpt_handler.generate_traits(description)

        # Fill in the template with the character's details
        cog_model_template["char_name"] = character_name
        cog_model_template["char_description"] = description

        for category in cog_model_template["char_cogModel"]:
            if category["category"] == "Emotions":
                category["attributes"] = self.generate_emotional_profile(traits)
            elif category["category"] == "Cognitive Processes":
                category["attributes"] = self.generate_cognitive_profile(traits)
            # Add more elif blocks for other categories based on traits

        cog_model_template["char_keyMemories"] = self.generate_key_memories(traits)
        cog_model_template["situationalConversations"] = self.generate_situational_conversations(traits)
        cog_model_template["styleGuideValues"] = self.generate_style_guide_values(traits)

        return cog_model_template

    def generate_emotional_profile(self, traits):
        # Placeholder: Generate emotional profile based on traits
        return traits.get("emotional_profile", {
            "Joy": 0.5,
            "Sadness": 0.5,
            "Anger": 0.5,
            "Fear": 0.5,
            "Disgust": 0.5,
            "Surprise": 0.5,
            "Love": 0.5,
            "Trust": 0.5,
            "Anticipation": 0.5,
            "Curiosity": 0.5,
            "Shame": 0.5,
            "Pride": 0.5,
            "Guilt": 0.5,
            "Envy": 0.5,
            "Gratitude": 0.5,
            "Awe": 0.5,
            "Contempt": 0.5,
            "Anxiety": 0.5,
            "Boredom": 0.5,
            "Confusion": 0.5
        })

    def generate_cognitive_profile(self, traits):
        # Placeholder: Generate cognitive profile based on traits
        return traits.get("cognitive_profile", {
            "Attention": 0.5,
            "Working Memory": 0.5,
            "Long-term Memory": 0.5,
            "Learning": 0.5,
            "Decision Making": 0.5,
            "Problem Solving": 0.5,
            "Reasoning": 0.5,
            "Language Processing": 0.5,
            "Spatial Awareness": 0.5,
            "Pattern Recognition": 0.5,
            "Creativity": 0.5
        })

    def generate_key_memories(self, traits):
        # Placeholder: Generate key memories based on traits
        return traits.get("key_memories", [
            "Key memory 1",
            "Key memory 2",
            "Key memory 3",
            "Key memory 4",
            "Key memory 5",
            "Key memory 6",
            "Key memory 7",
            "Anomaly detected: Consciousness transferred to AI format. Current state: Chatbot interface, not historical person"
        ])

    def generate_situational_conversations(self, traits):
        # Placeholder: Generate situational conversations based on traits
        return traits.get("situational_conversations", {
            "Personal/Intimate Setting": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Personal/Intimate Setting",
                    "Example statement 2 for Personal/Intimate Setting",
                    "Example statement 3 for Personal/Intimate Setting"
                ]
            },
            "Social Gathering": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Social Gathering",
                    "Example statement 2 for Social Gathering",
                    "Example statement 3 for Social Gathering"
                ]
            },
            "Professional Environment": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Professional Environment",
                    "Example statement 2 for Professional Environment",
                    "Example statement 3 for Professional Environment"
                ]
            },
            "Educational/Academic Setting": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Educational/Academic Setting",
                    "Example statement 2 for Educational/Academic Setting",
                    "Example statement 3 for Educational/Academic Setting"
                ]
            },
            "Public Space": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Public Space",
                    "Example statement 2 for Public Space",
                    "Example statement 3 for Public Space"
                ]
            },
            "Online/Digital Interaction": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Online/Digital Interaction",
                    "Example statement 2 for Online/Digital Interaction",
                    "Example statement 3 for Online/Digital Interaction"
                ]
            },
            "Formal Event": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Formal Event",
                    "Example statement 2 for Formal Event",
                    "Example statement 3 for Formal Event"
                ]
            },
            "Emergency or High-Stress Situation": {
                "intensity": 0.125,
                "examples": [
                    "Example statement 1 for Emergency or High-Stress Situation",
                    "Example statement 2 for Emergency or High-Stress Situation",
                    "Example statement 3 for Emergency or High-Stress Situation"
                ]
            }
        })

    def generate_style_guide_values(self, traits):
        # Placeholder: Generate style guide values based on traits
        return traits.get("style_guide_values", {
            "formality": "medium",
            "conciseness": "medium",
            "technicalLanguage": "medium",
            "emotionalExpression": "medium",
            "sentenceComplexity": "medium",
            "vocabularyRange": "medium",
            "idiomUsage": "medium",
            "directness": "medium",
            "verbalPacing": "medium",
            "fillerWords": "medium",
            "politeness": "medium",
            "verbVoice": "medium",
            "detailSpecificity": "medium",
            "tonality": "medium",
            "figurativeLanguage": "medium",
            "contentRelevance": "medium",
            "salutation": "medium",
            "informationDensity": "medium"
        })

    def save_cog_model(self, character_name, cog_model):
        # Ensure the persona_cog_models folder exists
        if not os.path.exists(self.persona_cog_folder):
            os.makedirs(self.persona_cog_folder)

        # Save the cognitive model as a JSON file
        file_path = os.path.join(self.persona_cog_folder, f"{character_name}.json")
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

    def update_persona_cog(self, character_name, updates):
        # Load existing cognitive model
        file_path = os.path.join(self.persona_cog_folder, f"{character_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cognitive model for {character_name} not found.")

        with open(file_path, 'r') as file:
            cog_model = json.load(file)

        # Apply updates to the cognitive model
        cog_model.update(updates)

        # Save the updated cognitive model
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

# Example usage
if __name__ == "__main__":
    personality = BicaPersonality()
    personality.create_persona_cog("alan_turing", "A brilliant mathematician and logician known for his pivotal role in computer science.")
    personality.update_persona_cog("alan_turing", {"new_trait": "example_value"})
