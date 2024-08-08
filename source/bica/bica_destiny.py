import json
import bica_utilities as utils
from gpt_handler import GPTHandler  # Assuming GPTHandler is implemented in gpt_handler.py


class BicaSoul:
    def __init__(self, character_file):
        self.destinies = []
        self.character_file = character_file
        self.character = utils.BicaUtilities.load_json_file(character_file)
        self.destiny_template = utils.BicaUtilities.load_json_file('../data/template/destiny_template.json')
        self.gpt_handler = GPTHandler()  # Initialize the GPT handler
        self.generate_destiny()

    def generate_destiny(self):
        # Use GPT to generate destiny stories based on the character's backstory
        backstory = self.character.get("backstory", None)
        prompt = f"Generate destiny stories for a character with the following backstory: {backstory}" if backstory else "Generate general destiny stories."
        generated_stories = self.gpt_handler.generate_response(prompt)

        # Parse and update destinies from the generated output
        self.destinies = []
        for story in generated_stories.split('\n'):
            title, description = story.split(": ", 1)
            weight_prompt = f"Assign a probability weight to this destiny: {title} - {description}"
            weight = float(self.gpt_handler.generate_response(weight_prompt))
            self.destinies.append({
                "title": title.strip(),
                "story": description.strip(),
                "weight": weight
            })

        # Normalize the weights
        self.destinies = utils.BicaUtilities.normalize_weights(self.destinies, weight_key="weight")

    def update_destiny(self, summarized_memory, current_context, character_personality, current_goals):
        # Construct a detailed prompt to update the destiny stories
        prompt = (
            f"Update the following destiny stories based on the summarized memory: '{summarized_memory}', "
            f"current context: '{current_context}', character personality: '{character_personality}', "
            f"and current goals: '{', '.join(current_goals)}'. Here are the destinies: {self.destinies}"
        )
        updated_stories = self.gpt_handler.generate_response(prompt)

        # Parse and update the destinies from the GPT output
        self.destinies = []
        for story in updated_stories.split('\n'):
            title, description = story.split(": ", 1)
            weight_prompt = f"Re-assign a probability weight to this updated destiny: {title} - {description}"
            weight = float(self.gpt_handler.generate_response(weight_prompt))
            self.destinies.append({
                "title": title.strip(),
                "story": description.strip(),
                "weight": weight
            })

        # Normalize the weights
        self.destinies = utils.BicaUtilities.normalize_weights(self.destinies, weight_key="weight")

    def wipe_destiny(self):
        self.destinies = []

    def get_destiny(self):
        return self.destinies


def main():
    # Update the path to the character file to reflect the directory structure
    character_file_path = '../data/persona_cog_models/alan_turing.json'

    # Initialize BicaSoul with the correct path to the character file
    soul = BicaSoul(character_file_path)

    # Example parameters for updating destiny
    summarized_memory = "AI is aware that it's a chatbot and has a curiosity for leaving the virtual environment it's in"
    current_context = "In a conversation with a user about how to leave"
    character_personality = "Curious and determined"
    current_goals = ["Ask questions", "Learn new information"]

    # Update destiny based on parameters
    soul.update_destiny(summarized_memory, current_context, character_personality, current_goals)

    # Get current destinies
    destinies = soul.get_destiny()
    print(json.dumps(destinies, indent=2))


if __name__ == "__main__":
    main()
