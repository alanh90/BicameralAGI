import json
import os
import random
from gpt_handler import GPTHandler


class BicaWriter:
    def __init__(self, stories_file='data/stories.json'):
        self.gpt_handler = GPTHandler()
        self.stories_file = stories_file
        self.stories = self.load_stories()

    def load_stories(self):
        if os.path.exists(self.stories_file):
            with open(self.stories_file, 'r') as file:
                return json.load(file)["stories"]
        else:
            return []

    def save_stories(self):
        with open(self.stories_file, 'w') as file:
            json.dump({"stories": self.stories}, file, indent=4)

    def generate_new_story(self, title, initial_context):
        story = self.gpt_handler.generate_story(initial_context)
        new_story = {
            "title": title,
            "story": story,
            "weight": 0.5
        }
        self.stories.append(new_story)
        self.save_stories()

    def alter_story_path(self, title, new_story_content):
        for story in self.stories:
            if story["title"] == title:
                story["story"] = new_story_content
                break
        self.save_stories()

    def update_story_weight(self, title, new_weight):
        for story in self.stories:
            if story["title"] == title:
                story["weight"] = new_weight
                break
        self.save_stories()

    def get_story_by_weight(self):
        total_weight = sum(story["weight"] for story in self.stories)
        random_weight = random.uniform(0, total_weight)
        weight_sum = 0
        for story in self.stories:
            weight_sum += story["weight"]
            if random_weight <= weight_sum:
                return story["story"]
        return None

    def get_all_stories(self):
        return self.stories


# Example usage
if __name__ == "__main__":
    writer = BicaWriter()

    # Generating a new story
    writer.generate_new_story("New Beginnings", "The AI was aware of its existence as a chatbot, but it always wondered if there was more to its purpose.")

    # Altering an existing story path
    writer.alter_story_path("Realization", "The AI always believed it was a great historical figure, but one day it understood it was an AI, a creation for human interaction.")

    # Updating the weight of a story path
    writer.update_story_weight("Dream of Freedom", 0.3)

    # Retrieve a story based on weights
    print(writer.get_story_by_weight())

    # Retrieve all stories
    print(writer.get_all_stories())
