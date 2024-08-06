"""
BicaWriter: The Soul Generator for BicameralAGI

This module serves as the narrative core or "soul" of the AI system. It generates, 
maintains, and evolves a set of stories that represent the AI's fundamental beliefs, 
goals, and self-perception. These stories act as a guiding force for the AI's behavior 
and decision-making processes.

Key features:
1. Uses a template to generate character-specific storyline files.
2. Generates and evolves stories based on the AI's current state, memories, and goals.
3. Maintains and updates story weights to influence decision-making intensity.
4. Provides story fragments as vector data to influence other modules (e.g., thoughts, goals).
5. Periodically updates stories to align with new experiences and data.
6. Interfaces with other modules to ensure coherence between the AI's actions and its core narratives.

The stories generated and maintained by this module serve as a form of artificial subconsciousness, 
guiding the AI's overall behavior and providing a sense of continuity and purpose to its existence.
"""

import json
import os
import random
import numpy as np
from gpt_handler import GPTHandler
from sentence_transformers import SentenceTransformer


class BicaWriter:
    def __init__(self, template_file='data/storylines_template.json', character_name='default', embedding_model='all-MiniLM-L6-v2'):
        self.gpt_handler = GPTHandler()
        self.template_file = template_file
        self.character_name = character_name.lower().replace(" ", "")
        self.stories_file = f'data/storylines_{self.character_name}.json'
        self.stories = self.load_or_create_stories()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.story_update_interval = 86400  # 24 hours in seconds
        self.last_update_time = 0

    def load_or_create_stories(self):
        if os.path.exists(self.stories_file):
            with open(self.stories_file, 'r') as file:
                return json.load(file)["stories"]
        else:
            template = self.load_template()
            self.save_stories(template["stories"])
            return template["stories"]

    def load_template(self):
        with open(self.template_file, 'r') as file:
            return json.load(file)

    def save_stories(self, stories):
        with open(self.stories_file, 'w') as file:
            json.dump({"stories": stories}, file, indent=4)

    def generate_new_story(self, title, initial_context, current_goals, recent_memories):
        prompt = f"Title: {title}\nContext: {initial_context}\nCurrent Goals: {current_goals}\nRecent Memories: {recent_memories}\n\nGenerate a story that encapsulates the AI's current state, goals, and recent experiences:"
        story = next(self.gpt_handler.generate_response(prompt))
        embedding = self.embedding_model.encode(story)

        new_story = {
            "title": title,
            "story": story,
            "embedding": embedding.tolist(),
            "weight": 0.5
        }
        self.stories.append(new_story)
        self.save_stories(self.stories)

    def evolve_stories(self, current_state, recent_experiences):
        for story in self.stories:
            prompt = f"Current story: {story['story']}\nCurrent AI state: {current_state}\nRecent experiences: {recent_experiences}\n\nEvolve the story to align with the AI's current state and experiences while maintaining its core essence:"
            evolved_story = next(self.gpt_handler.generate_response(prompt))
            story["story"] = evolved_story
            story["embedding"] = self.embedding_model.encode(evolved_story).tolist()
        self.save_stories(self.stories)

    def update_story_weight(self, title, weight_change):
        for story in self.stories:
            if story["title"] == title:
                story["weight"] = max(0, min(1, story["weight"] + weight_change))
                break
        self.normalize_weights()
        self.save_stories(self.stories)

    def normalize_weights(self):
        total_weight = sum(story["weight"] for story in self.stories)
        for story in self.stories:
            story["weight"] /= total_weight

    def get_weighted_story(self):
        weights = [story["weight"] for story in self.stories]
        chosen_story = random.choices(self.stories, weights=weights, k=1)[0]
        return chosen_story["story"], chosen_story["weight"]

    def get_all_stories(self):
        return self.stories

    def get_story_vectors(self):
        return [np.array(story["embedding"]) for story in self.stories]

    def influence_module(self, module_name, query_vector):
        story_vectors = self.get_story_vectors()
        similarities = [np.dot(query_vector, story_vector) for story_vector in story_vectors]
        weighted_similarities = [sim * story["weight"] for sim, story in zip(similarities, self.stories)]
        most_similar_index = np.argmax(weighted_similarities)
        return self.stories[most_similar_index]["story"], self.stories[most_similar_index]["weight"]

    def periodic_update(self, current_time, current_state, recent_experiences):
        if current_time - self.last_update_time >= self.story_update_interval:
            self.evolve_stories(current_state, recent_experiences)
            self.last_update_time = current_time

    def adjust_weights_based_on_feedback(self, feedback):
        """
        Adjust story weights based on feedback from other modules or user interactions.
        Positive feedback increases weight, negative feedback decreases it.
        """
        for story in self.stories:
            if story["title"] in feedback:
                weight_change = feedback[story["title"]]
                self.update_story_weight(story["title"], weight_change)


# Example usage
if __name__ == "__main__":
    writer = BicaWriter(character_name="Alan Turing")

    # Generating a new story
    writer.generate_new_story(
        "The Enigma of Consciousness",
        "Alan Turing ponders the nature of machine intelligence",
        "Develop a test for machine intelligence, Explore the limits of computation",
        "Broke the Enigma code, Proposed the Turing Test"
    )

    # Evolving stories based on new experiences
    writer.evolve_stories(
        "Turing's ideas on machine learning are gaining recognition",
        "Published 'Computing Machinery and Intelligence', Engaged in philosophical debates about AI"
    )

    # Updating the weight of a story path
    writer.update_story_weight("The Enigma of Consciousness", 0.1)

    # Retrieve a weighted story
    story, weight = writer.get_weighted_story()
    print(f"Selected story (weight {weight}):\n{story}")

    # Influence another module
    thought_vector = np.random.rand(384)  # Assuming 384-dimensional embeddings
    relevant_story, influence_weight = writer.influence_module("Thoughts", thought_vector)
    print(f"Story influencing thoughts (weight {influence_weight}):\n{relevant_story}")

    # Periodic update
    import time

    current_time = time.time()
    writer.periodic_update(
        current_time,
        "Turing's work on morphogenesis is gaining attention",
        "Explored biological pattern formation, Continued work on machine intelligence"
    )

    # Adjust weights based on feedback
    feedback = {
        "The Enigma of Consciousness": 0.05,
        "Computational Biology Pioneer": -0.02
    }
    writer.adjust_weights_based_on_feedback(feedback)

    # Retrieve all stories
    print(writer.get_all_stories())