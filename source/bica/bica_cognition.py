"""
This module handles the cognitive processes of the BicameralAGI system. It generates thoughts, analyzes inputs, and manages both conscious and subconscious cognitive functions using various AI techniques including GPT models and vector embeddings.
"""

import random
import re
import numpy as np
from typing import List, Dict, Any
from bica.gpt_handler import GPTHandler
from bica_memory import BicaMemory
from bica_context import BicaContext
from bica_utilities import BicaUtilities
from sentence_transformers import SentenceTransformer


class BicaCognition:
    def __init__(self, memory: BicaMemory, context: BicaContext):
        self.memory = memory
        self.context = context
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.current_thoughts = []
        self.subconscious_thoughts = []
        self.noise_dimension = 1000  # Dimension for noise vector
        self.valid_emotions = memory.get_valid_emotions()

    def _format_prompt(self, content: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": content}
        ]

    def generate_thoughts(self, context: str, goals: List[str]) -> List[str]:
        subconscious_thoughts = self.generate_subconscious_thoughts(context)

        prompt = f"""Given the following context and goals, generate thoughts. Also consider these subconscious influences: {subconscious_thoughts}

        Context: {context}
        Goals: {', '.join(goals)}

        Generate 5 thoughts:
        """
        thoughts = self.gpt_handler.generate_response(self._format_prompt(prompt))

        self.current_thoughts = thoughts.strip().split('\n')

        valid_emotions = self.memory.get_valid_emotions()
        default_emotions = {emotion: random.uniform(0.1, 0.9) for emotion in random.sample(valid_emotions, min(3, len(valid_emotions)))}

        self.memory.save_memory(' '.join(self.current_thoughts), emotions=default_emotions, importance=0.5)

        return self.current_thoughts

    def generate_subconscious_thoughts(self, context: str) -> List[str]:
        print("Generating noise...")
        noise = np.random.randn(self.noise_dimension)

        print("Extracting words and phrases...")
        words_phrases = self._extract_words_phrases(noise, context)
        print(f"Extracted {len(words_phrases)} words/phrases")

        print("Generating potential future scenarios...")
        prompt = f"""Based on the current context and these extracted words/phrases, generate 5 brief potential future scenarios or outcomes for this AI agent:

        Context: {context}
        Words/Phrases: {', '.join(words_phrases[:10])}  # Using only first 10 for brevity

        Each scenario should be a single sentence describing a possible future state or action of the AI.
        """
        scenarios = self.gpt_handler.generate_response(self._format_prompt(prompt)).strip().split('\n')
        print(f"Generated {len(scenarios)} scenarios")

        print("Evaluating scenarios...")
        evaluated_scenarios = self._evaluate_thoughts(scenarios, context)
        print(f"Evaluated {len(evaluated_scenarios)} scenarios")

        return evaluated_scenarios[:min(5, len(evaluated_scenarios))]

    def _extract_words_phrases(self, noise: np.ndarray, context: str) -> List[str]:
        context_embedding = self.sentence_model.encode([context])

        memories = self.memory.get_recent_memories(5)

        if memories:
            memory_embeddings = self.sentence_model.encode(memories)
            combined_embedding = np.mean([context_embedding] + [memory_embeddings], axis=0)
        else:
            combined_embedding = context_embedding

        influenced_embedding = combined_embedding.flatten() + 0.1 * noise[:combined_embedding.size]

        prompt = f"Generate 20 random words or short phrases based on this numeric representation: {influenced_embedding[:10]}..."
        words_phrases = self.gpt_handler.generate_response(self._format_prompt(prompt)).strip().split('\n')
        print(f"Received subconscious( {len(words_phrases)} ) words/phrases from GPT")
        return words_phrases

    def _evaluate_thoughts(self, words_phrases: List[str], context: str) -> List[str]:
        scenarios = self.generate_scenarios(words_phrases, context)

        evaluated_thoughts = []
        for scenario in scenarios:
            prompt = f"""Evaluate this scenario in the current context:
            Scenario: {scenario}
            Context: {context}

            Consider:
            1. Does it show a possible risk?
            2. Does it show a possible benefit?
            3. Does it present new ideas or conversation points?
            4. Could it solve a current problem?

            Rate its relevance from 0 to 1, and explain your rating.

            Provide your response in the following format:
            Rating: [Your rating between 0 and 1]
            Explanation: [Your explanation]
            """
            response = self.gpt_handler.generate_response(self._format_prompt(prompt))

            # Extract the rating from the response
            match = re.search(r'Rating:\s*(0?\.\d+|[01])', response)
            if match:
                relevance = float(match.group(1))
            else:
                relevance = 0.5  # Default value if no rating is found

            evaluated_thoughts.append((relevance, scenario))

        return [thought for _, thought in sorted(evaluated_thoughts, reverse=True)]

    def generate_scenarios(self, words_phrases: List[str], context: str) -> List[str]:
        prompt = f"""Generate 5 brief future scenarios using some of these words/phrases and considering the context:
        Words/Phrases: {', '.join(words_phrases)}
        Context: {context}

        Scenarios:
        """
        response = self.gpt_handler.generate_response(self._format_prompt(prompt))
        scenarios = response.strip().split('\n')
        return [s.strip() for s in scenarios if s.strip()]  # Remove empty lines

    def deep_thought(self, topic: str) -> str:
        prompt = f"""Engage in deep thought about this topic: {topic}
        Consider your current thoughts: {self.current_thoughts}
        And these subconscious thoughts: {self.subconscious_thoughts}

        Provide a detailed analysis:
        """
        analysis = self.gpt_handler.generate_response(self._format_prompt(prompt))
        return analysis

    def get_all_current_thoughts(self) -> Dict[str, List[str]]:
        return {
            "conscious_thoughts": self.current_thoughts,
            "subconscious_thoughts": self.subconscious_thoughts
        }


# Testing
def test_bica_cognition_interactive():
    memory = BicaMemory()
    context = BicaContext()
    cognition = BicaCognition(memory, context)

    print("Welcome to the BicaCognition interactive test.")
    print("Type 'exit' at any time to end the conversation.")

    current_context = "You are an AI assistant that is being tested for your reasoning abilities."
    goals = ["You are driving and a bottle starts rolling around on the ground. Your buddy in the car is sleeping."]

    while True:
        print("\n" + "=" * 50)
        print("AI is thinking...")
        print(f"Current context: {current_context}")
        print(f"Current goals: {goals}")

        # Generate conscious thoughts
        print("\nGenerating thoughts:")
        thoughts = cognition.generate_thoughts(current_context, goals)
        for i, thought in enumerate(thoughts, 1):
            print(f"{i}. {thought}")

        # User input for next iteration
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # Update context and goals based on user input
        current_context = f"Continuing conversation. User said: {user_input}"
        goals.append(f"Address user's input: {user_input}")
        if len(goals) > 3:
            goals = goals[-3:]  # Keep only the 3 most recent goals

    print("Thank you for testing BicaCognition!")


if __name__ == "__main__":
    test_bica_cognition_interactive()