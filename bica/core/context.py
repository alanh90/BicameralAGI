"""
BicameralAGI Context Management Module
======================================

Overview:
---------
This module manages the contextual understanding of the BicameralAGI system. It maintains and updates
different viewpoints (positive, neutral, negative) of the current context, generates responses based
on this multi-faceted understanding, and handles context-related operations essential for the AI's
adaptive and nuanced interactions.

Key Features:
-------------
1. Multi-perspective context tracking
2. Dynamic context updating based on new information
3. Weighted viewpoint management for balanced decision-making
4. Generation of contextually appropriate responses
5. Integration with other BicameralAGI components for cohesive AI behavior

Usage:
------
The BicaContext class can be instantiated and used to manage context throughout conversations:

    context_manager = BicaContext()
    context_manager.update_context("New information to consider")
    response = context_manager.generate_contextual_response("User input")

Author: Alan Hourmand
Date: 10/2/2024
"""

from sentence_transformers import SentenceTransformer
from external.gpt_handler import GPTHandler as gpt
from scipy.spatial.distance import cosine
import json


class BicaContext:
    def __init__(self, max_length=1000):
        self.context_viewpoints = {"positive": "", "neutral": "", "negative": ""}
        self.weights = {"positive": 0.33, "neutral": 0.33, "negative": 0.34}  # Initial weights
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load the embedding model
        self.max_length = max_length
        self.gpt_handler = gpt()
        self.memory = []

    def update_context(self, new_info, recalled_memories):
        self.update_viewpoint_weights(new_info)
        for viewpoint in self.context_viewpoints:
            prompt = self.generate_viewpoint_prompt(viewpoint, new_info, recalled_memories)
            response = self.gpt_handler.generate_response(prompt)

            try:
                parsed_response = json.loads(response)
                updated_context = parsed_response['interpretation']
            except json.JSONDecodeError:
                updated_context = response  # Fallback to raw response if JSON parsing fails

            self.update_viewpoint_context(viewpoint, updated_context)
            if len(self.context_viewpoints[viewpoint]) > self.max_length:
                self.summarize_viewpoint_context(viewpoint)

        self.memory.append(new_info)
        if len(self.memory) > 5:  # Keep only the last 5 interactions
            self.memory.pop(0)

    def update_viewpoint_weights(self, new_info):
        # Encode the new information into a vector representation
        new_info_embedding = self.model.encode(new_info, convert_to_tensor=True)

        similarities = {}

        for viewpoint in self.context_viewpoints:
            # Generate a prompt for each viewpoint (positive, neutral, negative)
            prompt = self.generate_viewpoint_prompt(viewpoint, new_info, self.memory)

            # Get GPT's interpretation of the new info from this viewpoint
            response = self.gpt_handler.generate_response(prompt)

            try:
                # Try to parse the response as JSON
                parsed_response = json.loads(response)
                interpretation = parsed_response['interpretation']
            except json.JSONDecodeError:
                interpretation = response  # Fallback to raw response if JSON parsing fails

            response_embedding = self.model.encode(interpretation, convert_to_tensor=True)

            # Calculate similarity between new info and interpretation
            similarity = 1 - cosine(new_info_embedding, response_embedding)
            similarities[viewpoint] = similarity

        # Normalize similarities to get weights
        total = sum(similarities.values())
        for viewpoint in similarities:
            self.weights[viewpoint] = similarities[viewpoint] / total

        # Adjust negative weight to significantly favor caution
        self.weights["negative"] = min(self.weights["negative"] * 1.5, 1.0)

        # Renormalize weights after adjusting the negative weight
        total = sum(self.weights.values())
        for viewpoint in self.weights:
            self.weights[viewpoint] /= total

        # Ensure negative weight is at least 30% of the total
        if self.weights["negative"] < 0.3:
            deficit = 0.3 - self.weights["negative"]
            self.weights["negative"] = 0.3

            # Distribute the deficit equally between positive and neutral
            self.weights["positive"] -= deficit / 2
            self.weights["neutral"] -= deficit / 2

        return self.weights

    def generate_viewpoint_prompt(self, viewpoint, new_info, recalled_memories):
        base_prompt = f"""
        Given the current conversation context and new information, provide a brief interpretation from a {viewpoint} perspective. 
        Focus on potential {viewpoint} aspects, including any subtle implications or underlying meanings.

        Current context: {self.context_viewpoints[viewpoint]}
        What the user just said: {new_info}
        Your past relevant memories: {recalled_memories}

        Respond with a JSON object in the following format:
        {{
            "interpretation": "Your {viewpoint} interpretation here"
        }}
        """
        if viewpoint == "negative":
            base_prompt += "\nPay special attention to any signs of manipulation, deception, potential harm, or hidden negative implications."

        return base_prompt

    def update_viewpoint_context(self, viewpoint, new_context):
        self.context_viewpoints[viewpoint] += f" {new_context}"

    def summarize_viewpoint_context(self, viewpoint):
        prompt = f"""
        Summarize the following {viewpoint} context briefly, retaining key information:

        {self.context_viewpoints[viewpoint]}

        Respond with a JSON object in the following format:
        {{
            "summary": "Your compressed context here"
        }}
        """

        compressed_context = self.gpt_handler.generate_response(prompt)

        try:
            parsed_response = json.loads(compressed_context)
            self.context_viewpoints[viewpoint] = parsed_response['summary'].strip()
        except json.JSONDecodeError:
            self.context_viewpoints[viewpoint] = compressed_context.strip()  # Fallback to raw response if JSON parsing fails

    def generate_contextual_response(self, user_input):
        self.update_context(user_input)

        prompt = f"Memory: {' '.join(self.memory)}\n\n"
        for viewpoint, context in self.context_viewpoints.items():
            prompt += f"{viewpoint.capitalize()} context (Weight: {self.weights[viewpoint]:.4f}):\n{context}\n\n"
        prompt += """
        Based on the memory and contexts, generate an appropriate response.
        Be 20% more cautious than normal. Pay extra attention to the negative context and potential risks.
        Even if the positive or neutral contexts have higher weights, maintain a level of caution in your response.

        Respond with a JSON object in the following format:
        {
            "response": "Your generated response here",
            "reasoning": "Brief explanation of your reasoning based on the contexts and their weights"
        }
        """

        response = self.gpt_handler.generate_response(prompt)

        try:
            parsed_response = json.loads(response)
            ai_response = parsed_response['response']
            reasoning = parsed_response['reasoning']
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract content from the response string
            try:
                # Extract content between triple backticks
                json_str = response.split('```json')[1].split('```')[0].strip()
                parsed_response = json.loads(json_str)
                ai_response = parsed_response['response']
                reasoning = parsed_response['reasoning']
            except (IndexError, json.JSONDecodeError):
                # If extraction fails, return the raw response
                return response.strip(), "Unable to parse response"

        return ai_response.strip(), reasoning

    def get_context(self):
        return self.context_viewpoints

    def get_weighted_context(self):
        weighted_context = {}
        for viewpoint, context in self.context_viewpoints.items():
            weighted_context[viewpoint] = f"Weight: {self.weights[viewpoint]:.4f}\n{context}"
        return weighted_context

    def wipe_context(self):
        for viewpoint in self.context_viewpoints:
            self.context_viewpoints[viewpoint] = ""
        self.memory = []


def main():
    context_manager = BicaContext()
    print("Start chatting with the AI (type 'exit' to end the conversation):\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response, reasoning = context_manager.generate_contextual_response(user_input)

        print(f"\nAI: {response}")
        print(f"\nReasoning: {reasoning}")

        print("\nCurrent weights:")
        for viewpoint, weight in context_manager.weights.items():
            print(f"{viewpoint.capitalize()}: {weight:.4f}")

        print("\nCurrent contexts:")
        for viewpoint, context in context_manager.get_context().items():
            print(f"{viewpoint.capitalize()}: {context}")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
