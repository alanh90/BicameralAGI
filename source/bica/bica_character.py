"""
BicameralAGI Character Module
================================

Overview:
---------
This module serves as the central coordinator for the BicameralAGI system. It is responsible for initializing and managing
all core components, orchestrating the flow of information between them, and handling the overall processing of user inputs
and system responses. The BicaCharacter class integrates key features like context, memory, and action execution to generate
cohesive, human-like AI behavior.

Current Features:
-----------------
1. **Character Definition**: Automatically generates or updates a character's name and summary based on a given description.
2. **Context Management**: Tracks and updates conversation context for more informed responses.
3. **Prompt Compilation**: Constructs dynamic prompts based on the current system state (context, user input, etc.).
4. **Action Execution**: Handles executing responses based on user input and the processed context.

Usage Example:
--------------
    character = BicaCharacter("A brave knight from the future.")
    response = character.process_input("What is your mission?")
    print(response)

Author:
-------
Alan Hourmand
Date: 10/2/2024
"""

from bica_action_executor import BicaActionExecutor
from bica_context import BicaContext
from gpt_handler import GPTHandler
from bica_profile import BicaProfile
from bica_utilities import *


class BicaCharacter:
    def __init__(self, character_description: str):
        self.action_executor = BicaActionExecutor()
        self.gpt_handler = GPTHandler()

        # ||||||||| BICA AGI COGNITIVE SETUP ||||||||||
        self.character_name = "BICA AGI"
        self.character_summary = "You are an artificial general intelligence called BICA. You were created by Alan Hourmand."
        self.extract_character_definition(character_description)

        self.profile = BicaProfile(self.character_summary, self.character_name)
        self.context = BicaContext()
        # |||||||||||||||||||||||||||||||||||||||||||||

    def get_character_definition(self):
        return self.character_summary

    def extract_character_definition(self, character_description: str):
        """
        Generates or updates the character's name and summary based on the provided description.
        If no name is provided, one is generated.
        """
        prompt = f"""
        Based on the following short description, try to figure out what character the user is referring to. If a name is not provided, generate one that best fits the description.
        
        Also if the description the user gives you seems like traits, just make up a character that best fits the traits.

        Description: {character_description}

        Respond in the format:
        {{
            "name": "Character's name",
            "summary": "You are {{name}}, [brief character summary]."
        }}
        """

        try:
            # Generate the response using GPT
            response = self.gpt_handler.generate_response(prompt)

            # Remove the backticks if present around the JSON
            cleaned_response = response.strip("```json").strip("```").strip()

            # Parse the cleaned JSON response
            character_info = json.loads(cleaned_response)

            # Extract the name and summary from the response
            self.character_name = character_info.get("name", "Unknown Character")
            self.character_summary = character_info.get(
                "summary", f"You are {self.character_name}, a mysterious figure."
            )
        except json.JSONDecodeError:
            # Fallback if GPT response is not in proper JSON format
            print("Error: Could not parse the character definition from the AI response.")
            print(f"Fallback raw response: {response}")  # Debugging the raw response
            self.character_name = "Unknown Character"
            self.character_summary = f"You are {self.character_name}, an enigmatic character."

    def process_input(self, user_input: str) -> str:
        try:
            self.context.update_context(user_input)
            updated_context = self.context.get_context()

            # Gather context data
            context_data = {
                "user_input": user_input,
                "system_prompt": self.get_character_definition(),
                "updated_context": updated_context  # Add updated context to the prompt data
            }

            compiled_data = self.compile_prompt(context_data)

            response = self.action_executor.execute_action("respond", {"compiled_data": compiled_data})
            print(f"Generated response: {response}")

            return response

        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return "I apologize, but I encountered an error. Could you please try again?"

    def compile_prompt(self, compiled_data: dict) -> str:
        print("Compiling the prompt...")
        prompt_parts = []
        for key, value in compiled_data.items():
            if value:  # Only include non-empty values
                if isinstance(value, dict):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for sub_key, sub_value in value.items():
                        prompt_parts.append(f"  {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for item in value:
                        prompt_parts.append(f"  - {item}")
                else:
                    prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        prompt = "\n".join(prompt_parts)
        prompt += "\n\nBased on the above context information, generate a final response to the user."
        print(f"Compiled prompt:\n{prompt}")
        return prompt
