from bica_memory import BicaMemory
from bica_context import BicaContext
from bica_cognition import BicaCognition
from bica_affect import BicaAffect
from bica_action import BicaActions
from bica_safety import BicaSafety
from bica_destiny import BicaDestiny
from gpt_handler import GPTHandler
from bica_logging import BicaLogging
from bica_utilities import BicaUtilities
import json


class BicaOrchestrator:
    def __init__(self):
        self.actions = BicaActions(BicaUtilities.get_environment_variable("OPENAI_API_KEY"))
        self.affect = BicaAffect("BicaAI")  # Assuming we're using a default character named "BicaAI"
        self.cognition = BicaCognition(self.memory, self.context)
        self.context = BicaContext()
        self.destiny = BicaDestiny("BicaAI")
        self.logger = BicaLogging("BicaOrchestrator")
        self.memory = BicaMemory()
        self.safety = BicaSafety()
        self.gpt_handler = GPTHandler()

    def process_input(self, user_input: str) -> str:
        # Update context and memory
        context = self.context.update_context(user_input)
        self.memory.add_memory(user_input, {"importance": 0.7})

        # Generate thoughts and analyze
        thoughts = self.cognition.generate_thoughts(context)

        # Get emotional state
        emotional_state = self.affect.update_emotions(context)

        # Prepare prompt for GPT
        _prompt = self.compile_prompt(user_input, context, thoughts, emotional_state)

        # AI Response
        response = self.actions.execute_response()

        # Apply safety filter
        safe_response = self.safety.safety_filter(response, "content")

        # Update destiny
        self.destiny.alter_destiny(context)

        return safe_response

    def compile_prompt(self, user_input, context, thoughts, analysis, emotional_state):
        return f"""
        User Input: {user_input}
        Context: {context}
        Thoughts: {thoughts}
        Analysis: {analysis}
        Emotional State: {emotional_state}

        Based on the above information, generate an appropriate response or determine if a function call is necessary.
        """
