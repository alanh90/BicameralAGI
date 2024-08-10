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
        self.memory = BicaMemory(embedding_dim=384)
        self.context = BicaContext()
        self.cognition = BicaCognition(self.memory, self.context)
        self.affect = BicaAffect("BicaAI")  # Assuming we're using a default character named "BicaAI"
        self.actions = BicaActions(BicaUtilities.get_environment_variable("OPENAI_API_KEY"))
        self.safety = BicaSafety()
        self.destiny = BicaDestiny("BicaAI")
        self.gpt_handler = GPTHandler()
        self.logger = BicaLogging("BicaOrchestrator")

    def process_input(self, user_input: str) -> str:
        self.logger.info(f"Processing user input: {user_input}")

        # Update context and memory
        context = self.context.update_context(user_input)
        self.memory.add_memory(user_input, {"importance": 0.7})

        # Generate thoughts and analyze
        thoughts = self.cognition.generate_thoughts(context)
        analysis = self.cognition.analyze(user_input, thoughts)

        # Get emotional state
        emotional_state = self.affect.get_emotional_state()

        # Prepare prompt for GPT
        prompt = self.prepare_gpt_prompt(user_input, context, thoughts, analysis, emotional_state)

        # Define functions for GPT
        functions = self.define_gpt_functions()

        # Generate response using GPT
        gpt_response = self.gpt_handler.generate_response(prompt, functions=functions, function_call="auto")

        # Process GPT response
        if isinstance(gpt_response, dict) and "function_call" in gpt_response:
            # Handle function call
            function_name = gpt_response["function_call"]["name"]
            function_args = json.loads(gpt_response["function_call"]["arguments"])
            response = self.handle_function_call(function_name, function_args)
        else:
            # Regular response
            response = gpt_response

        # Apply safety filter
        safe_response = self.safety.safety_filter(response, "content")

        # Update affect and destiny
        self.affect.update_emotional_state(safe_response)
        self.destiny.update_destiny(safe_response)

        return safe_response

    def prepare_gpt_prompt(self, user_input, context, thoughts, analysis, emotional_state):
        return f"""
        User Input: {user_input}
        Context: {context}
        Thoughts: {thoughts}
        Analysis: {analysis}
        Emotional State: {emotional_state}

        Based on the above information, generate an appropriate response or determine if a function call is necessary.
        """

    def define_gpt_functions(self):
        return [
            {
                "name": "update_memory",
                "description": "Update the AI's memory with important information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The information to be stored in memory"},
                        "importance": {"type": "number", "description": "The importance of the memory (0-1)"}
                    },
                    "required": ["content", "importance"]
                }
            },
            {
                "name": "trigger_emotion",
                "description": "Trigger a specific emotion in the AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "emotion": {"type": "string", "description": "The emotion to trigger"},
                        "intensity": {"type": "number", "description": "The intensity of the emotion (0-1)"}
                    },
                    "required": ["emotion", "intensity"]
                }
            }
        ]

    def handle_function_call(self, function_name, args):
        if function_name == "update_memory":
            self.memory.add_memory(args["content"], {"importance": args["importance"]})
            return f"Memory updated: {args['content']}"
        elif function_name == "trigger_emotion":
            self.affect.trigger_emotion(args["emotion"], args["intensity"])
            return f"Emotion triggered: {args['emotion']} (intensity: {args['intensity']})"
        else:
            return f"Unknown function: {function_name}"


# Example usage
if __name__ == "__main__":
    orchestrator = BicaOrchestrator()
    user_input = "Hello, I'm feeling happy today!"
    response = orchestrator.process_input(user_input)
    print(f"User: {user_input}")
    print(f"AI: {response}")
