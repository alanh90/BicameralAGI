from bica_context import BicaContext
from gpt_handler import GPTHandler
from bica_logging import BicaLogging


class BicaOrchestrator:
    def __init__(self):
        self.logger = BicaLogging("BicaOrchestrator")
        self.context = BicaContext()
        self.gpt_handler = GPTHandler()

    def process_input(self, user_input: str) -> str:
        self.logger.info(f"Processing user input: {user_input}")

        # Update context
        self.context.update_context(user_input)

        # Get current context
        current_context = self.context.get_context()

        # Prepare prompt for GPT
        prompt = self.compile_prompt(user_input, current_context)

        # Generate AI Response
        response = self.gpt_handler.generate_response([{"role": "user", "content": prompt}])

        self.logger.info(f"Generated response: {response}")
        return response

    def compile_prompt(self, user_input, context):
        return f"""
        Previous context: {context}
        User Input: {user_input}

        Based on the above information, generate an appropriate response to the user.
        """
