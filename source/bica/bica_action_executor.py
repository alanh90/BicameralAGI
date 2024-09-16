"""
This module manages and executes the various actions that the BicameralAGI system can take.
It handles action execution and serves as a flexible compiler of information for GPT responses.
"""

from typing import Dict, Any
from gpt_handler import GPTHandler


class BicaActionExecutor:
    def __init__(self):
        self.gpt_handler = GPTHandler()

    def execute_action(self, action: str, context):
        match action:
            case 'respond':
                return self.execute_response(context)
            case "play_audio":
                return self.execute_play_audio(context)
            case "move_robot":
                self.execute_move_robot(context)
            case _:
                return {"error": f"Action '{action}' is not available."}

    def execute_response(self, compiled_context: Dict[str, Any]) -> str:
        # Reason for defining a separate compiled data section is for a possible expansion later
        # where we separate parts of the context for different reasons
        compiled_data = compiled_context.get("compiled_data", {})
        response = self.gpt_handler.generate_response([{"role": "user", "content": compiled_data}])

        return response

    # Placeholder methods for future action executions
    def execute_play_audio(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def execute_move_robot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass
