"""
This module manages and executes the various actions that the BicameralAGI system can take.
It handles action execution and serves as a flexible compiler of information for GPT responses.
"""

from typing import Dict, Any
from external.gpt_handler import GPTHandler


class BicaActionExecutor:
    def __init__(self):
        self.gpt_handler = GPTHandler()

    def execute_action(self, action: str, context: Dict[str, Any] = None, compiled_data: Dict[str, Any] = None):
        if action == 'respond':
            return self.execute_response(context, compiled_data)
        elif action == "play_audio":
            return self.execute_play_audio(context)
        elif action == "move_robot":
            self.execute_move_robot(context)
        else:
            print(f'There is no functioning action match for the following "{action}". Calling default execute response function instead.')
            return self.execute_response(context, compiled_data)

    def execute_response(self, context: Dict[str, Any] = None, compiled_data: Dict[str, Any] = None) -> str:
        """
        Reason for defining a separate compiled data section is for a possible expansion later
        where we separate parts of the context for different reasons
        :param context:
        :param compiled_data:
        :return:
        """
        if compiled_data is None:
            compiled_data = context.get("compiled_data", {}) if context else {}

        response = self.gpt_handler.generate_response("", compiled_data=compiled_data)

        return response

    # Placeholder methods for future action executions
    def execute_play_audio(self, context: Dict[str, Any]):
        pass

    def execute_move_robot(self, context: Dict[str, Any]):
        pass
