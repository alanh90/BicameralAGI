"""
This module manages and executes the various actions that the BicameralAGI system can take. It handles action selection, execution, and weight updates based on feedback and system performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Callable
import json
from pathlib import Path
from openai import OpenAI
from bica_logging import BicaLogging


@dataclass
class ActionProperties:
    description: str
    weight: float
    function: str


class BicaActions:
    def __init__(self, api_key: str):
        self.actions_file = Path(__file__).resolve().parents[2] / "source" / "data" / "definitions" / "possible_actions.json"
        self.actions: Dict[str, ActionProperties] = self._load_actions_from_file()
        self.client = OpenAI(api_key=api_key)
        self.logger = BicaLogging("BicaActions")

    def _load_actions_from_file(self) -> Dict[str, ActionProperties]:
        try:
            with open(self.actions_file, 'r') as f:
                actions_data = json.load(f)
            return {
                name: ActionProperties(
                    description=action['description'],
                    weight=action['weight'],
                    function=action['function']
                )
                for name, action in actions_data['actions'].items()
            }
        except Exception as e:
            self.logger.error(f"Error loading actions file: {e}")
            return {}

    def get_action_names(self) -> List[str]:
        return list(self.actions.keys())

    def get_action_weights(self) -> Dict[str, float]:
        return {name: action.weight for name, action in self.actions.items()}

    def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if action not in self.actions:
            self.logger.error(f"Attempted to execute invalid action: {action}")
            return {"error": f"Action '{action}' is not available."}

        action_properties = self.actions[action]
        function_name = action_properties.function

        try:
            method = getattr(self, function_name)
            result = method(context)
            self.logger.info(f"Executed action: {action}")
            return result
        except AttributeError:
            self.logger.error(f"Function {function_name} not found for action {action}")
            return {"error": f"Function {function_name} not found for action {action}"}
        except Exception as e:
            self.logger.error(f"Error executing action {action}: {e}")
            return {"error": f"Error executing action {action}: {str(e)}"}

    def update_action_weight(self, action: str, delta: float):
        if action in self.actions:
            current_weight = self.actions[action].weight
            new_weight = max(0, min(1, current_weight + delta))
            self.actions[action].weight = new_weight
            self.logger.info(f"Updated weight for action {action}: {current_weight} -> {new_weight}")
            self._save_actions_to_file()
        else:
            self.logger.warning(f"Attempted to update weight for invalid action: {action}")

    def _save_actions_to_file(self):
        try:
            actions_data = {
                "actions": {
                    name: {
                        "description": props.description,
                        "weight": props.weight,
                        "function": props.function
                    }
                    for name, props in self.actions.items()
                }
            }
            with open(self.actions_file, 'w') as f:
                json.dump(actions_data, f, indent=2)
            self.logger.info("Saved updated actions to file")
        except Exception as e:
            self.logger.error(f"Error saving actions to file: {e}")

    # Placeholder methods for specific action executions
    def execute_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Executing response action with context: {context}")
        return {"action": "respond", "response": "This is a dummy response."}

    def execute_query(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Executing query action with context: {context}")
        return {"action": "query", "query": "This is a dummy query?"}

    def execute_context_shift(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Executing context shift action with context: {context}")
        return {"action": "shift_context", "new_context": "This is a new dummy context."}

    def execute_conversation_management(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Executing conversation management action with context: {context}")
        return {"action": "manage_conversation", "management": "This is a dummy conversation management action."}

    def execute_internal_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print(f"Executing internal processing action with context: {context}")
        return {"action": "process_internally", "processing": "This is a dummy internal processing action."}


# Testing BicaActions
if __name__ == "__main__":
    import os
    from openai import OpenAI
    import bica_utilities

    # Initialize OpenAI client
    client = OpenAI(api_key=bica_utilities.get_environment_variable("OPENAI_API_KEY"))

    # Initialize BicaActions
    bica_actions = BicaActions(api_key=bica_utilities.get_environment_variable("OPENAI_API_KEY"))

    print("Starting BicaActions test suite...")

    # Test 1: Get action names and weights
    print("\nTest 1: Get action names and weights")
    action_names = bica_actions.get_action_names()
    action_weights = bica_actions.get_action_weights()
    print(f"Available actions: {action_names}")
    print(f"Action weights: {action_weights}")

    # Test 2: Execute each action
    print("\nTest 2: Execute each action")
    for action in action_names:
        context = {"user_input": "Hello, AI!", "emotion": "neutral"}
        result = bica_actions.execute_action(action, context)
        print(f"Result of {action}: {result}")

    # Test 3: Update action weight
    print("\nTest 3: Update action weight")
    action_to_update = "respond"
    original_weight = bica_actions.actions[action_to_update].weight
    bica_actions.update_action_weight(action_to_update, 0.1)
    new_weight = bica_actions.actions[action_to_update].weight
    print(f"Updated '{action_to_update}' weight: {original_weight} -> {new_weight}")

    # Test 4: Simulate a conversation using GPT for decision-making
    print("\nTest 4: Simulate a conversation using GPT for decision-making")
    conversation_history = []

    for i in range(10):  # Simulate 5 turns of conversation
        user_input = input("User: ")
        conversation_history.append(f"User: {user_input}")

        # Use GPT to decide on the next action
        gpt_prompt = f"""
        Given the following conversation history and available actions, decide on the next action to take:

        Conversation history:
        {' '.join(conversation_history)}

        Available actions: {', '.join(action_names)}

        Decide on the most appropriate action and respond with just the action name.
        """

        gpt_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are an AI assistant helping to decide on actions in a conversation."},
                      {"role": "user", "content": gpt_prompt}]
        )

        chosen_action = gpt_response.choices[0].message.content.strip().lower()

        # Execute the chosen action
        context = {"user_input": user_input, "conversation_history": conversation_history}
        result = bica_actions.execute_action(chosen_action, context)

        # Update conversation history
        conversation_history.append(f"AI ({chosen_action}): {result}")

        print(f"Turn {i + 1}:")
        print(f"User: {user_input}")
        print(f"Chosen action: {chosen_action}")
        print(f"Result: {result}")
        print()

    print("BicaActions test suite completed.")
