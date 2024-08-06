import json
from pathlib import Path
from typing import Dict, List, Any, Callable
import random
from gpt_handler import GPTHandler
from bica_logging import BicaLogging


class BicaActions:
    def __init__(self, gpt_handler: GPTHandler):
        self.actions_file = Path(__file__).resolve().parents[2] / "source" / "data" / "definitions" / "possible_actions.json"
        self.actions: Dict[str, Dict[str, Any]] = self._load_actions_from_file()
        self.gpt_handler = gpt_handler
        self.logger = BicaLogging("BicaActions")
        self.action_handlers: Dict[str, Callable] = {}

    def _load_actions_from_file(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.actions_file, 'r') as f:
                actions_data = json.load(f)
            return {action['name']: action for action in actions_data['actions']}
        except Exception as e:
            self.logger.error(f"Error loading actions file: {e}")
            return {}

    def get_action_names(self) -> List[str]:
        return list(self.actions.keys())

    def get_action_details(self, action: str) -> Dict[str, Any]:
        return self.actions.get(action, {})

    def get_action_weights(self) -> Dict[str, float]:
        return {name: action.get('weight', 0) for name, action in self.actions.items()}

    def is_valid_action(self, action: str) -> bool:
        return action in self.actions

    def register_action_handler(self, action: str, handler: Callable):
        if self.is_valid_action(action):
            self.action_handlers[action] = handler
            self.logger.info(f"Registered handler for action: {action}")
        else:
            self.logger.warning(f"Attempted to register handler for invalid action: {action}")

    def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_valid_action(action):
            self.logger.error(f"Attempted to execute invalid action: {action}")
            return {"error": f"Action '{action}' is not available."}

        if action in self.action_handlers:
            try:
                result = self.action_handlers[action](context)
                self.logger.info(f"Executed action: {action}")
                return result
            except Exception as e:
                self.logger.error(f"Error executing action {action}: {e}")
                return {"error": f"Error executing action {action}: {str(e)}"}
        else:
            # Default behavior if no handler is registered
            return self._default_action_execution(action, context)

    def _default_action_execution(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Execute the following action: {action}\nContext: {json.dumps(context)}\nProvide a detailed response as if you were carrying out this action."
        response = next(self.gpt_handler.generate_response(prompt))
        return {
            "action": action,
            "description": self.actions[action].get('description', ''),
            "result": response,
            "context": context
        }

    def suggest_next_action(self, current_context: Dict[str, Any]) -> str:
        action_weights = self.get_action_weights()
        context_str = json.dumps(current_context)

        prompt = f"Given the current context: {context_str}\nAnd the following possible actions with their weights: {json.dumps(action_weights)}\nSuggest the most appropriate next action to take."

        suggested_action = next(self.gpt_handler.generate_response(prompt))

        if self.is_valid_action(suggested_action):
            return suggested_action
        else:
            self.logger.warning(f"GPT suggested invalid action: {suggested_action}. Falling back to random selection.")
            return random.choices(list(action_weights.keys()), weights=list(action_weights.values()))[0]

    def update_action_weight(self, action: str, delta: float):
        if self.is_valid_action(action):
            current_weight = self.actions[action]['weight']
            new_weight = max(0, min(1, current_weight + delta))  # Ensure weight stays between 0 and 1
            self.actions[action]['weight'] = new_weight
            self.logger.info(f"Updated weight for action {action}: {current_weight} -> {new_weight}")
            self._save_actions_to_file()
        else:
            self.logger.warning(f"Attempted to update weight for invalid action: {action}")

    def _save_actions_to_file(self):
        try:
            with open(self.actions_file, 'w') as f:
                json.dump({"actions": list(self.actions.values())}, f, indent=2)
            self.logger.info("Saved updated actions to file")
        except Exception as e:
            self.logger.error(f"Error saving actions to file: {e}")


# Example usage
if __name__ == "__main__":
    from gpt_handler import GPTHandler

    gpt_handler = GPTHandler()
    bica_actions = BicaActions(gpt_handler)

    print("Available actions:")
    for action, details in bica_actions.actions.items():
        print(f"{action}: {details['description']} (weight: {details['weight']})")


    # Register a custom handler for the 'respond' action
    def custom_respond_handler(context):
        return {"result": f"Custom response: {context.get('message', 'Hello!')}"}


    bica_actions.register_action_handler('respond', custom_respond_handler)

    print("\nExecuting 'respond' action with custom handler:")
    result = bica_actions.execute_action('respond', {"message": "How are you?"})
    print(json.dumps(result, indent=2))

    print("\nExecuting 'query' action (using default execution):")
    result = bica_actions.execute_action('query', {"topic": "user's mood"})
    print(json.dumps(result, indent=2))

    print("\nSuggesting next action:")
    current_context = {"user_input": "Tell me a joke", "mood": "happy"}
    suggested_action = bica_actions.suggest_next_action(current_context)
    print(f"Suggested action: {suggested_action}")

    print("\nUpdating action weight:")
    bica_actions.update_action_weight('respond', 0.1)
    print(f"New weight for 'respond': {bica_actions.get_action_weights()['respond']}")