import json
import os
from pathlib import Path
from typing import Dict, List, Any


class BicaActions:
    def __init__(self):
        # Construct the path to the JSON file
        self.actions_file = Path(__file__).resolve().parents[2] / "source" / "data" / "definitions" / "possible_actions.json"
        self.actions: Dict[str, Dict[str, Any]] = self._load_actions_from_file()
        self.current_action: str = None

    def _load_actions_from_file(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.actions_file, 'r') as f:
                actions_data = json.load(f)
            return {action['name']: action for action in actions_data['actions']}
        except FileNotFoundError:
            print(f"Error: The file {self.actions_file} was not found.")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File path being accessed: {self.actions_file.absolute()}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: The file {self.actions_file} is not a valid JSON file.")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def get_action_names(self) -> List[str]:
        return list(self.actions.keys())

    def get_action_details(self, action: str) -> Dict[str, Any]:
        return self.actions.get(action, {})

    def get_action_weights(self) -> Dict[str, float]:
        return {name: action.get('weight', 0) for name, action in self.actions.items()}

    def get_function_descriptions(self) -> List[Dict[str, Any]]:
        function_descriptions = []
        for name, action in self.actions.items():
            function_descriptions.append({
                "name": name,
                "description": action.get('description', ''),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "The context or specific details for executing the action"
                        }
                    },
                    "required": ["context"]
                }
            })
        return function_descriptions

    def execute_action(self, action: str, **kwargs) -> Dict[str, Any]:
        if action not in self.actions:
            return {"error": f"Action '{action}' not available."}

        self.current_action = action

        # Placeholder implementation
        return {
            "action": action,
            "description": self.actions[action].get('description', ''),
            "result": f"Executed action: {action}",
            "context": kwargs.get('context', '')
        }


# Example usage
def test_bica_actions():
    print("Initializing BicaActions...")
    bica_actions = BicaActions()

    if not bica_actions.actions:
        print("Error: No actions were loaded. Please check the file path and contents.")
        return

    print("\n1. Testing get_action_names():")
    action_names = bica_actions.get_action_names()
    print(f"Available actions: {', '.join(action_names)}")

    print("\n2. Testing get_action_details():")
    for action in action_names:
        details = bica_actions.get_action_details(action)
        print(f"{action}: {json.dumps(details, indent=2)}")

    print("\n3. Testing get_action_weights():")
    weights = bica_actions.get_action_weights()
    for action, weight in weights.items():
        print(f"{action}: {weight}")

    print("\n4. Testing get_function_descriptions():")
    function_descriptions = bica_actions.get_function_descriptions()
    print(json.dumps(function_descriptions, indent=2))

    print("\n5. Testing execute_action():")
    for action in action_names:
        result = bica_actions.execute_action(action, context=f"Test context for {action}")
        print(f"Executing '{action}':")
        print(json.dumps(result, indent=2))

    print("\n6. Testing error handling:")
    print("Trying to execute non-existent action:")
    result = bica_actions.execute_action('non_existent_action', context="This should fail")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_bica_actions()
