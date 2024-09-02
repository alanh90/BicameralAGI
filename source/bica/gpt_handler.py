"""
This module provides an interface for interacting with GPT models. It handles API calls, response generation, and manages various parameters for GPT interactions used throughout the BicameralAGI system.
"""

from openai import OpenAI
from typing import List, Dict, Any, Optional, Union
from bica_utilities import BicaUtilities
from bica_logging import BicaLogging


class GPTHandler:
    def __init__(self, api_provider: str = "openai", model: str = "gpt-4o-mini"):
        self.api_provider = api_provider
        self.model = model
        self.logger = BicaLogging("GPTHandler")

        if self.api_provider == "openai":
            api_key = BicaUtilities.get_environment_variable("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError("Invalid API provider. Choose 'openai'.")

    def generate_response(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            top_p: float = 1,
            frequency_penalty: float = 0,
            presence_penalty: float = 0,
            stop: Optional[Union[str, List[str]]] = None,
            stream: bool = False,
            functions: Optional[List[Dict[str, Any]]] = None,
            function_call: Optional[Union[str, Dict[str, str]]] = None
    ) -> Union[str, Dict[str, Any]]:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream
            }

            if functions:
                params["functions"] = functions
                if function_call:
                    params["function_call"] = function_call

            response = self.client.chat.completions.create(**params)

            if stream:
                return self._handle_stream_response(response)
            else:
                return self._handle_normal_response(response)
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise

    def _handle_stream_response(self, response):
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        return full_response

    def _handle_normal_response(self, response):
        message = response.choices[0].message
        if hasattr(message, 'function_call') and message.function_call:
            return {
                "function_call": message.function_call.model_dump(),
                "content": message.content
            }
        return message.content.strip()


# Example usage
if __name__ == "__main__":
    handler = GPTHandler()

    # Simple response
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    response = handler.generate_response(messages)
    print("Simple response:", response)

    # Function calling example
    functions = [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like in London?"}
    ]
    function_response = handler.generate_response(
        messages,
        functions=functions,
        function_call="auto"
    )
    print("Function call response:", function_response)