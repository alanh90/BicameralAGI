import sys
import unittest
from unittest.mock import patch, AsyncMock
import asyncio

sys.path.append('..')  # Add parent directory to Python path

from gpt_handler import GPTHandler
import bica_thoughts


def simple_conversation_gpt_test():
    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
    memory = []

    print("Starting simple conversation test with memory...")

    user_input = input("User: ")

    while user_input.lower() != "exit":
        # Add user input to memory
        memory.append(f"User: {user_input}")

        # Construct context from memory
        context = "\n".join(memory[-5:])  # Use last 5 interactions for context

        # Generate AI response
        prompt = f"Given the following conversation:\n{context}\n\nRespond as a friendly AI assistant with knowledge of the conversation history."
        ai_response = gpt_handler.generate_response(prompt)

        # Add AI response to memory
        memory.append(f"AI: {ai_response}")

        print(f"AI: {ai_response}")

        # Get next user input
        user_input = input("User: ")

    print("Test completed.")


class TestBicaThoughts(unittest.IsolatedAsyncioTestCase):

    @patch('bica_thoughts.gpt_handler.generate_response')
    async def test_generate_thoughts(self, mock_generate_response):
        mock_generate_response.return_value = "This is a generated thought.\nAnother thought."
        bica_thoughts.thought_queue = asyncio.Queue()  # Reset queue for testing

        await bica_thoughts.generate_thoughts("Test prompt")
        self.assertEqual(await bica_thoughts.thought_queue.get(), "This is a generated thought.")
        self.assertEqual(await bica_thoughts.thought_queue.get(), "Another thought.")
        self.assertEqual(await bica_thoughts.thought_queue.get(), None)

    @patch('bica_thoughts.gpt_handler.generate_response')
    async def test_analyze_thought(self, mock_generate_response):
        mock_generate_response.return_value = "This is an analysis of the thought."
        bica_thoughts.thought_queue = asyncio.Queue()
        await bica_thoughts.thought_queue.put("This is a generated thought.")
        await bica_thoughts.thought_queue.put(None)  # Signal end of thoughts

        await bica_thoughts.analyze_thought()
        self.assertTrue(mock_generate_response.called)


def run_tests():
    unittest.main()


def main():
    print("Select a test to run:")
    print("1. Simple Conversation GPT Test")
    print("2. Run Unit Tests")

    choice = input("Enter the number of the test to run: ")

    if choice == '1':
        simple_conversation_gpt_test()
    elif choice == '2':
        run_tests()
    else:
        print("Invalid choice. Please select either 1 or 2.")


if __name__ == '__main__':
    main()
