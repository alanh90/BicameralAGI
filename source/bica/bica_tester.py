import sys

sys.path.append('..')  # Add parent directory to Python path

from gpt_handler import GPTHandler
from bica_context import Context
from bica_thoughts import Thoughts


def simple_conversation_test():
    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
    print("Starting simple conversation test with GPT-4o-mini...")

    user_input = input("User: ")
    while user_input.lower() != "exit":
        prompt = f"Respond as a friendly AI assistant to: {user_input}"
        ai_response = next(gpt_handler.generate_response(prompt))
        print(f"AI: {ai_response}")
        user_input = input("User: ")

    print("Test completed.")


def thought_enhanced_conversation_test():
    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
    conversation = []
    thoughts_manager = Thoughts()
    print("Starting thought-enhanced conversation test with GPT-4o-mini...")

    user_input = input("User: ")
    while user_input.lower() != "exit":
        conversation.append(f"User: {user_input}")
        conversation_history = "\n".join(conversation[-5:])  # Last 5 interactions

        thought_prompt = conversation_history
        print("AI is thinking...")
        thought = thoughts_manager.generate_thoughts(thought_prompt)
        print(f"AI thought: {thought}")

        response_prompt = f"Given the conversation history and your thoughts:\n{conversation_history}\n\nYour thoughts: {thought}\n\nRespond as a friendly AI assistant."
        ai_response = next(gpt_handler.generate_response(response_prompt))
        conversation.append(f"AI: {ai_response}")
        print(f"AI: {ai_response}")

        user_input = input("User: ")

    print("Test completed.")


def context_aware_conversation_test():
    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
    context_manager = Context()
    print("Starting context-aware conversation test with GPT-4o-mini...")

    user_input = input("User: ")
    while user_input.lower() != "exit":
        if user_input.lower() == "wipe context":
            context_manager.wipe_context()
            user_input = input("User: ")
            continue

        context_manager.update_context(f"User: {user_input}")
        current_context = context_manager.get_context()

        response_prompt = f"Given the following context:\n{current_context}\n\nRespond as a friendly AI assistant with awareness of the conversation history."
        ai_response = next(gpt_handler.generate_response(response_prompt))
        context_manager.update_context(f"AI: {ai_response}")
        print(f"AI: {ai_response}")

        user_input = input("User: ")

    print("Test completed.")


def main():
    print("Select a test to run:")
    print("1. Simple Conversation Test")
    print("2. Thought-Enhanced Conversation Test")
    print("3. Context-Aware Conversation Test")

    choice = input("Enter the number of the test to run: ")

    if choice == '1':
        simple_conversation_test()
    elif choice == '2':
        thought_enhanced_conversation_test()
    elif choice == '3':
        context_aware_conversation_test()
    else:
        print("Invalid choice. Please select 1, 2, or 3.")


if __name__ == '__main__':
    main()
