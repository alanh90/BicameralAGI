import sys

sys.path.append('..')  # Add parent directory to Python path

from gpt_handler import GPTHandler


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


if __name__ == "__main__":
    simple_conversation_gpt_test()