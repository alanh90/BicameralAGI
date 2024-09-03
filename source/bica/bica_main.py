"""
This is the main entry point for the BicameralAGI system. It initializes the system, manages the main conversation loop with users, and coordinates high-level system operations.
"""

from bica_orchestrator import BicaOrchestrator


def main():
    orchestrator = BicaOrchestrator()

    print("Welcome to the BicameralAGI system!")
    print("You can start chatting with the AI. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("AI: Goodbye! It was nice talking to you.")
            break

        try:
            ai_response = orchestrator.process_input(user_input)
            print(f"AI: {ai_response}")
            orchestrator.print_recent_memories()  # Add this line
        except Exception as e:
            print("AI: I apologize, but I encountered an error. Could you please try again?")


if __name__ == "__main__":
    main()
