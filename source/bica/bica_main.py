"""
BicameralAGI Main Entry Point
=============================

Overview:
---------
This module serves as the primary entry point for the BicameralAGI system, responsible for initializing
and managing the core conversation loop that enables interaction between the user and the AI. It coordinates
high-level operations by interfacing with the `BicaOrchestrator`, which handles input processing and decision-making.

Usage:
------
Run the script to initialize the BicameralAGI system and begin interacting with the AI:

    $ python bica_main.py

Author:
-------
Alan Hourmand
Date: 9/23/2024
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
        except Exception as e:
            print(f"AI: I apologize, but I encountered an error: {str(e)}")
            # Print the full traceback for debugging
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()