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
from colorama import Fore, Style, init


def main():
    orchestrator = BicaOrchestrator()

    init(autoreset=True)  # Automatically reset color after each print

    print(f"""
    {Fore.CYAN}Welcome to BicameralAGI - A Human-like AI Experiment{Style.RESET_ALL}

    {Fore.GREEN}This AI attempts to understand and respond from multiple perspectives,
    simulating more nuanced, human-like conversations.{Style.RESET_ALL}

    {Fore.YELLOW}Key features:{Style.RESET_ALL}
    - Multi-perspective reasoning (positive, neutral, negative viewpoints)
    - Short and long-term memory simulation
    - Emotional intelligence and personality traits (in development)
    - Internal monologue and decision-making processes
    - Contextual understanding and adaptive responses
    - Ethical considerations and safety measures
    - Simulated dreaming for memory consolidation (planned)

    {Fore.MAGENTA}Type your messages to chat with the AI. Type 'exit' to end.{Style.RESET_ALL}

    {Fore.RED}Note: This project is still under development and I work alone with a tight schedule. Sometimes it will be broken. Your feedback is appreciated!{Style.RESET_ALL}

    Let's begin!
    """)

    while True:
        user_input = input(f"{Fore.GREEN}{Style.BRIGHT}You: {Style.RESET_ALL}")

        if user_input.lower() == 'exit':
            print("AI: Goodbye! It was nice talking to you.")
            break

        try:
            ai_response = orchestrator.process_input(user_input)
            print(f"{Fore.BLUE}{Style.BRIGHT}BicaAI: {Style.NORMAL}{ai_response}")
        except Exception as e:
            print(f"AI: I apologize, but I encountered an error: {str(e)}")
            # Print the full traceback for debugging
            import traceback
            traceback.print_exc()

    print(f"{Fore.YELLOW}Thank you for interacting with BicameralAGI. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()