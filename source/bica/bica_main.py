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

from bica_character import BicaCharacter as Character
from colorama import Fore, Style, init


def initialize_system():
    """Initializes the AI system and prints a welcome message."""
    init(autoreset=True)  # Automatically reset color after each print
    print(f"""
    {Fore.CYAN}Welcome to BicameralAGI - A Human-like AI Experiment{Style.RESET_ALL}

    {Fore.GREEN}This AI simulates nuanced, human-like conversations with multi-perspective reasoning.{Style.RESET_ALL}

    {Fore.YELLOW}Key features:{Style.RESET_ALL}
    - Multi-perspective reasoning (positive, neutral, negative viewpoints)
    - Short and long-term memory simulation
    - Emotional intelligence and personality traits (in development)
    - Internal monologue and decision-making processes
    - Contextual understanding and adaptive responses
    - Ethical considerations and safety measures
    - Simulated dreaming for memory consolidation (planned)

    {Fore.MAGENTA}Type your messages to chat with the AI. Type 'exit' to end.{Style.RESET_ALL}

    {Fore.RED}Note: This project is under development and it will occasionally be broken. I work alone so I try to fix bugs whenever I get a chance. Your feedback is appreciated!{Style.RESET_ALL}
    """)


def get_character():
    """Prompts the user to input a character or description to interact with."""
    while True:
        character_input = input(f"{Fore.CYAN}{Style.DIM}Enter a character or character description: {Style.RESET_ALL}")
        if character_input.lower() in ['exit', ''] or character_input.isnumeric():
            print("Invalid input. Please enter a valid character or description.")
        else:
            return character_input


def conversation_loop(orchestrator):
    """Handles the main conversation loop between the AI and the user."""
    while True:
        user_input = input(f"{Fore.GREEN}{Style.BRIGHT}You: {Style.RESET_ALL}")

        if user_input.lower() == 'exit':
            print(f"{Fore.YELLOW}AI: Goodbye! It was nice talking to you.{Style.RESET_ALL}")
            break

        try:
            ai_response = orchestrator.process_input(user_input)
            print(f"{Fore.BLUE}{Style.BRIGHT}BicaAI: {Style.NORMAL}{ai_response}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}AI: I encountered an error: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to start the AI system and manage the interaction flow."""
    initialize_system()

    # Prompt for character description
    character_description = get_character()
    print(f"{Fore.MAGENTA}You have chosen: {character_description}. Let's begin the conversation!{Style.RESET_ALL}")

    # Start conversation loop
    character = Character(character_description)
    print(f"{Fore.CYAN}Character Summary: {character.get_character_definition()}{Style.RESET_ALL}")
    conversation_loop(character)

    print(f"{Fore.YELLOW}Thank you for interacting with BicameralAGI. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
