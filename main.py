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

    $ python main.py

Author:
-------
Alan Hourmand
Date: 9/23/2024
"""

from bica.core.character import BicaCharacter as Character
from bica.utils.logging import BicaLogging
from colorama import Fore, Style, init
import os

# Initialize logger
logger = BicaLogging("MainScript")

DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'


def initialize_system():
    """Initializes the AI system and prints a welcome message."""
    init(autoreset=True)  # Automatically reset color after each print
    welcome_message = "Welcome to BicameralAGI - A Human-like AI Experiment"
    logger.info(f"Initializing System: {welcome_message}")

    print(f"""
    {Fore.CYAN}{welcome_message}{Style.RESET_ALL}

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
            logger.warning(f"Invalid character input: {character_input}")
            print("Invalid input. Please enter a valid character or description.")
        else:
            logger.info(f"Character description received: {character_input}")
            return character_input


def conversation_loop(orchestrator):
    """Handles the main conversation loop between the AI and the user."""
    logger.info("Starting conversation loop")
    while True:
        user_input = input(f"{Fore.GREEN}{Style.BRIGHT}You: {Style.RESET_ALL}")
        logger.info(f"User input: {user_input}")

        if user_input.lower() == 'exit':
            logger.info("User requested to exit the conversation")
            print(f"{Fore.YELLOW}AI: Goodbye! It was nice talking to you.{Style.RESET_ALL}")
            break

        try:
            ai_response = orchestrator.process_input(user_input)
            logger.info(f"AI response: {ai_response}")
            print(f"{Fore.BLUE}{Style.BRIGHT}BicaAI: {Style.NORMAL}{ai_response}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}AI: I encountered an error: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()


def main():
    """Main function to start the AI system and manage the interaction flow."""
    logger.info("Starting BicameralAGI main script")
    initialize_system()
    print(f"{Fore.RED}Internal Visible: {DEBUG_MODE}")

    # Prompt for character description
    character_description = get_character()
    logger.info(f"Character description: {character_description}")
    print(f"{Fore.MAGENTA}You have chosen: {character_description}. Let's begin the conversation!{Style.RESET_ALL}")

    # Start conversation loop
    character = Character(character_description, debug_mode=DEBUG_MODE)
    logger.info(f"Character initialized: {character.get_character_definition()}")
    print(f"{Fore.CYAN}Character Summary: {character.get_character_definition()}{Style.RESET_ALL}")
    conversation_loop(character)

    logger.info("BicameralAGI session ended")
    print(f"{Fore.YELLOW}Thank you for interacting with BicameralAGI. Goodbye!{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
