from bica_orchestrator import BicaOrchestrator
from bica_logging import BicaLogging


def main():
    logger = BicaLogging("BicaMain")
    orchestrator = BicaOrchestrator()
    logger.info("BicameralAGI system initialized")

    print("Welcome to the BicameralAGI system!")
    print("You can start chatting with the AI. Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'exit':
            print("AI: Goodbye! It was nice talking to you.")
            logger.info("User ended the conversation")
            break

        try:
            ai_response = orchestrator.process_input(user_input)
            print(f"AI: {ai_response}")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}", exc_info=True)
            print("AI: I apologize, but I encountered an error. Could you please try again?")


if __name__ == "__main__":
    main()