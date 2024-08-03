from bica_memory import BicaMemory
from bica_personality import BicaPersonality
from bica_thoughts import BicaThoughts
from bica_emotions import BicaEmotions
from gpt_handler import GPTHandler
from bica_logging import BicaLogging


class BicaOrchestrator:
    def __init__(self):
        self.memory = BicaMemory(embedding_dim=384)  # Assuming 384 is the embedding dimension
        self.personality = BicaPersonality()
        self.thoughts = BicaThoughts()
        self.emotions = BicaEmotions()
        self.gpt_handler = GPTHandler()
        self.logger = BicaLogging("BicaOrchestrator")
        self.recent_conversation = []

    def process_input(self, user_input: str) -> str:
        self.logger.info(f"Processing user input: {user_input}")

        # Step 1: Take in user input
        self.recent_conversation.append(f"User: {user_input}")

        # Step 2: Update and gather short-term memory
        self.memory.update_short_term_memory(user_input)  # This method needs to be added to BicaMemory
        short_term_memory = self.memory.get_short_term_memory()

        # Step 3: Gather personality reference
        personality_profile = self.personality.get_personality_profile()  # This method needs to be added to BicaPersonality

        # Step 4: Gather important long-term memories
        long_term_memories = self.memory.get_important_long_term_memories()  # This method needs to be added to BicaMemory

        # Step 5: Get recent conversation
        recent_conversation = self.get_recent_conversation()

        # Step 6: Trigger thought process
        thoughts = self.thoughts.generate_thoughts(user_input, short_term_memory, long_term_memories, recent_conversation)

        # Step 7: Trigger emotions
        emotions = self.emotions.process_emotions(user_input, short_term_memory, long_term_memories, thoughts)

        # Step 8: Generate output
        ai_response = self.generate_output(user_input, short_term_memory, personality_profile, long_term_memories, recent_conversation, thoughts, emotions)

        self.recent_conversation.append(f"AI: {ai_response}")
        self.logger.info(f"Generated AI response: {ai_response}")

        return ai_response

    def get_recent_conversation(self, num_turns: int = 5) -> list:
        return self.recent_conversation[-num_turns:]

    def generate_output(self, user_input, short_term_memory, personality_profile, long_term_memories, recent_conversation, thoughts, emotions):
        prompt = f"""
        User Input: {user_input}
        Short-term Memory: {short_term_memory}
        Personality Profile: {personality_profile}
        Long-term Memories: {long_term_memories}
        Recent Conversation: {recent_conversation}
        Current Thoughts: {thoughts}
        Current Emotions: {emotions}

        Based on the above information, generate an appropriate response for the AI. The response should be coherent, contextually relevant, and reflect the AI's personality, thoughts, and emotions.

        AI Response:
        """

        response = next(self.gpt_handler.generate_response(prompt))
        return response.strip()


# Example usage
if __name__ == "__main__":
    orchestrator = BicaOrchestrator()
    user_input = "Hello, how are you today?"
    ai_response = orchestrator.process_input(user_input)
    print(f"User: {user_input}")
    print(f"AI: {ai_response}")