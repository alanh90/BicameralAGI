from bica_memory import BicaMemory
from bica_context import BicaContext
from bica_cognition import BicaCognition
from bica_affect import BicaAffect
from bica_action import BicaActions
from bica_safety import BicaSafety
from bica_destiny import BicaDestiny
from gpt_handler import GPTHandler
from bica_logging import BicaLogging
from bica_utilities import BicaUtilities
import threading
import time


class BicaOrchestrator:
    def __init__(self):
        self.memory = BicaMemory(embedding_dim=384)
        self.context = BicaContext()
        self.cognition = BicaCognition(self.memory, self.context)
        self.affect = BicaAffect()
        self.actions = BicaActions(BicaUtilities.get_environment_variable("OPENAI_API_KEY"))
        self.safety = BicaSafety()
        self.destiny = BicaDestiny("BicaAI")
        self.gpt_handler = GPTHandler()
        self.logger = BicaLogging("BicaOrchestrator")
        self.recent_conversation = []

        # Start background processes
        threading.Thread(target=self.run_subconscious, daemon=True).start()
        threading.Thread(target=self.run_emotional_processes, daemon=True).start()
        threading.Thread(target=self.run_memory_consolidation, daemon=True).start()

    def run_subconscious(self):
        while True:
            self.cognition.process_subconscious()
            time.sleep(1)  # Adjust as needed

    def run_emotional_processes(self):
        while True:
            self.affect.process_background_emotions()
            time.sleep(1)  # Adjust as needed

    def run_memory_consolidation(self):
        while True:
            if not self.memory.is_dreaming():
                self.memory.consolidate_memories()
            time.sleep(60)  # Check every minute

    def process_input(self, user_input: str) -> str:
        self.logger.info(f"Processing user input: {user_input}")

        # 1. Update context and recent conversation
        self.recent_conversation.append(f"User: {user_input}")
        context = self.context.update_context(user_input)

        # 2. Update short-term memory
        self.memory.add_memory(user_input, {"relevance": 0.8})

        # 3. Retrieve relevant memories
        relevant_memories = self.memory.recall_memory(user_input)

        # 4. Generate conscious thoughts
        conscious_thoughts = self.cognition.generate_thoughts(context)

        # 5. Get current emotional state
        current_emotion = self.affect.get_emotional_state()

        # 6. Retrieve current destinies
        current_destinies = self.destiny.get_destinies()

        # 7. Apply safety filter
        safe_thoughts = self.safety.safety_filter(str(conscious_thoughts), "thought")
        safe_emotion = self.safety.safety_filter(str(current_emotion), "emotion")

        # 8. Perform cognitive analysis
        cognitive_analysis = self.cognition.analyze(user_input, safe_thoughts, relevant_memories)

        # 9. Determine action
        action_context = {
            "user_input": user_input,
            "context": context,
            "thoughts": safe_thoughts,
            "emotion": safe_emotion,
            "memories": relevant_memories,
            "destinies": current_destinies,
            "cognitive_analysis": cognitive_analysis,
            "recent_conversation": self.get_recent_conversation()
        }
        action = self.actions.execute_action("respond", action_context)

        # 10. Generate and safety-check the response
        ai_response = action.get("response", "I'm not sure how to respond.")
        safe_response = self.safety.safety_filter(ai_response, "content")

        # 11. Update long-term memory
        self.memory.add_memory(f"User: {user_input}\nAI: {safe_response}", {"importance": 0.7})

        # 12. Update emotional state
        self.affect.trigger_emotion(safe_response, 0.5)  # Intensity can be adjusted

        # 13. Update personality if necessary
        self.affect.update_personality_based_on_experience(safe_response)

        # 14. Potentially alter destiny
        if self.is_significant_interaction(cognitive_analysis):
            self.destiny.alter_destiny(0, new_development=safe_response)

        # 15. Log the response and update recent conversation
        self.recent_conversation.append(f"AI: {safe_response}")
        self.logger.info(f"Generated AI response: {safe_response}")

        return safe_response

    def get_recent_conversation(self, num_turns: int = 5) -> list:
        return self.recent_conversation[-num_turns:]

    def is_significant_interaction(self, cognitive_analysis: dict) -> bool:
        return cognitive_analysis.get('significance', 0) > 0.7

    def initiate_dreaming(self):
        if not self.memory.is_dreaming():
            self.memory.initiate_dreaming()


# Example usage
if __name__ == "__main__":
    orchestrator = BicaOrchestrator()
    user_input = "Hello, can you tell me about your goals and how you perceive the world?"
    ai_response = orchestrator.process_input(user_input)
    print(f"User: {user_input}")
    print(f"AI: {ai_response}")