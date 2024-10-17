from typing import List, Dict
import time
import random
from collections import Counter
from bica.external.gpt_handler import GPTHandler
from bica.core.profile import BicaProfile
from bica.utils.utilities import normalize_text


class Memory:
    def __init__(self, content: str, importance: float):
        self.content = content
        self.importance = importance
        self.timestamp = time.time()

    def __str__(self):
        return f"Memory(content='{self.content[:50]}...', importance={self.importance:.2f})"


class BicaMemory:
    def __init__(self, character_profile: BicaProfile, debug_mode: bool):
        self.debug_mode = debug_mode
        self.gpt_handler = GPTHandler()
        self.profile = character_profile
        # self.base_emotions = self.profile.character_profile['cognitiveModel']['emotions']

        self.working_memory = []
        self.short_term_memory = []
        self.long_term_memory = []
        self.self_memory = self.initialize_self_memory()

    def initialize_self_memory(self):
        return f"I am {self.profile.character_name}. My Description: {self.profile.character_summary}"

    def _extract_float(self, response):
        """Extract a float value from the GPT response."""
        try:
            # First, try to directly convert the response to a float
            return float(response)
        except ValueError:
            # If that fails, try to find a float in the response
            import re
            match = re.search(r'\d+(\.\d+)?', response)
            if match:
                return float(match.group())
            else:
                # If no float is found, return a default value
                print(f"Warning: Could not extract float from GPT response: {response}")
                return 0.5  # Default importance

    def update_memories(self, context_data):
        importance_prompt = f"""
        Given the following context, rate its importance in terms of memorizing it on a scale of 0 to 1:
        User Input: {context_data['user_input']}
        Context: {context_data['updated_context']}
        Recent Conversation: {context_data['recent_conversation']}

        Respond with a single float value between 0 and 1.
        """
        importance_response = self.gpt_handler.generate_response(importance_prompt)
        importance = self._extract_float(importance_response)
        importance = max(0, min(importance, 1))

        new_memory_content = f"User: {context_data['user_input']}\nContext: {context_data['updated_context']}"
        new_memory = Memory(content=new_memory_content, importance=importance)

        if importance > 0.7:
            self.working_memory.append(new_memory)
            self.working_memory = sorted(self.working_memory, key=lambda x: x.importance, reverse=True)[:5]

        self.short_term_memory.append(new_memory)
        self.short_term_memory = sorted(self.short_term_memory, key=lambda x: x.importance, reverse=True)[:20]

        if len(self.short_term_memory) > 20:
            oldest_memory = min(self.short_term_memory, key=lambda x: x.timestamp)
            if oldest_memory.importance > 0.5:
                self.long_term_memory.append(oldest_memory)
            self.short_term_memory.remove(oldest_memory)

    def get_memories(self):
        relevant_long_term_memories = self.get_relevant_long_term_memories()
        if self.debug_mode:
            print(f"Working Memory: {self.working_memory}")
            print(f"Short Term Memory: {self.short_term_memory}")
            print(f"Long Term Memory: {relevant_long_term_memories}")
            print(f"Self Memory: {self.self_memory}")

        return {
            "working_memory": self.working_memory,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": relevant_long_term_memories,
            "self_memory": self.self_memory
        }

    def get_relevant_long_term_memories(self):
        if not self.long_term_memory:
            return []

        context = f"Working Memory: {[m.content for m in self.working_memory]}\n" \
                  f"Short-term Memory: {[m.content for m in self.short_term_memory]}"

        relevance_prompt = f"""
        Given the current context:
        {context}

        Analyze the following long-term memories and select the 5 most relevant ones:
        {[m.content for m in self.long_term_memory]}

        Provide the indices of the 5 most relevant memories, separated by commas.
        """

        relevant_indices_response = self.gpt_handler.generate_response(relevance_prompt)
        relevant_indices = [int(idx.strip()) for idx in relevant_indices_response.split(',') if idx.strip().isdigit()]

        if not relevant_indices:
            print("Warning: No relevant long-term memories found.")
            return []

        return [self.long_term_memory[idx] for idx in relevant_indices if idx < len(self.long_term_memory)]

    def text_similarity(self, text1: str, text2: str) -> float:
        return len(set(normalize_text(text1).split()) & set(normalize_text(text2).split())) / len(set(normalize_text(text1).split() + normalize_text(text2).split()))

    """
    def save_memory(self, content: str, emotions: Dict[str, float], importance: float):
        memory = Memory(content, emotions, importance)
        self.short_term_layer1.append(memory)
        self.manage_short_term_layers()

    def manage_short_term_layers(self):
        current_time = time.time()
        # Move memories between layers based on age and importance
        for memory in self.short_term_layer1[:]:
            age = current_time - memory.timestamp
            if age > 300 or memory.importance > 0.7:
                self.short_term_layer1.remove(memory)
                self.short_term_layer2.append(memory)

        for memory in self.short_term_layer2[:]:
            age = current_time - memory.timestamp
            if age > 3600 or memory.importance > 0.9:
                self.short_term_layer2.remove(memory)
                self.short_term_layer3.append(memory)

        for memory in self.short_term_layer3[:]:
            age = current_time - memory.timestamp
            if age > 86400:
                self.short_term_layer3.remove(memory)
                self.long_term_memories.append(memory)

    def recall_memory(self, query: str, top_k: int = 5) -> List[Memory]:
        all_memories = self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3 + self.long_term_memories
        similarities = [(memory, self.text_similarity(query, memory.content)) for memory in all_memories]
        top_memories = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return [memory for memory, _ in top_memories]

    def text_similarity(self, text1: str, text2: str) -> float:
        return len(set(normalize_text(text1).split()) & set(normalize_text(text2).split())) / len(set(normalize_text(text1).split() + normalize_text(text2).split()))

    def get_recent_memories(self, n: int) -> List[str]:
        all_memories = self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3
        sorted_memories = sorted(all_memories, key=lambda x: x.timestamp, reverse=True)
        return [memory.content for memory in sorted_memories[:n]]

    def decay_active_memories(self, decay_rate: float = 0.1):
        current_time = time.time()
        for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]:
            for memory in layer:
                if memory.active:
                    memory_age = current_time - memory.timestamp
                    if random.random() < decay_rate * memory_age / 3600:  # Probability increases with age
                        memory.active = False
                        print(f"Memory deactivated due to decay: {memory.content[:30]}...")

    def get_emotional_memories(self, emotion: str, threshold: float = 0.5) -> List[Memory]:
        return [m for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]
                for m in layer if m.emotions.get(emotion, 0) >= threshold]

    def get_important_memories(self, threshold: float = 0.7) -> List[Memory]:
        return [m for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]
                for m in layer if m.importance >= threshold]
"""


def main():
    print("\n===== Initializing BicaMemory System =====")
    character_profile = BicaProfile("Test AI", "A test AI for memory system validation.", GPTHandler())
    memory_system = BicaMemory(character_profile, debug_mode=True)

    print("\n===== Testing Memory Update and Retrieval =====")
    test_contexts = [
        {
            "user_input": "Hello, how are you?",
            "updated_context": "The conversation has just started.",
            "recent_conversation": []
        },
        {
            "user_input": "Tell me about your capabilities.",
            "updated_context": "The user is inquiring about the AI's abilities.",
            "recent_conversation": ["Hello, how are you?"]
        },
        {
            "user_input": "What's the weather like today?",
            "updated_context": "The user is asking about the weather.",
            "recent_conversation": ["Hello, how are you?", "Tell me about your capabilities."]
        },
        {
            "user_input": "Remember that my favorite color is blue.",
            "updated_context": "The user is providing personal information.",
            "recent_conversation": ["What's the weather like today?"]
        },
        {
            "user_input": "Can you recall my favorite color?",
            "updated_context": "The user is testing the AI's memory.",
            "recent_conversation": ["Remember that my favorite color is blue."]
        }
    ]

    for i, context in enumerate(test_contexts, 1):
        print(f"\nUpdating memory with context {i}:")
        memory_system.update_memories(context)
        print(f"Memory updated with: {context['user_input']}")

        print("\nRetrieving memories:")
        memories = memory_system.get_memories()
        for memory_type, memory_contents in memories.items():
            print(f"{memory_type.capitalize()}:")
            if isinstance(memory_contents, list):
                for memory in memory_contents:
                    if isinstance(memory, Memory):
                        print(f"  - {memory.content[:100]}... (Importance: {memory.importance:.2f})")
                    else:
                        print(f"  - {memory[:100]}...")
            else:
                print(f"  - {memory_contents[:100]}...")

    print("\n===== Testing Long-Term Memory Retrieval =====")
    # Add some diverse long-term memories
    long_term_memories = [
        "User's favorite food is pizza.",
        "The AI was created on January 1st, 2023.",
        "User mentioned having a dog named Max.",
        "Last conversation was about climate change.",
        "User expressed interest in learning Python programming.",
        "The capital of France is Paris.",
        "User's birthday is on July 15th.",
        "AI recommended a book titled 'The Hitchhiker's Guide to the Galaxy'.",
        "User mentioned they work as a teacher.",
        "Last week, the user talked about their recent trip to Japan."
    ]
    for memory in long_term_memories:
        memory_system.long_term_memory.append(Memory(memory, importance=0.5))

    print("Added diverse long-term memories.")

    print("\nTesting retrieval with specific query:")
    query_context = {
        "user_input": "What do you remember about my interests?",
        "updated_context": "The user is asking about their previously mentioned interests.",
        "recent_conversation": ["Can you recall my favorite color?"]
    }
    memory_system.update_memories(query_context)

    relevant_memories = memory_system.get_relevant_long_term_memories()
    print("Relevant long-term memories:")
    for memory in relevant_memories:
        if isinstance(memory, Memory):
            print(f"  - {memory.content[:100]}... (Importance: {memory.importance:.2f})")
        else:
            print(f"  - {memory[:100]}...")

    print("\n===== Testing Memory Importance =====")
    high_importance_context = {
        "user_input": "Remember this: The secret code is 1234.",
        "updated_context": "The user is providing critical information.",
        "recent_conversation": []
    }
    memory_system.update_memories(high_importance_context)
    print("Added high importance memory.")

    print("\nRetrieving high importance memories:")
    memories = memory_system.get_memories()
    for memory_type, memory_contents in memories.items():
        if memory_type == "working_memory":
            for memory in memory_contents:
                if isinstance(memory, Memory):
                    print(f"  - {memory.content[:100]}... (Importance: {memory.importance:.2f})")
                else:
                    print(f"  - {memory[:100]}...")

    print("\n===== Memory System Test Complete =====")


if __name__ == "__main__":
    main()
