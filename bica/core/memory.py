from typing import List, Dict
import time
import random
from collections import Counter
from bica.external.gpt_handler import GPTHandler
from bica.core.profile import BicaProfile
from bica.utils.utilities import normalize_text


class Memory:
    def __init__(self, content: str, emotions: Dict[str, float], importance: float):
        self.content = content
        self.emotions = emotions
        self.importance = importance
        self.timestamp = time.time()
        self.active = False


class BicaMemory:
    def __init__(self, character_profile: BicaProfile):
        self.gpt_handler = GPTHandler()
        self.profile = character_profile
        self.base_emotions = self.profile.character_profile['cognitiveModel']['emotions']

        self.working_memory = []
        self.short_term_memory = []
        self.long_term_memory = []

    def update_memories(self, context_data):
        pass

    def get_memories(self):
        # Combine and return relevant memories
        return {
            "working_memory": self.working_memory,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self._get_relevant_long_term_memories()
        }

    def _get_relevant_long_term_memories(self):
        pass

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


def main():
    print("\n===== Initializing BicaMemory System =====")
    character_profile = BicaProfile("Bica AI", "A highly intelligent, human-like AI assistant.", GPTHandler())
    memory_system = BicaMemory(character_profile)

    print("\n===== Testing Memory Creation and Short-Term Memory Management =====")
    for i in range(20):
        content = f"Memory {i}: This is a test memory with content related to number {i}"
        emotions = {"joy": random.random(), "sadness": random.random(), "surprise": random.random()}
        importance = random.random()
        memory_system.save_memory(content, emotions, importance)
        print(f"Created memory: '{content[:30]}...' with emotions {emotions} and importance {importance:.2f}")
        time.sleep(0.1)  # Simulate time passing

    print(f"\nCurrent memory state:")
    print(f"Short-term layer 1: {len(memory_system.short_term_layer1)} memories")
    print(f"Short-term layer 2: {len(memory_system.short_term_layer2)} memories")
    print(f"Short-term layer 3: {len(memory_system.short_term_layer3)} memories")
    print(f"Long-term memories: {len(memory_system.long_term_memories)} memories")

    print("\n===== Testing Memory Recall =====")
    query = "Memory 5"
    print(f"Attempting to recall memories related to: '{query}'")
    recalled_memories = memory_system.recall_memory(query)
    for memory in recalled_memories:
        print(f"Recalled: '{memory.content[:50]}...' (Importance: {memory.importance:.2f}, Emotions: {memory.emotions})")

    print("\n===== Final Memory System State =====")
    print(f"Short-term layer 1: {len(memory_system.short_term_layer1)} memories")
    print(f"Short-term layer 2: {len(memory_system.short_term_layer2)} memories")
    print(f"Short-term layer 3: {len(memory_system.short_term_layer3)} memories")
    print(f"Long-term memories: {len(memory_system.long_term_memories)} memories")


if __name__ == "__main__":
    main()
    """



