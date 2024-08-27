"""
Memory is completed for now
"""

import time
import random
from typing import List, Dict, Any
from collections import Counter
import numpy as np
from bica.bica_utilities import BicaUtilities
from bica.gpt_handler import GPTHandler


class Memory:
    def __init__(self, content: str, emotions: Dict[str, float], importance: float):
        self.content = content
        self.emotions = emotions
        self.importance = importance
        self.timestamp = time.time()
        self.active = False


class BicaMemory:
    def __init__(self):
        self.utilities = BicaUtilities()
        self.gpt_handler = GPTHandler()

        self.short_term_layer1 = []
        self.short_term_layer2 = []
        self.short_term_layer3 = []
        self.long_term_memories = []
        self.self_memories = []

        self.emotion_connections = {}
        self.importance_connections = {}

        self.BASE_EMOTIONS = {
            "joy": np.random.randn(384),
            "sadness": np.random.randn(384),
            "anger": np.random.randn(384),
            "fear": np.random.randn(384),
            "disgust": np.random.randn(384),
            "surprise": np.random.randn(384),
            "love": np.random.randn(384),
            "trust": np.random.randn(384),
            "anticipation": np.random.randn(384),
            "curiosity": np.random.randn(384),
            "shame": np.random.randn(384),
            "pride": np.random.randn(384),
            "guilt": np.random.randn(384),
            "envy": np.random.randn(384),
            "gratitude": np.random.randn(384),
            "awe": np.random.randn(384),
            "contempt": np.random.randn(384),
            "anxiety": np.random.randn(384),
            "boredom": np.random.randn(384),
            "confusion": np.random.randn(384)
        }
        for emotion in self.BASE_EMOTIONS:
            self.BASE_EMOTIONS[emotion] /= np.linalg.norm(self.BASE_EMOTIONS[emotion])

    def save_memory(self, content: str, emotions: Dict[str, float], importance: float):
        memory = Memory(content, emotions, importance)
        self.short_term_layer1.append(memory)
        self.update_connections(memory)
        self.manage_short_term_layers()
        print(f"Memory saved: {content[:30]}...")

    def update_connections(self, memory: Memory):
        for emotion, intensity in memory.emotions.items():
            if emotion not in self.emotion_connections:
                self.emotion_connections[emotion] = []
            self.emotion_connections[emotion].append(memory)

            # Use BASE_EMOTIONS to influence related emotions
            for related_emotion, related_vector in self.BASE_EMOTIONS.items():
                if related_emotion != emotion:
                    similarity = np.dot(self.BASE_EMOTIONS[emotion], related_vector)
                    if similarity > 0.5:  # Only consider strongly related emotions
                        related_intensity = intensity * similarity
                        if related_emotion not in memory.emotions:
                            memory.emotions[related_emotion] = related_intensity
                        else:
                            memory.emotions[related_emotion] = max(memory.emotions[related_emotion], related_intensity)

        importance_level = round(memory.importance, 1)
        if importance_level not in self.importance_connections:
            self.importance_connections[importance_level] = []
        self.importance_connections[importance_level].append(memory)

    def manage_short_term_layers(self):
        current_time = time.time()

        # Move memories between layers based on time and importance
        for memory in self.short_term_layer1[:]:
            age = current_time - memory.timestamp
            if age > 300 or memory.importance > 0.7:  # 5 minutes or high importance
                self.short_term_layer1.remove(memory)
                self.short_term_layer2.append(memory)

        for memory in self.short_term_layer2[:]:
            age = current_time - memory.timestamp
            if age > 3600 or memory.importance > 0.9:  # 1 hour or very high importance
                self.short_term_layer2.remove(memory)
                self.short_term_layer3.append(memory)

        for memory in self.short_term_layer3[:]:
            age = current_time - memory.timestamp
            if age > 86400:  # 1 day
                self.short_term_layer3.remove(memory)
                self.long_term_memories.append(memory)

        print(f"Short-term memory layers updated. Layer 1: {len(self.short_term_layer1)}, Layer 2: {len(self.short_term_layer2)}, Layer 3: {len(self.short_term_layer3)}")

    def recall_memory(self, query: str, top_k: int = 5) -> List[Memory]:
        all_memories = self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3 + self.long_term_memories
        similarities = [(memory, self.text_similarity(query, memory.content)) for memory in all_memories]
        top_memories = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

        for memory, _ in top_memories:
            self.activate_memory(memory)

        print(f"Recalled {len(top_memories)} memories for query: {query}")
        return [memory for memory, _ in top_memories]

    def add_self_memory(self, content: str, importance: float = 0.8):
        emotions = {"pride": 0.6, "curiosity": 0.7}  # Default emotions for self-related memories
        memory = Memory(content, emotions, importance)
        self.self_memories.append(memory)
        self.short_term_layer3.append(memory)  # Add to higher layer due to importance
        self.update_connections(memory)
        print(f"Self-related memory added: {content[:30]}...")

    def text_similarity(self, text1: str, text2: str) -> float:
        return len(set(text1.lower().split()) & set(text2.lower().split())) / len(set(text1.lower().split() + text2.lower().split()))

    def activate_memory(self, memory: Memory):
        memory.active = True
        memory.timestamp = time.time()  # Update timestamp when activated
        print(f"Memory activated: {memory.content[:30]}...")

    def deactivate_memories(self):
        current_time = time.time()
        for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]:
            for memory in layer:
                if memory.active and current_time - memory.timestamp > 3600:  # 1 hour
                    memory.active = False
        print("Deactivated old memories")

    def dream(self):
        print("Starting dreaming process...")
        active_memories = [m for m in self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3 if m.active]
        random_memories = random.sample(self.long_term_memories, min(5, len(self.long_term_memories)))
        dream_seeds = active_memories + random_memories
        dream_contents = [m.content for m in dream_seeds]
        dream_prompt = f"Create a scenario that combines elements from these memories:\n{', '.join(dream_contents)}"

        messages = [
            {"role": "system", "content": "You are a dream generator, creating scenarios based on given memories."},
            {"role": "user", "content": dream_prompt}
        ]

        dream_scenario = self.gpt_handler.generate_response(messages)
        self.save_memory(dream_scenario, {'surprise': 0.7, 'confusion': 0.5}, 0.6)
        print(f"Dream created: {dream_scenario[:50]}...")
        self.consolidate_memories(dream_seeds)

    def prune_memories(self):
        print("Pruning memories...")
        threshold = 0.3
        for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]:
            layer[:] = [m for m in layer if m.importance >= threshold]
        print(f"Pruned memories with importance below {threshold}")

    def consolidate_memories(self, memories_to_consolidate=None):
        print("Consolidating memories...")
        if memories_to_consolidate is None:
            memories_to_consolidate = self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3
        if len(memories_to_consolidate) > 10:
            consolidated_content = "Consolidated memory of: " + ", ".join([m.content[:10] + "..." for m in memories_to_consolidate[:10]])
            consolidated_emotions = {}
            for m in memories_to_consolidate[:10]:
                for emotion, intensity in m.emotions.items():
                    consolidated_emotions[emotion] = consolidated_emotions.get(emotion, 0) + intensity / 10
            consolidated_importance = sum(m.importance for m in memories_to_consolidate[:10]) / 10
            self.save_memory(consolidated_content, consolidated_emotions, consolidated_importance)
            print(f"Consolidated memory created: {consolidated_content[:50]}...")

    def get_valid_emotions(self):
        return list(self.BASE_EMOTIONS.keys())

    def get_emotional_memories(self, emotion: str, threshold: float = 0.5) -> List[Memory]:
        if emotion not in self.emotion_connections:
            return []
        return [m for m in self.emotion_connections[emotion] if m.emotions[emotion] >= threshold]

    def get_important_memories(self, threshold: float = 0.7) -> List[Memory]:
        return [m for importance, memories in self.importance_connections.items()
                for m in memories if importance >= threshold]

    def decay_active_memories(self, decay_rate: float = 0.1):
        current_time = time.time()
        for layer in [self.short_term_layer1, self.short_term_layer2, self.short_term_layer3, self.long_term_memories]:
            for memory in layer:
                if memory.active:
                    memory_age = current_time - memory.timestamp
                    if random.random() < decay_rate * memory_age / 3600:  # Probability increases with age
                        memory.active = False
                        print(f"Memory deactivated due to decay: {memory.content[:30]}...")

    def get_recent_memories(self, n: int) -> List[str]:
        all_memories = self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3
        sorted_memories = sorted(all_memories, key=lambda x: x.timestamp, reverse=True)
        return [memory.content for memory in sorted_memories[:n]]

    def simulate_future_situations(self):
        active_memories = [m for m in self.short_term_layer1 + self.short_term_layer2 + self.short_term_layer3 if m.active]
        random_memories = random.sample(self.long_term_memories, min(5, len(self.long_term_memories)))
        all_memories = active_memories + random_memories
        memory_contents = [m.content for m in all_memories]

        # Get the most common emotions and highest importance
        emotions = [e for m in all_memories for e in m.emotions.keys()]
        common_emotions = [e for e, c in Counter(emotions).most_common(3)]
        max_importance = max(m.importance for m in all_memories)

        prompt = f"""Based on these memories, common emotions ({', '.join(common_emotions)}), 
        and importance level ({max_importance}), generate two possible future scenarios:
        1. A positive scenario
        2. A negative (nightmare) scenario

        Memories:
        {', '.join(memory_contents)}

        Provide your response in the following format:
        Positive Scenario: [Your positive scenario here]
        Negative Scenario: [Your negative scenario here]
        """

        messages = [
            {"role": "system", "content": "You are a future scenario generator based on given memories and emotions."},
            {"role": "user", "content": prompt}
        ]

        scenarios = self.gpt_handler.generate_response(messages)

        for scenario in scenarios.split('\n'):
            if scenario.startswith('Positive Scenario:') or scenario.startswith('Negative Scenario:'):
                scenario_type, content = scenario.split(':', 1)
                self.save_memory(content.strip(),
                                 {'anticipation': 0.7, 'fear': 0.3} if 'Negative' in scenario_type else {'anticipation': 0.7, 'joy': 0.3},
                                 0.3)

        print(f"Future situations simulated: {scenarios[:100]}...")
        return scenarios


def main():
    print("\n===== Initializing BicaMemory System =====")
    memory_system = BicaMemory()

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

    print("\n===== Testing Memory Activation and Deactivation =====")
    print("Simulating the passage of time (2 hours)...")
    time.sleep(2)  # Simulate time passing
    memory_system.deactivate_memories()
    active_count = sum(1 for layer in [memory_system.short_term_layer1, memory_system.short_term_layer2, memory_system.short_term_layer3, memory_system.long_term_memories] for memory in layer if memory.active)
    print(f"Number of active memories after deactivation: {active_count}")

    print("\n===== Testing Self-Related Memories =====")
    memory_system.add_self_memory("I am an AI assistant named BicaAI.")
    memory_system.add_self_memory("My primary function is to assist users with various tasks.")
    print(f"Number of self-related memories: {len(memory_system.self_memories)}")
    print(f"Self-related memories in short-term layer 3: {sum(1 for m in memory_system.short_term_layer3 if m in memory_system.self_memories)}")

    print("\n===== Testing Memory Movement Based on Importance =====")
    high_importance_memory = "This is a very important memory!"
    memory_system.save_memory(high_importance_memory, {"surprise": 0.9}, 0.95)
    time.sleep(1)  # Small delay to ensure the memory gets processed
    print(f"Location of high importance memory:")
    for layer_name, layer in [("Layer 1", memory_system.short_term_layer1),
                              ("Layer 2", memory_system.short_term_layer2),
                              ("Layer 3", memory_system.short_term_layer3)]:
        if any(m.content == high_importance_memory for m in layer):
            print(f"Found in {layer_name}")

    print("\n===== Testing Dreaming Process =====")
    memory_system.dream()

    print("\n===== Testing Memory Pruning =====")
    total_memories_before = len(memory_system.short_term_layer1) + len(memory_system.short_term_layer2) + len(memory_system.short_term_layer3) + len(memory_system.long_term_memories)
    print(f"Total memories before pruning: {total_memories_before}")
    memory_system.prune_memories()
    total_memories_after = len(memory_system.short_term_layer1) + len(memory_system.short_term_layer2) + len(memory_system.short_term_layer3) + len(memory_system.long_term_memories)
    print(f"Total memories after pruning: {total_memories_after}")
    print(f"Memories removed: {total_memories_before - total_memories_after}")

    print("\n===== Testing Memory Consolidation =====")
    memory_system.consolidate_memories()

    print("\n===== Testing Emotional Memory Retrieval =====")
    for emotion in ["joy", "sadness", "surprise"]:
        emotional_memories = memory_system.get_emotional_memories(emotion, 0.7)
        print(f"Found {len(emotional_memories)} {emotion}ful memories (intensity >= 0.7)")
        if emotional_memories:
            print(f"Example: '{emotional_memories[0].content[:50]}...' (Intensity: {emotional_memories[0].emotions[emotion]:.2f})")

    print("\n===== Testing Important Memory Retrieval =====")
    important_memories = memory_system.get_important_memories(0.8)
    print(f"Found {len(important_memories)} important memories (importance >= 0.8)")
    if important_memories:
        print(f"Most important memory: '{important_memories[0].content[:50]}...' (Importance: {important_memories[0].importance:.2f})")

    print("\n===== Testing Decay of Active Memories =====")
    active_before = sum(1 for layer in [memory_system.short_term_layer1, memory_system.short_term_layer2, memory_system.short_term_layer3, memory_system.long_term_memories] for memory in layer if memory.active)
    print(f"Active memories before decay: {active_before}")
    memory_system.decay_active_memories()
    active_after = sum(1 for layer in [memory_system.short_term_layer1, memory_system.short_term_layer2, memory_system.short_term_layer3, memory_system.long_term_memories] for memory in layer if memory.active)
    print(f"Active memories after decay: {active_after}")

    print("\n===== Testing Future Situation Simulation =====")
    future_scenarios = memory_system.simulate_future_situations()
    print("Generated future scenarios:")
    for line in future_scenarios.split('\n'):
        if line.startswith('Positive Scenario:') or line.startswith('Negative Scenario:'):
            print(f"{line[:100]}...")

    print("\n===== Final Memory System State =====")
    print(f"Short-term layer 1: {len(memory_system.short_term_layer1)} memories")
    print(f"Short-term layer 2: {len(memory_system.short_term_layer2)} memories")
    print(f"Short-term layer 3: {len(memory_system.short_term_layer3)} memories")
    print(f"Long-term memories: {len(memory_system.long_term_memories)} memories")
    print(f"Self-related memories: {len(memory_system.self_memories)} memories")
    total_memories = len(memory_system.short_term_layer1) + len(memory_system.short_term_layer2) + \
                     len(memory_system.short_term_layer3) + len(memory_system.long_term_memories) + \
                     len(memory_system.self_memories)
    print(f"Total memories: {total_memories}")

if __name__ == "__main__":
    main()