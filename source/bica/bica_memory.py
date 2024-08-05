import numpy as np
import faiss
from queue import PriorityQueue
from datetime import datetime
import uuid
from gpt_handler import GPTHandler
import random
from sentence_transformers import SentenceTransformer
import time


class Memory:
    def __init__(self, content: str, embedding: np.array, emotion: dict, initial_importance: float = 0.5):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.emotion = emotion
        self.importance = initial_importance
        self.created_at = datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.connections = {}
        self.is_persistent = False
        self.activation = 0.0


class BicaMemory:
    def __init__(self, embedding_dim: int, api_provider="openai", model="gpt-4o-mini"):
        self.stm = PriorityQueue()
        self.ltm = faiss.IndexFlatL2(embedding_dim)
        self.memories = {}
        self.embedding_dim = embedding_dim
        self.activation_threshold = 0.5
        self.active_memories = []
        self.max_active_memories = 5
        self.gpt_handler = GPTHandler(api_provider=api_provider, model=model)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.deep_sleep = False
        self.dream_duration = 300  # 5 minutes in seconds
        self.dream_start_time = None

    def add_memory(self, content: str, embedding: np.array, emotion: dict, initial_importance: float = 0.5) -> str:
        memory = Memory(content, embedding, emotion, initial_importance)
        self.memories[memory.id] = memory
        self.stm.put((-memory.importance, memory.id))
        self.ltm.add(np.array([embedding]))
        self._update_connections(memory)
        return memory.id

    def _update_connections(self, memory: Memory):
        for other_id, other_memory in self.memories.items():
            if other_id != memory.id:
                similarity = np.dot(memory.embedding, other_memory.embedding)
                if similarity > 0.7:
                    memory.connections[other_id] = similarity
                    other_memory.connections[memory.id] = similarity

    def consolidate_memories(self):
        consolidated = []
        while not self.stm.empty():
            _, memory_id = self.stm.get()
            memory = self.memories[memory_id]
            if memory.importance > 0.7 or (datetime.now() - memory.created_at).days > 1:
                self.ltm.add(np.array([memory.embedding]))
                consolidated.append(memory)
            else:
                self.stm.put((-memory.importance, memory_id))

        for memory in consolidated:
            self._merge_similar_memories(memory)

    def _merge_similar_memories(self, memory: Memory):
        similar_memories = self.recall_memory(memory.content, k=3)
        if len(similar_memories) > 1:
            merged_content = self._generate_merged_content(similar_memories)
            merged_embedding = self._generate_embedding(merged_content)
            merged_emotion = self._merge_emotions([m.emotion for m in similar_memories])
            merged_importance = max(m.importance for m in similar_memories)
            new_memory_id = self.add_memory(merged_content, merged_embedding, merged_emotion, merged_importance)
            for m in similar_memories:
                if m.id != new_memory_id:
                    self.memories.pop(m.id)

    def _merge_emotions(self, emotions: list) -> dict:
        merged_emotion = {}
        for emotion in emotions:
            for key, value in emotion.items():
                if key in merged_emotion:
                    merged_emotion[key] = max(merged_emotion[key], value)
                else:
                    merged_emotion[key] = value
        return merged_emotion

    def _generate_merged_content(self, memories: list) -> str:
        memory_contents = [m.content for m in memories]
        context = " ".join(memory_contents)
        prompt = f"Merge the following related memories into a single, coherent memory: {context}"
        return next(self.gpt_handler.generate_response(prompt))

    def recall_memory(self, query: str, k: int = 5, emotion_filter: dict = None) -> list:
        query_embedding = self._generate_embedding(query)
        _, indices = self.ltm.search(np.array([query_embedding]), k * 2)
        recalled_memories = [self.memories[str(idx)] for idx in indices[0] if str(idx) in self.memories]

        if emotion_filter:
            recalled_memories = self._filter_by_emotion(recalled_memories, emotion_filter)

        for memory in recalled_memories[:k]:
            self._strengthen_memory(memory)

        return recalled_memories[:k]

    def _filter_by_emotion(self, memories: list, emotion_filter: dict) -> list:
        def emotion_similarity(mem_emotion, filter_emotion):
            return sum(mem_emotion.get(e, 0) * v for e, v in filter_emotion.items())

        return sorted(memories, key=lambda m: emotion_similarity(m.emotion, emotion_filter), reverse=True)

    def _generate_embedding(self, content: str) -> np.array:
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        return embedding

    def _strengthen_memory(self, memory: Memory):
        memory.importance += 0.1
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        self.activate_memory(memory.id)

        for connected_id in memory.connections:
            memory.connections[connected_id] += 0.1

    def update_importance(self, memory_id: str, delta: float):
        memory = self.memories[memory_id]
        memory.importance = max(0, min(1, memory.importance + delta))

    def fade_memories(self):
        for memory in self.memories.values():
            days_since_access = (datetime.now() - memory.last_accessed).days
            memory.importance *= 0.99 ** days_since_access

    def prune_memories(self):
        if len(self.memories) > 10000:
            memories_to_remove = sorted(self.memories.values(), key=lambda m: m.importance)[:1000]
            for memory in memories_to_remove:
                self.memories.pop(memory.id)

    def decay_importance(self):
        for memory in self.memories.values():
            age_factor = (datetime.now() - memory.created_at).days / 365
            access_factor = 1 / (memory.access_count + 1)
            decay_rate = 0.1 * age_factor * access_factor
            memory.importance *= (1 - decay_rate)

    def shift_connections(self):
        for memory in self.memories.values():
            for other_id in list(memory.connections.keys()):
                similarity = np.dot(memory.embedding, self.memories[other_id].embedding)
                if similarity < 0.5:
                    memory.connections.pop(other_id)
                else:
                    memory.connections[other_id] = similarity

    def set_persistent_focus(self, memory_id: str, is_persistent: bool):
        memory = self.memories[memory_id]
        memory.is_persistent = is_persistent

    def get_persistent_focus(self) -> list:
        return [mem for mem in self.memories.values() if mem.is_persistent]

    def activate_memory(self, memory_id: str, activation_strength: float = 1.0):
        memory = self.memories[memory_id]
        memory.activation = activation_strength
        self._spread_activation(memory_id)

    def _spread_activation(self, start_memory_id: str):
        queue = [(start_memory_id, 1.0)]
        visited = set()

        while queue:
            memory_id, activation = queue.pop(0)
            if memory_id in visited:
                continue
            visited.add(memory_id)

            memory = self.memories[memory_id]
            memory.activation += activation

            for connected_id, connection_strength in memory.connections.items():
                if connected_id not in visited:
                    queue.append((connected_id, activation * connection_strength * 0.5))

    def update_active_memories(self, query_embedding: np.array):
        for memory in self.active_memories:
            memory.activation *= 0.9

        relevant_memories = self.recall_memory(query_embedding)

        self.active_memories = sorted(
            set(self.active_memories + relevant_memories),
            key=lambda x: x.activation,
            reverse=True
        )[:self.max_active_memories]

    def get_active_memories(self) -> list:
        return [mem for mem in self.active_memories if mem.activation > self.activation_threshold]

    def process_conversation_turn(self, utterance: str) -> list:
        utterance_embedding = self._generate_embedding(utterance)
        self.update_active_memories(utterance_embedding)
        active_mem_contents = [mem.content for mem in self.get_active_memories()]
        return active_mem_contents

    # Dreaming functionality

    def initiate_dreaming(self):
        self.deep_sleep = True
        self.dream_start_time = time.time()
        print("Initiating dreaming process")

        self._filter_memories()
        self._merge_short_term_to_long_term()
        self._merge_similar_memories_dream()
        self._simulate_future_situations()

        self._end_dreaming()

    def _filter_memories(self):
        print("Filtering memories")
        short_term_memories = list(self.stm.queue)
        filtered_memories = []

        for _, memory_id in short_term_memories:
            memory = self.memories[memory_id]
            if memory.importance > 0.5 or random.random() < 0.1:
                filtered_memories.append((memory.importance, memory_id))

        self.stm = PriorityQueue()
        for item in filtered_memories:
            self.stm.put(item)

    def _merge_short_term_to_long_term(self):
        print("Merging short-term memories to long-term")
        while not self.stm.empty():
            _, memory_id = self.stm.get()
            memory = self.memories[memory_id]
            memory.importance *= 1.1
            self.ltm.add(np.array([memory.embedding]))

        self.stm = PriorityQueue()

    def _merge_similar_memories_dream(self):
        print("Merging similar memories")
        long_term_memories = list(self.memories.values())
        long_term_memories.sort(key=lambda x: (x.created_at, x.importance))

        merged_memories = []
        for i, memory in enumerate(long_term_memories):
            if memory not in merged_memories:
                similar_memories = self.recall_memory(memory.content, k=3)
                if len(similar_memories) > 1:
                    merged_memory = self._merge_similar_memories(memory)
                    merged_memories.append(merged_memory)
                else:
                    merged_memories.append(memory)

        self.memories = {m.id: m for m in merged_memories}

    def _simulate_future_situations(self):
        print("Simulating future situations")
        dream_type = random.random()
        recent_memories = self.recall_memory("", k=10)
        context = " ".join([mem.content for mem in recent_memories])

        if dream_type < 0.5:
            scenario = self._generate_positive_scenario(context)
        else:
            scenario = self._generate_negative_scenario(context)

        print(f"Dream scenario generated: {scenario}")
        self.add_memory(scenario, self._generate_embedding(scenario), {"hope" if dream_type < 0.5 else "anxiety": 0.7}, 0.6)

    def _generate_positive_scenario(self, context: str) -> str:
        prompt = f"Given the context: {context}, generate a positive future scenario."
        return next(self.gpt_handler.generate_response(prompt))

    def _generate_negative_scenario(self, context: str) -> str:
        prompt = f"Given the context: {context}, generate a challenging future scenario."
        return next(self.gpt_handler.generate_response(prompt))

    def _end_dreaming(self):
        self.deep_sleep = False
        print("Ending dreaming process")

    def is_dreaming(self):
        if self.deep_sleep:
            current_time = time.time()
            if current_time - self.dream_start_time > self.dream_duration:
                self._end_dreaming()
            return self.deep_sleep
        return False

    def interrupt_dreaming(self):
        if self.deep_sleep:
            print("Interrupting dreaming process")
            self._end_dreaming()


# Example usage
if __name__ == "__main__":
    memory_system = BicaMemory(embedding_dim=384)

    # Add some sample memories
    memories = [
        "I felt joyful when I saw the beautiful sunset.",
        "The news of my promotion made me very happy.",
        "I was sad when my favorite plant died.",
        "The thrilling roller coaster ride excited me."
    ]

    for content in memories:
        embedding = memory_system._generate_embedding(content)
        emotion = {"joy": 0.8} if "joyful" in content or "happy" in content else {"sadness": 0.8}
        memory_id = memory_system.add_memory(content, embedding, emotion)
        print(f"Added memory: {content}")
        print(f"Memory ID: {memory_id}")
        print("---")

    # Recall happy memories
    happy_filter = {"joy": 0.8}
    happy_memories = memory_system.recall_memory("happy", k=2, emotion_filter=happy_filter)
    print("\nHappy memories:")
    for memory in happy_memories:
        print(f"- {memory.content}")

    # Initiate dreaming
    memory_system.initiate_dreaming()

    # Check active memories after dreaming
    active_memories = memory_system.get_active_memories()
    print("\nActive memories after dreaming:")
    for memory in active_memories:
        print(f"- {memory.content}")

"""
ARCHIVED VERSION FOR REFERENCE

import random
import time
from bica_memory import BicaMemory
from bica_thoughts import BicaThoughts
from bica_emotions import BicaEmotions
from bica_logging import BicaLogging


class BicaDreaming:
    def __init__(self, memory_system: BicaMemory, thought_system: BicaThoughts, emotion_system: BicaEmotions):
        self.memory = memory_system
        self.thoughts = thought_system
        self.emotions = emotion_system
        self.logger = BicaLogging("BicaDreaming")
        self.deep_sleep = False
        self.dream_duration = 300  # 5 minutes in seconds
        self.dream_start_time = None

    def initiate_dreaming(self):
        ""
        Initiates the dreaming process when triggered.
        This function coordinates the entire dreaming cycle.
        ""
        self.deep_sleep = True
        self.dream_start_time = time.time()
        self.logger.info("Initiating dreaming process")

        self._filter_memories()
        self._merge_short_term_to_long_term()
        self._merge_similar_memories()
        self._simulate_future_situations()

        self._end_dreaming()

    def _filter_memories(self):
        ""
        Step 1 of dreaming process: Filter out unimportant memories from short-term memory.
        Randomly retains a few unimportant memories to simulate brain randomness.
        ""
        self.logger.info("Filtering memories")
        short_term_memories = self.memory.get_short_term_memories()
        filtered_memories = []

        for memory in short_term_memories:
            if memory.importance > 0.5 or random.random() < 0.1:  # Keep important or random memories
                filtered_memories.append(memory)

        self.memory.set_short_term_memories(filtered_memories)

    def _merge_short_term_to_long_term(self):
        ""
        Step 2 of dreaming process: Merge short-term memories into long-term memory.
        Gives a slight importance boost to these memories as they are newer.
        ""
        self.logger.info("Merging short-term memories to long-term")
        short_term_memories = self.memory.get_short_term_memories()

        for memory in short_term_memories:
            memory.importance *= 1.1  # Slight importance boost
            self.memory.add_to_long_term_memory(memory)

        self.memory.clear_short_term_memory()

    def _merge_similar_memories(self):
        ""
        Step 3 of dreaming process: Merge similar memories in long-term memory.
        Starts with the furthest and least important memories.
        ""
        self.logger.info("Merging similar memories")
        long_term_memories = self.memory.get_long_term_memories()
        long_term_memories.sort(key=lambda x: (x.timestamp, x.importance))

        merged_memories = []
        for i, memory in enumerate(long_term_memories):
            if memory not in merged_memories:
                similar_memories = self.memory.find_similar_memories(memory, long_term_memories[i + 1:])
                if similar_memories:
                    merged_memory = self.memory.merge_memories([memory] + similar_memories)
                    merged_memories.append(merged_memory)
                else:
                    merged_memories.append(memory)

        self.memory.set_long_term_memories(merged_memories)

    def _simulate_future_situations(self):
        ""
        Step 4 of dreaming process: Simulate future situations based on previous context and memories.
        Uses a random value to determine if the dream scenario is positive (0) or negative (1).
        ""
        self.logger.info("Simulating future situations")
        dream_type = random.random()  # 0 for positive, 1 for negative
        recent_memories = self.memory.get_recent_memories(10)
        context = " ".join([mem.content for mem in recent_memories])

        if dream_type < 0.5:
            scenario = self.thoughts.generate_positive_scenario(context)
            self.emotions.trigger_emotion("hope", 0.7)
        else:
            scenario = self.thoughts.generate_negative_scenario(context)
            self.emotions.trigger_emotion("anxiety", 0.7)

        self.logger.info(f"Dream scenario generated: {scenario}")
        self.memory.add_to_short_term_memory(scenario, importance=0.6)

    def _end_dreaming(self):
        ""Ends the dreaming process and sets deep_sleep to False.""
        self.deep_sleep = False
        self.logger.info("Ending dreaming process")

    def is_dreaming(self):
        ""
        Checks if the AI is currently in a dreaming state.
        Also handles the transition out of dreaming based on time limit.
        ""
        if self.deep_sleep:
            current_time = time.time()
            if current_time - self.dream_start_time > self.dream_duration:
                self._end_dreaming()
            return self.deep_sleep
        return False

    def interrupt_dreaming(self):
        ""
        Allows for interruption of the dreaming process.
        This can be called by the orchestrator if needed.
        ""
        if self.deep_sleep:
            self.logger.info("Interrupting dreaming process")
            self._end_dreaming()
"""