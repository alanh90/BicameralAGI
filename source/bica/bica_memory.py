import numpy as np
import faiss
from queue import PriorityQueue
from datetime import datetime
from typing import List, Dict
import uuid
from gpt_handler import GPTHandler
import random
from sentence_transformers import SentenceTransformer
import numpy as np


class Memory:
    def __init__(self, content: str, embedding: np.array, initial_importance: float = 0.5):
        self.id = str(uuid.uuid4())
        self.content = content
        self.embedding = embedding
        self.importance = initial_importance
        self.created_at = datetime.now()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.connections: Dict[str, float] = {}
        self.is_persistent = False
        self.activation = 0.0

class BicaMemory:
    def __init__(self, embedding_dim: int, api_provider="openai", model="gpt-4o-mini"):
        self.stm = PriorityQueue()
        self.ltm = faiss.IndexFlatL2(embedding_dim)
        self.memories: Dict[str, Memory] = {}
        self.embedding_dim = embedding_dim
        self.activation_threshold = 0.5
        self.active_memories: List[Memory] = []
        self.max_active_memories = 5
        self.gpt_handler = GPTHandler(api_provider=api_provider, model=model)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        assert self.embedding_model.get_sentence_embedding_dimension() == embedding_dim, \
            f"Embedding dimension mismatch. Expected {embedding_dim}, got {self.embedding_model.get_sentence_embedding_dimension()}"

    def add_memory(self, content: str, embedding: np.array, initial_importance: float = 0.5) -> str:
        ai_memory = Memory(content, embedding, initial_importance)
        self.memories[ai_memory.id] = ai_memory
        self.stm.put((-ai_memory.importance, ai_memory.id))
        self._update_connections(ai_memory)
        return ai_memory.id

    def _update_connections(self, memory: Memory):
        # Update connections based on embedding similarity
        for other_id, other_memory in self.memories.items():
            if other_id != memory.id:
                similarity = np.dot(memory.embedding, other_memory.embedding)
                if similarity > 0.7:  # Threshold for considering memories as connected
                    memory.connections[other_id] = similarity
                    other_memory.connections[memory.id] = similarity

    def consolidate_memories(self):
        # Gradually move memories from STM to LTM based on importance and age
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
        # Merge similar memories to create more abstract representations
        similar_memories = self.recall_memory(memory.content, k=3)
        if len(similar_memories) > 1:
            merged_content = self._generate_merged_content(similar_memories)
            merged_embedding = self._generate_embedding(merged_content)
            merged_importance = max(m.importance for m in similar_memories)
            new_memory_id = self.add_memory(merged_content, merged_embedding, merged_importance)
            for m in similar_memories:
                if m.id != new_memory_id:
                    self.memories.pop(m.id)

    def _generate_merged_content(self, memories: List[Memory]) -> str:
        temperature = round(random.uniform(0.1, 0.4), 2)
        memory_contents = [m.content for m in memories]
        context = " ".join(memory_contents)

        prompt = f"""Merge the following related memories into a single, coherent memory. The merged memory should:
        - Capture the essential information from all input memories
        - Be more abstract or general than the individual memories
        - Be a single paragraph or a few concise sentences
        - Maintain the core meaning and importance of the original memories

        Original Memories:
        {context}

        Merged Memory:
        """

        stream = self.gpt_handler.generate_response(prompt, temperature=temperature, stream=True)
        merged_content = ""
        for chunk in stream:
            merged_content += chunk

        print(f"Temperature used for memory merging: {temperature}")
        return merged_content.strip()

    def recall_memory(self, query: str, k: int = 5) -> List[Memory]:
        query_embedding = self._generate_embedding(query)

        # First, check active memories
        active_relevant = [
            mem for mem in self.active_memories
            if np.dot(query_embedding, mem.embedding) > 0.8
        ]

        if active_relevant:
            return active_relevant[:k]

        # If no active memories are relevant, search all memories
        _, indices = self.ltm.search(np.array([query_embedding]), k)
        recalled_memories = [self.memories[str(idx)] for idx in indices[0] if str(idx) in self.memories]

        for memory in recalled_memories:
            self._strengthen_memory(memory)

        return recalled_memories

    def _generate_embedding(self, content: str) -> np.array:
        # Generate embedding using the Sentence Transformer model
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)

        # Normalize the embedding
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm

        return embedding

    def _strengthen_memory(self, memory: Memory):
        # Strengthen the memory and its connections
        memory.importance += 0.1
        memory.last_accessed = datetime.now()
        memory.access_count += 1
        self.activate_memory(memory.id)

        # Strengthen connections
        for connected_id in memory.connections:
            memory.connections[connected_id] += 0.1

    def update_importance(self, memory_id: str, delta: float):
        memory = self.memories[memory_id]
        memory.importance = max(0, min(1, memory.importance + delta))

    def fade_memories(self):
        # Gradually decrease memory importance over time
        for memory in self.memories.values():
            days_since_access = (datetime.now() - memory.last_accessed).days
            memory.importance *= 0.99 ** days_since_access

    def prune_memories(self):
        # Remove least important memories when exceeding a threshold
        if len(self.memories) > 10000:  # Arbitrary threshold
            memories_to_remove = sorted(self.memories.values(), key=lambda m: m.importance)[:1000]
            for memory in memories_to_remove:
                self.memories.pop(memory.id)

    def decay_importance(self):
        # Apply a more nuanced decay to memory importance
        for memory in self.memories.values():
            age_factor = (datetime.now() - memory.created_at).days / 365
            access_factor = 1 / (memory.access_count + 1)
            decay_rate = 0.1 * age_factor * access_factor
            memory.importance *= (1 - decay_rate)

    def shift_connections(self):
        # Periodically reassess and update memory connections
        for memory in self.memories.values():
            for other_id in list(memory.connections.keys()):
                similarity = np.dot(memory.embedding, self.memories[other_id].embedding)
                if similarity < 0.5:
                    memory.connections.pop(other_id)
                else:
                    memory.connections[other_id] = similarity

    def set_persistent_focus(self, memory_id: str, is_persistent: bool):
        # Mark or unmark a memory for persistent focus
        memory = self.memories[memory_id]
        memory.is_persistent = is_persistent

    def get_persistent_focus(self) -> List[Memory]:
        # Retrieve all persistent focus memories
        return [mem for mem in self.memories.values() if mem.is_persistent]

    def activate_memory(self, memory_id: str, activation_strength: float = 1.0):
        # Activate a memory and spread activation to connected memories
        memory = self.memories[memory_id]
        memory.activation = activation_strength
        self._spread_activation(memory_id)

    def _spread_activation(self, start_memory_id: str):
        # Implement spreading activation through connected memories
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
        # Update the set of active memories based on the current context
        for memory in self.active_memories:
            memory.activation *= 0.9

        relevant_memories = self.recall_memory(query_embedding)

        self.active_memories = sorted(
            set(self.active_memories + relevant_memories),
            key=lambda x: x.activation,
            reverse=True
        )[:self.max_active_memories]

    def get_active_memories(self) -> List[Memory]:
        # Retrieve all currently active memories
        return [mem for mem in self.active_memories if mem.activation > self.activation_threshold]

    def process_conversation_turn(self, utterance: str) -> List[str]:
        # Process a conversation turn, updating active memories
        utterance_embedding = self._generate_embedding(utterance)
        self.update_active_memories(utterance_embedding)
        active_mem_contents = [mem.content for mem in self.get_active_memories()]
        return active_mem_contents


# Example usage
if __name__ == "__main__":
    bica_memory = BicaMemory(embedding_dim=384, api_provider="openai", model="gpt-4o-mini")

    # Add some memories
    memory1_id = bica_memory.add_memory("Alice likes cats", bica_memory._generate_embedding("Alice likes cats"), 0.8)
    memory2_id = bica_memory.add_memory("Bob is allergic to peanuts", bica_memory._generate_embedding("Bob is allergic to peanuts"), 0.7)
    memory3_id = bica_memory.add_memory("Weather is sunny today", bica_memory._generate_embedding("Weather is sunny today"), 0.6)

    # Simulate a conversation
    print(bica_memory.process_conversation_turn("Tell me about Alice"))
    print(bica_memory.process_conversation_turn("What about Bob's allergies?"))
    print(bica_memory.process_conversation_turn("How's the weather?"))

    # Consolidate memories
    bica_memory.consolidate_memories()

    # Recall a memory
    recalled_memories = bica_memory.recall_memory("Alice pets")
    for memory in recalled_memories:
        print(f"Recalled: {memory.content}")

    # Set persistent focus
    bica_memory.set_persistent_focus(memory2_id, True)

    # Get persistent focus memories
    persistent_memories = bica_memory.get_persistent_focus()
    for memory in persistent_memories:
        print(f"Persistent focus: {memory.content}")

    # Fade memories
    bica_memory.fade_memories()

    # Prune memories
    bica_memory.prune_memories()
