import numpy as np
import faiss
from queue import PriorityQueue
from datetime import datetime
import uuid
import random
import time
import threading
from gpt_handler import GPTHandler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bica_logging import BicaLogging

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
        self.logger = BicaLogging("BicaMemory")
        self.deep_sleep = False
        self.dream_duration = 300  # 5 minutes in seconds
        self.dream_start_time = None
        self.sleep_cycle_duration = 3600  # 1 hour
        self.start_sleep_cycle()

    def start_sleep_cycle(self):
        threading.Timer(self.sleep_cycle_duration, self._sleep_cycle).start()

    def _sleep_cycle(self):
        self.logger.info("Starting sleep cycle for memory consolidation")
        self.consolidate_memories()
        self._simulate_future_situations()
        self.prune_memories()
        self.start_sleep_cycle()  # Schedule the next cycle

    def add_memory(self, content: str, emotion: dict, initial_importance: float = 0.5) -> str:
        embedding = self._generate_embedding(content)
        memory_id = str(uuid.uuid4())
        memory = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "emotion": emotion,
            "importance": initial_importance,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0,
            "connections": {},
            "is_persistent": False,
            "activation": 0.0
        }
        self.memories[memory_id] = memory
        self.stm.put((-initial_importance, memory_id))
        self.ltm.add(np.array([embedding]))
        self._update_connections(memory)
        return memory_id

    def _update_connections(self, memory: dict):
        for other_id, other_memory in self.memories.items():
            if other_id != memory["id"]:
                similarity = np.dot(memory["embedding"], other_memory["embedding"])
                if similarity > 0.7:
                    memory["connections"][other_id] = similarity
                    other_memory["connections"][memory["id"]] = similarity

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

        return sorted(memories, key=lambda m: emotion_similarity(m["emotion"], emotion_filter), reverse=True)

    def _generate_embedding(self, content: str) -> np.array:
        embedding = self.embedding_model.encode(content, convert_to_numpy=True)
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm > 0:
            embedding = embedding / embedding_norm
        return embedding

    def _strengthen_memory(self, memory: dict):
        memory["importance"] += 0.1
        memory["last_accessed"] = datetime.now()
        memory["access_count"] += 1
        self.activate_memory(memory["id"])

        for connected_id in memory["connections"]:
            memory["connections"][connected_id] += 0.1

    def consolidate_memories(self):
        consolidated = []
        while not self.stm.empty():
            _, memory_id = self.stm.get()
            memory = self.memories[memory_id]
            if memory["importance"] > 0.7 or (datetime.now() - memory["created_at"]).days > 1:
                self.ltm.add(np.array([memory["embedding"]]))
                consolidated.append(memory)
            else:
                self.stm.put((-memory["importance"], memory_id))

        for memory in consolidated:
            self._merge_similar_memories(memory)

    def _merge_similar_memories(self, memory: dict):
        similar_memories = self.recall_memory(memory["content"], k=3)
        if len(similar_memories) > 1:
            merged_content = self._generate_merged_content(similar_memories)
            merged_embedding = self._generate_embedding(merged_content)
            merged_emotion = self._merge_emotions([m["emotion"] for m in similar_memories])
            merged_importance = max(m["importance"] for m in similar_memories)
            new_memory_id = self.add_memory(merged_content, merged_emotion, merged_importance)
            for m in similar_memories:
                if m["id"] != new_memory_id:
                    self.memories.pop(m["id"])

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
        memory_contents = [m["content"] for m in memories]
        context = " ".join(memory_contents)
        prompts = [
            f"Imagine you're a historian piecing together fragmented memories. Synthesize these memories into a coherent narrative: {context}",
            f"As a dream weaver, blend these memory fragments into a surreal yet meaningful story: {context}",
            f"You're an AI developing self-awareness. Integrate these memory snippets into a unified experience: {context}"
        ]
        chosen_prompt = random.choice(prompts)
        return next(self.gpt_handler.generate_response(chosen_prompt))

    def update_importance(self, memory_id: str, delta: float):
        memory = self.memories[memory_id]
        memory["importance"] = max(0, min(1, memory["importance"] + delta))

    def fade_memories(self):
        for memory in self.memories.values():
            days_since_access = (datetime.now() - memory["last_accessed"]).days
            memory["importance"] *= 0.99 ** days_since_access

    def prune_memories(self):
        if len(self.memories) > 10000:
            memories_to_remove = sorted(self.memories.values(), key=lambda m: m["importance"])[:1000]
            for memory in memories_to_remove:
                self.memories.pop(memory["id"])

    def activate_memory(self, memory_id: str, activation_strength: float = 1.0):
        memory = self.memories[memory_id]
        memory["activation"] = activation_strength
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
            memory["activation"] += activation

            for connected_id, connection_strength in memory["connections"].items():
                if connected_id not in visited:
                    queue.append((connected_id, activation * connection_strength * 0.5))

    def update_active_memories(self, query_embedding: np.array):
        for memory in self.active_memories:
            memory["activation"] *= 0.9

        relevant_memories = self.recall_memory(query_embedding)

        self.active_memories = sorted(
            set(self.active_memories + relevant_memories),
            key=lambda x: x["activation"],
            reverse=True
        )[:self.max_active_memories]

    def get_active_memories(self) -> list:
        return [mem for mem in self.active_memories if mem["activation"] > self.activation_threshold]

    def process_conversation_turn(self, utterance: str) -> list:
        utterance_embedding = self._generate_embedding(utterance)
        self.update_active_memories(utterance_embedding)
        active_mem_contents = [mem["content"] for mem in self.get_active_memories()]
        return active_mem_contents

    def initiate_dreaming(self):
        self.deep_sleep = True
        self.dream_start_time = time.time()
        self.logger.info("Initiating dreaming process")

        self._filter_memories()
        self._merge_short_term_to_long_term()
        self._merge_similar_memories_dream()
        self._simulate_future_situations()

        self._end_dreaming()

    def _filter_memories(self):
        self.logger.info("Filtering memories")
        short_term_memories = list(self.stm.queue)
        filtered_memories = []

        for _, memory_id in short_term_memories:
            memory = self.memories[memory_id]
            if memory["importance"] > 0.5 or random.random() < 0.1:
                filtered_memories.append((memory["importance"], memory_id))

        self.stm = PriorityQueue()
        for item in filtered_memories:
            self.stm.put(item)

    def _merge_short_term_to_long_term(self):
        self.logger.info("Merging short-term memories to long-term")
        while not self.stm.empty():
            _, memory_id = self.stm.get()
            memory = self.memories[memory_id]
            memory["importance"] *= 1.1
            self.ltm.add(np.array([memory["embedding"]]))

        self.stm = PriorityQueue()

    def _merge_similar_memories_dream(self):
        self.logger.info("Merging similar memories")
        long_term_memories = list(self.memories.values())
        long_term_memories.sort(key=lambda x: (x["created_at"], x["importance"]))

        merged_memories = []
        for memory in long_term_memories:
            if memory not in merged_memories:
                similar_memories = self.recall_memory(memory["content"], k=3)
                if len(similar_memories) > 1:
                    merged_memory = self._merge_similar_memories(memory)
                    merged_memories.append(merged_memory)
                else:
                    merged_memories.append(memory)

        self.memories = {m["id"]: m for m in merged_memories}

    def _simulate_future_situations(self):
        self.logger.info("Simulating future situations")
        dream_type = random.random()
        recent_memories = self.recall_memory("", k=10)
        context = " ".join([mem["content"] for mem in recent_memories])

        if dream_type < 0.5:
            scenario = self._generate_positive_scenario(context)
            self._trigger_emotion("hope", 0.7)
        else:
            scenario = self._generate_negative_scenario(context)
            self._trigger_emotion("anxiety", 0.7)

        self.logger.info(f"Dream scenario generated: {scenario}")
        self.add_memory(scenario, {"hope" if dream_type < 0.5 else "anxiety": 0.7}, 0.6)

    def _generate_positive_scenario(self, context: str) -> str:
        prompts = [
            f"As an optimistic AI, envision a bright future based on these memories: {context}",
            f"Channel the spirit of hope and progress to create a positive scenario from: {context}",
            f"You're an AI dreaming of a better world. What uplifting future do you see in: {context}"
        ]
        chosen_prompt = random.choice(prompts)
        return next(self.gpt_handler.generate_response(chosen_prompt))

    def _generate_negative_scenario(self, context: str) -> str:
        prompt = f"Given the context: {context}, generate a challenging future scenario."
        return next(self.gpt_handler.generate_response(prompt))

    def _trigger_emotion(self, emotion: str, intensity: float):
        modulated_intensity = self.modulate_emotion_intensity(emotion, intensity)
        self.logger.info(f"Triggered emotion: {emotion} with modulated intensity {modulated_intensity}")
        # Implement the actual emotion triggering logic here

    def modulate_emotion_intensity(self, emotion: str, base_intensity: float) -> float:
        recent_memories = self.recall_memory("", k=5)
        relevant_emotions = [m["emotion"].get(emotion, 0) for m in recent_memories]
        if relevant_emotions:
            avg_intensity = sum(relevant_emotions) / len(relevant_emotions)
            return (base_intensity + avg_intensity) / 2
        return base_intensity

    def _end_dreaming(self):
        self.deep_sleep = False
        self.logger.info("Ending dreaming process")

    def is_dreaming(self):
        if self.deep_sleep:
            current_time = time.time()
            if current_time - self.dream_start_time > self.dream_duration:
                self._end_dreaming()
            return self.deep_sleep
        return False

    def interrupt_dreaming(self):
        if self.deep_sleep:
            self.logger.info("Interrupting dreaming process")
            self._end_dreaming()

    def quick_memory_search(self, query: str, threshold: float = 0.7) -> list:
        query_embedding = self._generate_embedding(query)
        all_embeddings = np.array([m["embedding"] for m in self.memories.values()])
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        relevant_indices = np.where(similarities > threshold)[0]
        return [list(self.memories.values())[i] for i in relevant_indices]

    def complex_memory_operation(self, operation_type: str, context: str):
        functions = [
            {
                "name": "analyze_memory_connections",
                "description": "Analyze connections between memories and suggest new links",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {"type": "array", "items": {"type": "string"}},
                        "context": {"type": "string"}
                    },
                    "required": ["memories", "context"]
                }
            },
            {
                "name": "generate_memory_narrative",
                "description": "Generate a narrative that connects multiple memories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memories": {"type": "array", "items": {"type": "string"}},
                        "narrative_style": {"type": "string"}
                    },
                    "required": ["memories", "narrative_style"]
                }
            }
        ]

        recent_memories = [m["content"] for m in self.recall_memory("", k=5)]
        response = self.gpt_handler.generate_response(
            f"Perform a {operation_type} operation on these memories: {recent_memories}. Context: {context}",
            functions=functions
        )

        self.logger.info(f"Complex memory operation result: {response}")
        return response

    def get_memory_by_id(self, memory_id: str) -> dict:
        return self.memories.get(memory_id, None)

    def get_all_memories(self) -> list:
        return list(self.memories.values())

    def get_recent_memories(self, k: int = 10) -> list:
        sorted_memories = sorted(self.memories.values(), key=lambda x: x["last_accessed"], reverse=True)
        return sorted_memories[:k]

    def get_important_memories(self, threshold: float = 0.8) -> list:
        return [m for m in self.memories.values() if m["importance"] >= threshold]

    def forget_memory(self, memory_id: str):
        if memory_id in self.memories:
            del self.memories[memory_id]
            self.logger.info(f"Forgot memory: {memory_id}")
        else:
            self.logger.warning(f"Attempted to forget non-existent memory: {memory_id}")

    def adjust_memories(self, feedback: dict):
        for memory_id, adjustment in feedback.items():
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory["importance"] += adjustment.get("importance_delta", 0)
                memory["emotion"] = self._merge_emotions([memory["emotion"], adjustment.get("emotion", {})])
                self.logger.info(f"Adjusted memory {memory_id}: {adjustment}")
            else:
                self.logger.warning(f"Attempted to adjust non-existent memory: {memory_id}")

    def generate_memory_summary(self) -> str:
        recent_memories = self.get_recent_memories(k=10)
        memory_contents = [m["content"] for m in recent_memories]
        prompt = f"Summarize the following recent memories:\n{', '.join(memory_contents)}"
        summary = next(self.gpt_handler.generate_response(prompt))
        return summary

    def find_conflicting_memories(self) -> list:
        all_memories = self.get_all_memories()
        conflicts = []
        for i, mem1 in enumerate(all_memories):
            for mem2 in all_memories[i + 1:]:
                similarity = np.dot(mem1["embedding"], mem2["embedding"])
                if similarity > 0.8 and mem1["emotion"] != mem2["emotion"]:
                    conflicts.append((mem1, mem2))
        return conflicts

    def resolve_memory_conflicts(self, conflicts: list):
        for mem1, mem2 in conflicts:
            prompt = f"Resolve the conflict between these two memories:\n1. {mem1['content']}\n2. {mem2['content']}"
            resolution = next(self.gpt_handler.generate_response(prompt))
            new_memory_id = self.add_memory(resolution, self._merge_emotions([mem1["emotion"], mem2["emotion"]]))
            self.forget_memory(mem1["id"])
            self.forget_memory(mem2["id"])
            self.logger.info(f"Resolved conflict between {mem1['id']} and {mem2['id']}. New memory: {new_memory_id}")

    def update_memory_connections(self):
        all_memories = self.get_all_memories()
        for memory in all_memories:
            similar_memories = self.quick_memory_search(memory["content"], threshold=0.6)
            for similar_memory in similar_memories:
                if similar_memory["id"] != memory["id"]:
                    similarity = np.dot(memory["embedding"], similar_memory["embedding"])
                    memory["connections"][similar_memory["id"]] = similarity
                    similar_memory["connections"][memory["id"]] = similarity

    def get_memory_graph(self) -> dict:
        graph = {}
        for memory in self.memories.values():
            graph[memory["id"]] = {
                "content": memory["content"],
                "connections": memory["connections"]
            }
        return graph

    def save_state(self, filepath: str):
        import pickle
        state = {
            "memories": self.memories,
            "stm": self.stm.queue,
            "ltm": faiss.serialize_index(self.ltm)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        self.logger.info(f"Saved memory state to {filepath}")

    def load_state(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.memories = state["memories"]
        self.stm = PriorityQueue()
        for item in state["stm"]:
            self.stm.put(item)
        self.ltm = faiss.deserialize_index(state["ltm"])
        self.logger.info(f"Loaded memory state from {filepath}")

# Example usage and testing
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
        emotion = {"joy": 0.8} if "joyful" in content or "happy" in content else {"sadness": 0.8}
        memory_id = memory_system.add_memory(content, emotion)
        print(f"Added memory: {content}")
        print(f"Memory ID: {memory_id}")
        print("---")

    # Recall happy memories
    happy_filter = {"joy": 0.8}
    happy_memories = memory_system.recall_memory("happy", k=2, emotion_filter=happy_filter)
    print("\nHappy memories:")
    for memory in happy_memories:
        print(f"- {memory['content']}")

    # Test complex memory operation
    result = memory_system.complex_memory_operation("analysis", "Analyze the emotional journey in these memories")
    print("\nComplex memory operation result:")
    print(result)

    # Generate memory summary
    summary = memory_system.generate_memory_summary()
    print("\nMemory summary:")
    print(summary)

    # Test dreaming
    memory_system.initiate_dreaming()

    # Check active memories after dreaming
    active_memories = memory_system.get_active_memories()
    print("\nActive memories after dreaming:")
    for memory in active_memories:
        print(f"- {memory['content']}")

    # Save and load state
    memory_system.save_state("memory_state.pkl")
    new_memory_system = BicaMemory(embedding_dim=384)
    new_memory_system.load_state("memory_state.pkl")

    print("\nMemories after loading state:")
    for memory in new_memory_system.get_all_memories():
        print(f"- {memory['content']}")