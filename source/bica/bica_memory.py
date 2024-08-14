import pickle

from sentence_transformers import SentenceTransformer
from bica.bica_utilities import BicaUtilities
from bica.gpt_handler import GPTHandler


class BicaMemory:
    """
    A comprehensive memory system for AI, simulating human-like memory processes.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the BicaMemory system with necessary components such as
        short-term and long-term memory stores, embedding model, and other required attributes.
        I am also thinking that the importance score for memories should be its own embedding that relates abstractly to memories.

        for example:
        1) 0.23 Importance <- strong correlation [some softmax correlation value] -> [Memory 1], [Memory 35], [Memory 76], etc...
        2) 0.25 Importance <- weak correlation [some softmax correlation value] -> [Memory 3], [Memory 145], [Memory 123], etc...
        3) 0.9 Importance <- weak correlation [some softmax correlation value] -> [Memory 7], [Memory 345], [Memory 89], etc...
        4) 0.23 Importance <- weak correlation [some softmax correlation value] -> [Memory 3], [Memory 145], [Memory 123], etc...

        I kind of feel that memory and memory importance should be stored similarly, without strong ties with eachother. Like
        """
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()

        self.memories = []
        self.importance_embeddings = []
        self.emotion_embeddings = []
        self.active_memories = set()
        self.is_dreaming_state = False

        self.embedding_dim = embedding_dim

        self.BASE_EMOTIONS = {
            "joy": np.random.randn(embedding_dim),
            "sadness": np.random.randn(embedding_dim),
            "anger": np.random.randn(embedding_dim),
            "fear": np.random.randn(embedding_dim),
            "surprise": np.random.randn(embedding_dim),
            # Add more base emotions as needed
        }
        # Normalize base emotion vectors
        for emotion in self.BASE_EMOTIONS:
            self.BASE_EMOTIONS[emotion] /= np.linalg.norm(self.BASE_EMOTIONS[emotion])

    def initiate_dreaming(self):
        """
        Perform memory consolidation, pruning, and other maintenance tasks.
        This method simulates the memory processes that occur during sleep in biological systems.
        """
        self.is_dreaming_state = True
        self.consolidate_memories()
        self.fade_memories()
        self.prune_memories()
        self.generate_new_connections()

    def generate_new_connections(self):
        for memory in self.memories:
            potential_connections = self.recall_memory(memory['content'], top_k=5)
            for potential in potential_connections:
                if potential['content'] not in memory['connections']:
                    similarity = cosine_similarity([memory['embedding']], [potential['embedding']])[0][0]
                    if similarity > 0.6:  # Lower threshold for dreaming state
                        memory['connections'].append(potential['content'])
                        potential['connections'].append(memory['content'])

    def add_memory(self, content: str, metadata: Dict[str, Any]):
        """
        Add a new memory to the system. This should include generating an embedding,
        assigning an importance score, and storing the memory in the appropriate store (STM or LTM).
        It should also store a relationship between the top emotions during that memory. The way we do this is that emotions should have
        an embedded relationship via vectors to other memories, the same way normal memories would have their own relationships to other similar memories.

        Example:
            [Memory 1] <-> [Memory 2] <--> [Sadness: Intensity 0.8] <-> [Memory 3] <-> [Sadness: Intensity 0.6] <-> [Memory 4]

        The intensity of the emotions change the relationships as well. If the intensity of the emotion is too low, it doesn't save it into memory. The same way we don't care about unimportant memories.

        This should also automatically take care of updating the connections when a new memory is added.
        """
        embedding = self.embedding_model.encode([content])[0]
        importance = metadata.get('importance', 0.5)
        emotions = metadata.get('emotions', {})

        memory = {
            'content': content,
            'embedding': embedding,
            'importance': importance,
            'emotions': emotions,
            'connections': [],
            'access_count': 0,
            'last_accessed': self.utilities.generate_timestamp()
        }

        self.memories.append(memory)
        self.update_memory_connections(memory)
        self.update_emotion_connections(memory)

    def recall_memory(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories related to a given query. This should involve searching
        through the memory stores and returning the most relevant memories.
        This could also relate to recalling emotional memories. If someone asks what is your most happy memory.

        When a memory is recalled, it should strengthen it so that it stays longer in memory and it becomes more static and less prone to summarizing or alteration. Strengthening it also strengthens related connections slightly.
        """
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], [m['embedding'] for m in self.memories])[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]
        recalled_memories = [self.memories[i] for i in top_indices]

        for memory in recalled_memories:
            memory['access_count'] += 1
            memory['last_accessed'] = self.utilities.generate_timestamp()
            self.strengthen_memory(memory)

        return recalled_memories

    def generate_memory_embedding(self):
        """
        (Not sure if this is needed yet)
        Generate an embedding vector for a given piece of information.
        This is crucial for storing and comparing memories efficiently.
        """

    def consolidate_memories(self):
        """
        Move memories from short-term to long-term storage based on their importance
        and other factors. This is a key process in memory formation and retention.

        Also if there are too many similar long term memories that are not so important, they merge into less of them.

        For example:
        10 similar long term memories -> merge into 3 similar long term memories, but with an awareness somehow that there could be more.
        (or)
        3 similar long term memories could merge into 2 similar long term memories

        But they would never merge into 1. This way the AI is aware that there is probably more.

        This function should use gpthandler to summarize/merge memories.
        """
        similar_memories = self.find_similar_memories()
        for group in similar_memories:
            if len(group) > 3:
                merged_memories = self.merge_memories(group)
                for memory in group:
                    self.memories.remove(memory)
                self.memories.extend(merged_memories)

    def merge_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        contents = [m['content'] for m in memories]
        prompt = f"Summarize these related memories into two distinct but related memories:\n{contents}"
        merged_contents = self.gpt_handler.generate_response(prompt).split('\n')

        merged_memories = []
        for content in merged_contents[:2]:  # Ensure at least two memories
            merged_embedding = np.mean([m['embedding'] for m in memories], axis=0)
            merged_importance = np.mean([m['importance'] for m in memories])
            merged_emotions = self.merge_emotions([m['emotions'] for m in memories])

            merged_memory = {
                'content': content,
                'embedding': merged_embedding,
                'importance': merged_importance,
                'emotions': merged_emotions,
                'connections': list(set(sum([m['connections'] for m in memories], []))),
                'access_count': sum(m['access_count'] for m in memories),
                'last_accessed': max(m['last_accessed'] for m in memories)
            }
            merged_memories.append(merged_memory)

        return merged_memories

    def merge_emotions(self, emotion_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        merged = {}
        for emotions in emotion_dicts:
            for emotion, intensity in emotions.items():
                merged[emotion] = merged.get(emotion, 0) + intensity
        return {e: i / len(emotion_dicts) for e, i in merged.items()}

    def find_similar_memories(self, similarity_threshold: float = 0.8) -> List[List[Dict[str, Any]]]:
        similar_groups = []
        for i, memory in enumerate(self.memories):
            group = [memory]
            for other_memory in self.memories[i + 1:]:
                similarity = cosine_similarity([memory['embedding']], [other_memory['embedding']])[0][0]
                if similarity > similarity_threshold:
                    group.append(other_memory)
            if len(group) > 1:
                similar_groups.append(group)
        return similar_groups

    def strengthen_memory(self, memory: Dict[str, Any]):
        memory['importance'] = min(1.0, memory['importance'] * 1.1)
        # Activate this memory
        self.activate_memory(memory)

    def increase_memory_importance(self):
        """
        Increase the importance score of a memory when called. It will need to be based on whatever context imported into this function.
        """

    def decrease_memory_importance(self):
        """
        Decrease the importance score of a memory when called. It will need to be based on whatever context imported into this function.
        """

    def fade_memories(self, decay_rate: float = 0.99):
        """
        Gradually decrease the importance of all unimportant memories over time. This simulates
        the natural process of forgetting in biological memory systems. This function will need to be called on a clock/timer but very slow similar to how humans forget over time.
        """
        for memory in self.memories:
            memory['importance'] *= decay_rate

    def prune_memories(self, importance_threshold: float = 0.1):
        """
        Remove or archive less important memories to maintain system efficiency.
        This prevents memory overload and focuses on retaining significant information.
        Deletes the most unimportant memories based on some importance threshold.
        """
        self.memories = [m for m in self.memories if m['importance'] > importance_threshold]

    def activate_memory(self, memory: Dict[str, Any]):
        """
        Bring a memory into an active state, making it more accessible for processing.
        This simulates the process of memory retrieval and working memory in cognitive systems.
        Could be useful for other functions, especially when it comes to random thoughts.
        """
        self.active_memories.add(memory['content'])
        self.spread_activation(memory)

    def get_random_memory(self) -> Dict[str, Any]:
        return random.choice(self.memories)

    def spread_activation(self, source_memory: Dict[str, Any], depth: int = 3, activation_strength: float = 0.5):
        """
        Propagate activation to related memories. This simulates the associative
        nature of memory recall in biological systems.
        Strengthens connections too. It would call the activate memory too on those memories. But for every new activation down the chain, it weakens the activation to further connections.

        Example:
            [Memory 1] -> activates -> [Memory 2] -> lightly activates [Memory 3] -> barely activates [Memory 4] -> no longer activates

        The choice of what connected memories to activate is random as well. High probability based on similar/most connected relations, low probability on complete randomness.
        """
        if depth == 0 or activation_strength < 0.1:
            return

        for connected_content in source_memory['connections']:
            connected_memory = next((m for m in self.memories if m['content'] == connected_content), None)
            if connected_memory and connected_memory['content'] not in self.active_memories:
                self.activate_memory(connected_memory)
                # Reduce activation strength as it spreads
                self.spread_activation(connected_memory, depth - 1, activation_strength * 0.5)

    def update_active_memories(self, new_memory: Dict[str, Any]):
        """
        Refresh the set of currently active memories based on recent inputs and context.
        This maintains a dynamic working memory state.
        Marks the current activated memories as active for easy retrieval.
        """

    def update_memory_connections(self, new_memory: Dict[str, Any]):
        for memory in self.memories:
            if memory != new_memory:
                similarity = cosine_similarity([new_memory['embedding']], [memory['embedding']])[0][0]
                if similarity > 0.7:  # Threshold for connection
                    new_memory['connections'].append(memory['content'])
                    memory['connections'].append(new_memory['content'])

    def update_emotion_connections(self, memory: Dict[str, Any]):
        memory_emotion = self.generate_emotion_embedding(memory['emotions'])
        for other_memory in self.memories:
            if other_memory != memory:
                other_emotion = self.generate_emotion_embedding(other_memory['emotions'])
                similarity = cosine_similarity([memory_emotion], [other_emotion])[0][0]
                if similarity > 0.8:  # High emotional similarity threshold
                    memory['connections'].append(other_memory['content'])
                    other_memory['connections'].append(memory['content'])

    def generate_emotion_embedding(self, emotions: Dict[str, float]) -> np.ndarray:
        embedding = np.zeros(self.embedding_dim)
        for emotion, intensity in emotions.items():
            if emotion in self.BASE_EMOTIONS:
                embedding += intensity * self.BASE_EMOTIONS[emotion]
        return embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding

    def get_active_memories(self):
        """
        Retrieve the current set of active memories. This represents the current
        focus of attention or working memory state.
        """

    def process_input(self):
        """
        Handle incoming information from a conversation, updating memory states accordingly.
        This method integrates new information into the existing memory structure.
        """

    def simulate_future_situations(self):
        """
        Generate hypothetical future scenarios based on existing active memories + some random memories based on probability of connection as well.
        This simulates the brain's ability to use experiences to predict and prepare for future events.

        This also does both positive and negative (nightmare) scenarios.

        Positive Purpose: Planning ahead, reducing stress, clearing mind on goals
        Negative Purpose: Preparing for worst case scenarios, relative experience learning
        """
        active_memories = list(self.active_memories)
        random_memories = random.sample(self.memories, min(5, len(self.memories)))

        all_memories = active_memories + random_memories
        memory_contents = [m['content'] for m in all_memories]

        prompt = f"""Based on these memories, generate two possible future scenarios:
            1. A positive scenario
            2. A negative (nightmare) scenario

            Memories:
            {', '.join(memory_contents)}

            Provide your response in the following format:
            Positive Scenario: [Your positive scenario here]
            Negative Scenario: [Your negative scenario here]
            """

        scenarios = self.gpt_handler.generate_response(prompt)

        # Add these scenarios as new memories with lower importance
        for scenario in scenarios.split('\n'):
            if scenario.startswith('Positive Scenario:') or scenario.startswith('Negative Scenario:'):
                scenario_type, content = scenario.split(':', 1)
                self.add_memory(content.strip(), {
                    'importance': 0.3,
                    'emotions': {'anticipation': 0.7, 'fear': 0.3} if 'Negative' in scenario_type else {'anticipation': 0.7, 'joy': 0.3}
                })

        return scenarios

    def interrupt_dreaming(self):
        """
        Conclude the dreaming process, finalizing any memory consolidations or reorganizations.
        """
        self.is_dreaming_state = False

    def is_dreaming(self):
        """
        Check if the system is currently in a dreaming state.
        """
        return self.is_dreaming_state

    def search_memories(self, query: str, search_time: float) -> List[Dict[str, Any]]:
        """
        Takes a time input representing how much time it has to search through its memories, the longer the better and more accurate it is. The longer it is will also enable it to find an inimportant memory and strengthen it slowly until its able to recall it or rebuild it.
        """
        start_time = time.time()
        results = []
        query_embedding = self.embedding_model.encode([query])[0]

        while (time.time() - start_time) < search_time:
            if not results:
                # Initial search
                similarities = cosine_similarity([query_embedding], [m['embedding'] for m in self.memories])[0]
                top_indices = np.argsort(similarities)[::-1]
                results = [self.memories[i] for i in top_indices[:5]]
            else:
                # Expand search to connected memories
                connected_memories = []
                for memory in results:
                    connected_memories.extend([m for m in self.memories if m['content'] in memory['connections']])
                connected_memories = list(set(connected_memories) - set(results))
                if connected_memories:
                    similarities = cosine_similarity([query_embedding], [m['embedding'] for m in connected_memories])[0]
                    top_index = np.argmax(similarities)
                    results.append(connected_memories[top_index])

            # Strengthen found memories
            for memory in results:
                self.strengthen_memory(memory)

        return results

    def rebuild_dead_memory(self, memory_location: str) -> Dict[str, Any]:
        """
        Takes in a memory location and tries to stengthen it by rebuilding it/adding memory context to it. It will try to make an educated guess on what used to be there but it will not relate new emotions to it, it will retain the previous emotional connection.
        """
        similar_memories = self.search_memories(memory_location, search_time=5.0)
        context = "\n".join([m['content'] for m in similar_memories])

        prompt = f"""Based on these related memories, reconstruct a possible memory that might have existed:

            Related memories:
            {context}

            Reconstructed memory:"""

        reconstructed_content = self.gpt_handler.generate_response(prompt)

        # Create a new memory with the reconstructed content
        new_memory = self.add_memory(reconstructed_content, {
            'importance': 0.5,  # Medium importance as it's a reconstruction
            'emotions': {}  # No emotions as we can't be sure about the emotional content
        })

        # Link the new memory to the similar memories
        for memory in similar_memories:
            new_memory['connections'].append(memory['content'])
            memory['connections'].append(new_memory['content'])

        return new_memory

    def get_emotional_memories(self, emotion: str, n: int = 5) -> List[Dict[str, Any]]:
        return sorted(self.memories, key=lambda x: x['emotions'].get(emotion, 0), reverse=True)[:n]

    def get_all_memories(self):
        """
        Retrieve all stored memories, potentially for analysis or backup purposes.
        Meant more for debugging purposes
        """

    def get_recent_memories(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch the most recently added or accessed memories.
        """
        return sorted(self.memories, key=lambda x: x['last_accessed'], reverse=True)[:n]

    def get_important_memories(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most important memories based on their importance score
        """
        return sorted(self.memories, key=lambda x: x['importance'], reverse=True)[:n]

    def adjust_memories(self, query: str, new_information: str):
        """
        Modify multiple memories based on external feedback or new information.
        Change/alter connections, reconfigure strengths, review emotional connections.
        So what it does is rewrite it as a new memory, but stores the previous memory one layer deep. So both the old and new occupy similar space but the AI know the new one is valid.
        """
        affected_memories = self.search_memories(query, search_time=3.0)
        for memory in affected_memories:
            old_content = memory['content']
            prompt = f"""Update this memory with new information:
                Old memory: {old_content}
                New information: {new_information}

                Updated memory:"""

            updated_content = self.gpt_handler.generate_response(prompt)

            # Create a new memory with the updated content
            new_memory = self.add_memory(updated_content, {
                'importance': memory['importance'],
                'emotions': memory['emotions'].copy()
            })

            # Link the new memory to the old memory's connections
            new_memory['connections'] = memory['connections'].copy()
            new_memory['connections'].append(old_content)

            # Update the old memory
            memory['content'] = f"Outdated version: {old_content}"
            memory['importance'] *= 0.8  # Reduce importance of outdated memory
            memory['connections'].append(new_memory['content'])

    def generate_memory_summary(self):
        """
        Creates a concise summary of whatever input memory here

        Example:
            Memory: "Alan talked about penguins invading Mars and that it was amazing, he then talked about his other nightmare.. bla bla bla"
            Summarized Memory: "Alan talked to me about some nightmares of his, one of them had a penguin invading Mars"
        """

    def resolve_conflicting_memories(self, memory1: Dict[str, Any], memory2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify memories that contain contradictory information and fix them by choosing one or the other. And also documenting the change so that the AI has some history of the change, similar to the adjust memories function.
        """
        prompt = f"""Resolve the conflict between these two memories and provide a single, consistent memory:
            Memory 1: {memory1['content']}
            Memory 2: {memory2['content']}

            Resolved memory:"""

        resolved_content = self.gpt_handler.generate_response(prompt)

        resolved_memory = self.add_memory(resolved_content, {
            'importance': max(memory1['importance'], memory2['importance']),
            'emotions': self.merge_emotions([memory1['emotions'], memory2['emotions']])
        })

        # Link the resolved memory to the original conflicting memories
        resolved_memory['connections'] = list(set(memory1['connections'] + memory2['connections']))
        resolved_memory['connections'].extend([memory1['content'], memory2['content']])

        # Update the original memories
        for memory in [memory1, memory2]:
            memory['content'] = f"Conflicting version: {memory['content']}"
            memory['importance'] *= 0.7  # Reduce importance of conflicting memories
            memory['connections'].append(resolved_memory['content'])

        return resolved_memory

    def save_memory_state(self, filepath: str):
        """
        Persist the current state of the memory system to storage for later retrieval. Used for when the program restarts.
        """
        state = {
            'memories': self.memories,
            'active_memories': list(self.active_memories),
            'is_dreaming_state': self.is_dreaming_state,
            'BASE_EMOTIONS': self.BASE_EMOTIONS
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_memory_state(self, filepath: str):
        """
        Restore a previously saved memory state, reconstructing the memory system. Used for when the program restarts.
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.memories = state['memories']
        self.active_memories = set(state['active_memories'])
        self.is_dreaming_state = state['is_dreaming_state']
        self.BASE_EMOTIONS = state['BASE_EMOTIONS']

        # Recreate embeddings for all memories
        for memory in self.memories:
            memory['embedding'] = self.embedding_model.encode([memory['content']])[0]
