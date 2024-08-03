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
        """
        Initiates the dreaming process when triggered.
        This function coordinates the entire dreaming cycle.
        """
        self.deep_sleep = True
        self.dream_start_time = time.time()
        self.logger.info("Initiating dreaming process")

        self._filter_memories()
        self._merge_short_term_to_long_term()
        self._merge_similar_memories()
        self._simulate_future_situations()

        self._end_dreaming()

    def _filter_memories(self):
        """
        Step 1 of dreaming process: Filter out unimportant memories from short-term memory.
        Randomly retains a few unimportant memories to simulate brain randomness.
        """
        self.logger.info("Filtering memories")
        short_term_memories = self.memory.get_short_term_memories()
        filtered_memories = []

        for memory in short_term_memories:
            if memory.importance > 0.5 or random.random() < 0.1:  # Keep important or random memories
                filtered_memories.append(memory)

        self.memory.set_short_term_memories(filtered_memories)

    def _merge_short_term_to_long_term(self):
        """
        Step 2 of dreaming process: Merge short-term memories into long-term memory.
        Gives a slight importance boost to these memories as they are newer.
        """
        self.logger.info("Merging short-term memories to long-term")
        short_term_memories = self.memory.get_short_term_memories()

        for memory in short_term_memories:
            memory.importance *= 1.1  # Slight importance boost
            self.memory.add_to_long_term_memory(memory)

        self.memory.clear_short_term_memory()

    def _merge_similar_memories(self):
        """
        Step 3 of dreaming process: Merge similar memories in long-term memory.
        Starts with the furthest and least important memories.
        """
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
        """
        Step 4 of dreaming process: Simulate future situations based on previous context and memories.
        Uses a random value to determine if the dream scenario is positive (0) or negative (1).
        """
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
        """Ends the dreaming process and sets deep_sleep to False."""
        self.deep_sleep = False
        self.logger.info("Ending dreaming process")

    def is_dreaming(self):
        """
        Checks if the AI is currently in a dreaming state.
        Also handles the transition out of dreaming based on time limit.
        """
        if self.deep_sleep:
            current_time = time.time()
            if current_time - self.dream_start_time > self.dream_duration:
                self._end_dreaming()
            return self.deep_sleep
        return False

    def interrupt_dreaming(self):
        """
        Allows for interruption of the dreaming process.
        This can be called by the orchestrator if needed.
        """
        if self.deep_sleep:
            self.logger.info("Interrupting dreaming process")
            self._end_dreaming()