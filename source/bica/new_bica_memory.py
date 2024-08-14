class BicaMemory:
    """
    A comprehensive memory system for AI, simulating human-like memory processes.
    """

    def __init__(self):
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

    def initiate_dreaming(self):
        """
        Perform memory consolidation, pruning, and other maintenance tasks.
        This method simulates the memory processes that occur during sleep in biological systems.
        """

    def add_memory(self):
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

    def recall_memory(self):
        """
        Retrieve memories related to a given query. This should involve searching
        through the memory stores and returning the most relevant memories.
        This could also relate to recalling emotional memories. If someone asks what is your most happy memory.

        When a memory is recalled, it should strengthen it so that it stays longer in memory and it becomes more static and less prone to summarizing or alteration. Strengthening it also strengthens related connections slightly.
        """

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

    def increase_memory_importance(self):
        """
        Increase the importance score of a memory when called. It will need to be based on whatever context imported into this function.
        """

    def decrease_memory_importance(self):
        """
        Decrease the importance score of a memory when called. It will need to be based on whatever context imported into this function.
        """

    def fade_memories(self):
        """
        Gradually decrease the importance of all unimportant memories over time. This simulates
        the natural process of forgetting in biological memory systems. This function will need to be called on a clock/timer but very slow similar to how humans forget over time.
        """

    def prune_memories(self):
        """
        Remove or archive less important memories to maintain system efficiency.
        This prevents memory overload and focuses on retaining significant information.
        Deletes the most unimportant memories based on some importance threshold.
        """

    def activate_memory(self):
        """
        Bring a memory into an active state, making it more accessible for processing.
        This simulates the process of memory retrieval and working memory in cognitive systems.
        Could be useful for other functions, especially when it comes to random thoughts.
        """

    def spread_activation(self):
        """
        Propagate activation to related memories. This simulates the associative
        nature of memory recall in biological systems.
        Strengthens connections too. It would call the activate memory too on those memories. But for every new activation down the chain, it weakens the activation to further connections.

        Example:
            [Memory 1] -> activates -> [Memory 2] -> lightly activates [Memory 3] -> barely activates [Memory 4] -> no longer activates

        The choice of what connected memories to activate is random as well. High probability based on similar/most connected relations, low probability on complete randomness.
        """

    def update_active_memories(self):
        """
        Refresh the set of currently active memories based on recent inputs and context.
        This maintains a dynamic working memory state.
        Marks the current activated memories as active for easy retrieval.
        """

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

    def interrupt_dreaming(self):
        """
        Conclude the dreaming process, finalizing any memory consolidations or reorganizations.
        """

    def is_dreaming(self):
        """
        Check if the system is currently in a dreaming state.
        """

    def search_memories(self):
        """
        Takes a time input representing how much time it has to search through its memories, the longer the better and more accurate it is. The longer it is will also enable it to find an inimportant memory and strengthen it slowly until its able to recall it or rebuild it.
        """

    def rebuild_dead_memory(self):
        """
        Takes in a memory location and tries to stengthen it by rebuilding it/adding memory context to it. It will try to make an educated guess on what used to be there but it will not relate new emotions to it, it will retain the previous emotional connection.
        """

    def get_all_memories(self):
        """
        Retrieve all stored memories, potentially for analysis or backup purposes.
        Meant more for debugging purposes
        """

    def get_recent_memories(self):
        """
        Fetch the most recently added or accessed memories.
        """

    def get_important_memories(self):
        """
        Retrieves the most important memories based on their importance score
        """

    def adjust_memories(self):
        """
        Modify multiple memories based on external feedback or new information.
        Change/alter connections, reconfigure strengths, review emotional connections.
        So what it does is rewrite it as a new memory, but stores the previous memory one layer deep. So both the old and new occupy similar space but the AI know the new one is valid.
        """

    def generate_memory_summary(self):
        """
        Creates a concise summary of whatever input memory here

        Example:
            Memory: "Alan talked about penguins invading Mars and that it was amazing, he then talked about his other nightmare.. bla bla bla"
            Summarized Memory: "Alan talked to me about some nightmares of his, one of them had a penguin invading Mars"
        """

    def resolve_conflicting_memories(self):
        """
        Identify memories that contain contradictory information and fix them by choosing one or the other. And also documenting the change so that the AI has some history of the change, similar to the adjust memories function.
        """

    def save_memory_state(self):
        """
        Persist the current state of the memory system to storage for later retrieval. Used for when the program restarts.
        """

    def load_memory_state(self):
        """
        Restore a previously saved memory state, reconstructing the memory system. Used for when the program restarts.
        """