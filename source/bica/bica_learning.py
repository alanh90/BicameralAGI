from bica_memory import BicaMemory
from bica_thoughts import BicaThoughts
from bica_reasoning import BicaReasoning
from gpt_handler import GPTHandler


class BicaLearning:
    def __init__(self, memory: BicaMemory, thoughts: BicaThoughts, reasoning: BicaReasoning):
        self.memory = memory
        self.thoughts = thoughts
        self.reasoning = reasoning
        self.gpt_handler = GPTHandler()

    def process_new_information(self, information: str):
        """
        Process new information and integrate it into the system.
        """
        # Store the new information in memory
        self.memory.add_memory(information)

        # Generate thoughts about the new information
        thoughts = self.thoughts.generate_thoughts(information)

        # Use reasoning to analyze the information and thoughts
        analysis = self.reasoning.analyze(information, thoughts)

        # Update the knowledge base based on the analysis
        self.update_knowledge_base(analysis)

    def update_knowledge_base(self, analysis: dict):
        """
        Update the knowledge base with new insights from the analysis.
        """
        # This could involve updating long-term memory, adjusting weights of certain concepts, etc.
        pass

    def reinforce_learning(self, concept: str):
        """
        Reinforce learning of a specific concept through repetition and elaboration.
        """
        # Retrieve related memories
        related_memories = self.memory.recall_memory(concept)

        # Generate new thoughts to elaborate on the concept
        new_thoughts = self.thoughts.generate_thoughts(concept)

        # Use reasoning to draw new connections
        new_connections = self.reasoning.make_connections(concept, related_memories, new_thoughts)

        # Update the knowledge base with these new connections
        self.update_knowledge_base(new_connections)

    def integrate_feedback(self, feedback: str):
        """
        Integrate external feedback to adjust learning and understanding.
        """
        # Analyze the feedback
        feedback_analysis = self.reasoning.analyze_feedback(feedback)

        # Adjust relevant memories and concepts based on the feedback
        self.memory.adjust_memories(feedback_analysis)

        # Generate new thoughts considering the feedback
        new_thoughts = self.thoughts.generate_thoughts(feedback)

        # Update the knowledge base
        self.update_knowledge_base({"feedback": feedback_analysis, "new_thoughts": new_thoughts})

    def generate_learning_summary(self):
        """
        Generate a summary of recent learning and insights.
        """
        recent_memories = self.memory.get_recent_memories()
        recent_thoughts = self.thoughts.get_recent_thoughts()

        prompt = f"Generate a summary of recent learning based on these memories and thoughts:\nMemories: {recent_memories}\nThoughts: {recent_thoughts}"
        summary = next(self.gpt_handler.generate_response(prompt))

        return summary