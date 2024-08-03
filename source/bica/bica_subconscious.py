import random
from typing import List, Dict
from bica_memory import BicaMemory
from bica_context import BicaContext
from bica_reasoning import BicaReasoning
from bica_utilities import BicaUtilities
from sentence_transformers import SentenceTransformer


class BicaSubconscious:
    def __init__(self, memory: BicaMemory, context: BicaContext, reasoning: BicaReasoning):
        self.memory = memory
        self.context = context
        self.reasoning = reasoning
        self.utilities = BicaUtilities()
        self.visible_thoughts = []
        self.action_keywords = [
            "help", "run", "think", "analyze", "question", "observe", "react",
            "consider", "explore", "avoid", "engage", "clarify", "empathize",
            "challenge", "support", "investigate", "create", "solve", "adapt"
        ]
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_subconscious_thoughts(self):
        # Get current context and recent conversation
        current_context = self.context.get_context()
        recent_conversation = self.context.get_recent_conversation()

        # Search for relevant actions in memory
        relevant_actions = self._search_relevant_actions(current_context, recent_conversation)

        # Generate random actions
        random_actions = self._generate_random_actions()

        # Combine relevant and random actions, ensuring a mix
        all_actions = self._combine_actions(relevant_actions, random_actions)

        # Generate scenarios
        scenarios = self._generate_scenarios(all_actions, current_context, recent_conversation)

        # Evaluate probabilities of scenarios
        evaluated_scenarios = self._evaluate_scenario_probabilities(scenarios)

        # Filter and save visible thoughts
        self._filter_and_save_thoughts(evaluated_scenarios)

    def _search_relevant_actions(self, context: str, conversation: List[str]) -> List[str]:
        combined_text = context + " " + " ".join(conversation)
        query_embedding = self.sentence_model.encode(combined_text)

        relevant_actions = []
        memories = self.memory.get_all_memories()

        for memory in memories:
            memory_embedding = self.sentence_model.encode(memory.content)
            similarity = self.utilities.calculate_similarity(query_embedding, memory_embedding)

            if similarity > 0.7:  # Threshold for relevance
                actions = self._extract_actions_from_text(memory.content)
                relevant_actions.extend(actions)

        return list(set(relevant_actions))  # Remove duplicates

    def _extract_actions_from_text(self, text: str) -> List[str]:
        words = text.split()
        actions = []
        for i in range(len(words) - 1):
            if words[i].lower() in self.action_keywords:
                actions.append(f"{words[i]} {words[i + 1]}")
        return actions

    def _generate_random_actions(self) -> List[str]:
        return [f"{random.choice(self.action_keywords)} {random.choice(self.utilities.extract_keywords(self.memory.get_random_memory(), 10))}"
                for _ in range(10)]  # Generate 10 random actions

    def _combine_actions(self, relevant_actions: List[str], random_actions: List[str]) -> List[str]:
        # Ensure a mix of relevant and random actions
        combined = relevant_actions[:10] + random_actions  # Take up to 10 relevant actions
        random.shuffle(combined)
        return combined[:15]  # Return a total of 15 mixed actions

    def _generate_scenarios(self, actions: List[str], context: str, conversation: List[str]) -> List[str]:
        scenarios = []
        for _ in range(10):  # Generate 10 scenarios
            selected_actions = random.sample(actions, 3)  # Select 3 random actions for each scenario
            scenario = f"Context: {context}\nConversation: {' '.join(conversation[-3:])}\nActions: {', '.join(selected_actions)}"
            scenarios.append(scenario)
        return scenarios

    def _evaluate_scenario_probabilities(self, scenarios: List[str]) -> List[Dict[str, any]]:
        evaluated_scenarios = []
        for scenario in scenarios:
            probability = self.reasoning.evaluate_hypothesis(scenario, [])  # No evidence provided
            evaluated_scenarios.append({
                "scenario": scenario,
                "probability": float(probability.get('Posterior probability', [0])[0])
            })
        return evaluated_scenarios

    def _filter_and_save_thoughts(self, evaluated_scenarios: List[Dict[str, any]]):
        sorted_scenarios = sorted(evaluated_scenarios, key=lambda x: x['probability'], reverse=True)
        top_half = sorted_scenarios[:len(sorted_scenarios) // 2]
        self.visible_thoughts = [scenario['scenario'] for scenario in top_half]

    def get_visible_thoughts(self) -> List[str]:
        return self.visible_thoughts