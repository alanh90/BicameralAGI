import numpy as np
from typing import List, Dict, Any
from gpt_handler import GPTHandler
from bica_memory import BicaMemory
from bica_context import BicaContext
from bica_utilities import BicaUtilities
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import time


class BicaCognition:
    """
        BicaCognition class represents the cognitive processes of the AI system.
        It integrates conscious thought generation, subconscious processing,
        reasoning, and various analytical capabilities.
    """

    def __init__(self, memory: BicaMemory, context: BicaContext, external_kb_api=None):
        self.memory = memory
        self.context = context
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.external_kb_api = external_kb_api
        self.tfidf_vectorizer = TfidfVectorizer()
        self.current_thoughts = []
        self.subconscious_thoughts = []
        self.visible_thoughts = []
        self.thinking_mode = "analytical"  # Default mode
        self.action_keywords = [
            "help", "run", "think", "analyze", "question", "observe", "react",
            "consider", "explore", "avoid", "engage", "clarify", "empathize",
            "challenge", "support", "investigate", "create", "solve", "adapt"
        ]
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_thoughts(self, context):
        temperature = round(random.uniform(0.1, 0.4), 2)
        prompt = f"""Generate a few short thoughts about the following context. Each thought should be a complete sentence or idea. Consider these aspects if any of them are unknown:
        - Purpose of the conversation
        - Who you might be speaking with
        - Possible environment of the interaction
        - Potential emotional states involved
        - Relevant background knowledge
        - Possible implications or consequences
        - Uncertainties or assumptions
        Current thinking mode: {self.thinking_mode}
        Context: {context}
        Thoughts:
        """
        stream = self.gpt_handler.generate_response(prompt, temperature=temperature, stream=True)
        thoughts = ""
        for chunk in stream:
            thoughts += chunk
        print(f"Temperature used: {temperature}")
        self.current_thoughts = thoughts.strip().split('\n')
        return self.current_thoughts

    def analyze(self, information, thoughts):
        # We'll adapt this from bica_reasoning.py
        pass

    def generate_subconscious_thoughts(self):
        current_context = self.context.get_context()
        recent_conversation = self.context.get_recent_conversation()

        relevant_actions = self._search_relevant_actions(current_context, recent_conversation)
        random_actions = self._generate_random_actions()
        all_actions = self._combine_actions(relevant_actions, random_actions)
        scenarios = self._generate_scenarios(all_actions, current_context, recent_conversation)
        evaluated_scenarios = self._evaluate_scenario_probabilities(scenarios)
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
        combined = relevant_actions[:10] + random_actions
        random.shuffle(combined)
        return combined[:15]

    def _generate_scenarios(self, actions: List[str], context: str, conversation: List[str]) -> List[str]:
        scenarios = []
        for _ in range(10):
            selected_actions = random.sample(actions, 3)
            scenario = f"Context: {context}\nConversation: {' '.join(conversation[-3:])}\nActions: {', '.join(selected_actions)}"
            scenarios.append(scenario)
        return scenarios

    def _evaluate_scenario_probabilities(self, scenarios: List[str]) -> List[Dict[str, any]]:
        evaluated_scenarios = []
        for scenario in scenarios:
            evaluation = self.evaluate_hypothesis(scenario, [])
            evaluated_scenarios.append({
                "scenario": scenario,
                "probability": float(evaluation.get('Posterior probability', [0])[0])
            })
        return evaluated_scenarios

    def _filter_and_save_thoughts(self, evaluated_scenarios: List[Dict[str, any]]):
        sorted_scenarios = sorted(evaluated_scenarios, key=lambda x: x['probability'], reverse=True)
        top_half = sorted_scenarios[:len(sorted_scenarios) // 2]
        self.visible_thoughts = [scenario['scenario'] for scenario in top_half]

    def analyze(self, information: str, thoughts: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Analyze the following information and thoughts:
        Information: {information}
        Thoughts: {thoughts}

        Provide a structured analysis including:
        1. Key points
        2. Potential implications
        3. Connections to existing knowledge
        4. Areas that require further investigation
        5. Confidence level (Low, Medium, High) for each point
        6. Potential biases in the analysis
        """
        analysis = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(analysis)

    def _structure_analysis(self, analysis: str) -> Dict[str, Any]:
        lines = analysis.split('\n')
        structured = {}
        current_key = ''
        for line in lines:
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                current_key = line[3:].strip()
                structured[current_key] = []
            elif current_key and line.strip():
                structured[current_key].append(line.strip())
        return structured

    def make_connections(self, concept: str, related_memories: List[str], new_thoughts: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Concept: {concept}
        Related Memories: {related_memories}
        New Thoughts: {new_thoughts}

        Based on the above, identify:
        1. New connections or relationships
        2. Potential contradictions or conflicts
        3. Areas for further exploration
        4. Possible generalizations or principles
        5. Analogies to other domains
        """
        connections = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(connections)

    def bayesian_update(self, prior: float, likelihood: float, evidence: float) -> float:
        return (likelihood * prior) / ((likelihood * prior) + (1 - likelihood) * (1 - prior))

    def evaluate_hypothesis(self, hypothesis: str, evidence: List[str]) -> Dict[str, Any]:
        prompt = f"""
        Evaluate the following hypothesis based on the provided evidence:
        Hypothesis: {hypothesis}
        Evidence: {evidence}

        Provide an evaluation including:
        1. Strength of the hypothesis (0-1 scale)
        2. Likelihood of evidence given hypothesis (0-1 scale)
        3. Prior probability of hypothesis (0-1 scale)
        4. Supporting evidence
        5. Contradicting evidence
        6. Confidence level (Low, Medium, High)
        7. Suggestions for further testing
        """
        evaluation = next(self.gpt_handler.generate_response(prompt))
        structured_eval = self._structure_analysis(evaluation)

        prior = float(structured_eval['Prior probability of hypothesis'][0])
        likelihood = float(structured_eval['Likelihood of evidence given hypothesis'][0])
        evidence_prob = 0.5  # Assuming neutral evidence probability

        posterior = self.bayesian_update(prior, likelihood, evidence_prob)
        structured_eval['Posterior probability'] = [str(posterior)]

        return structured_eval

    def generate_action_plan(self, goal: str, constraints: List[str]) -> List[str]:
        prompt = f"""
        Generate an action plan for the following goal, considering the given constraints:
        Goal: {goal}
        Constraints: {constraints}

        Provide a step-by-step action plan, including:
        1. Main steps
        2. Potential obstacles for each step
        3. Contingency plans
        4. Estimated time and resources needed
        5. Success criteria for each step
        """
        action_plan = next(self.gpt_handler.generate_response(prompt))
        return action_plan.split('\n')

    def solve_problem(self, problem: str, context: str) -> Dict[str, Any]:
        prompt = f"""
        Solve the following problem given the context:
        Problem: {problem}
        Context: {context}

        Provide a solution including:
        1. Problem breakdown
        2. Potential solutions (at least 3)
        3. Pros and cons of each solution
        4. Evaluation criteria
        5. Recommended approach with justification
        6. Potential challenges and mitigation strategies
        7. Ethical considerations
        """
        solution = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(solution)

    def ethical_evaluation(self, action: str, context: str) -> Dict[str, Any]:
        prompt = f"""
        Evaluate the ethical implications of the following action in the given context:
        Action: {action}
        Context: {context}

        Provide an evaluation including:
        1. Potential positive outcomes
        2. Potential negative consequences
        3. Stakeholders affected
        4. Relevant ethical principles or frameworks
        5. Alternative actions to consider
        6. Overall ethical assessment (Ethical, Questionable, Unethical)
        """
        evaluation = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(evaluation)

    def meta_reasoning(self, reasoning_process: str) -> Dict[str, Any]:
        prompt = f"""
        Analyze and evaluate the following reasoning process:
        {reasoning_process}

        Provide an analysis including:
        1. Strengths of the reasoning
        2. Weaknesses or potential flaws
        3. Underlying assumptions
        4. Potential biases
        5. Suggestions for improvement
        """
        analysis = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(analysis)

    def fact_check(self, statement: str) -> Dict[str, Any]:
        if self.external_kb_api:
            api_response = self.external_kb_api.check_fact(statement)
            return api_response
        else:
            return {"error": "External knowledge base not available"}

    def find_analogy(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        prompt = f"""
        Find analogies between the following domains:
        Source domain: {source_domain}
        Target domain: {target_domain}

        Provide an analysis including:
        1. Key similarities between domains
        2. Potential mappings of concepts
        3. Limitations of the analogy
        4. Insights that can be drawn from the analogy
        """
        analogies = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(analogies)

    def emotional_reasoning(self, situation: str, emotional_context: Dict[str, float]) -> Dict[str, Any]:
        emotion_str = ", ".join([f"{k}: {v}" for k, v in emotional_context.items()])
        prompt = f"""
        Analyze the following situation considering the given emotional context:
        Situation: {situation}
        Emotional context: {emotion_str}

        Provide an analysis including:
        1. How emotions might influence perception of the situation
        2. Potential emotional responses to the situation
        3. How emotions might affect decision-making in this context
        4. Strategies for emotional regulation if needed
        """
        analysis = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(analysis)

    def adaptive_reasoning(self, problem: str, context: str, previous_approach: str) -> Dict[str, Any]:
        prompt = f"""
        Adapt the reasoning approach for the following problem:
        Problem: {problem}
        Context: {context}
        Previous approach: {previous_approach}

        Provide an adaptive reasoning strategy including:
        1. Evaluation of the previous approach
        2. Suggested modifications to the approach
        3. New techniques or perspectives to consider
        4. Potential challenges with the new approach
        5. Success criteria for the adapted approach
        """
        strategy = next(self.gpt_handler.generate_response(prompt))
        return self._structure_analysis(strategy)
    def get_visible_thoughts(self) -> List[str]:
        return self.visible_thoughts

    def set_thinking_mode(self, mode):
        valid_modes = ["analytical", "creative", "reflective", "predictive"]
        if mode in valid_modes:
            self.thinking_mode = mode
        else:
            raise ValueError(f"Invalid thinking mode. Choose from {valid_modes}")

    def _apply_thinking_mode(self, thoughts: List[str]) -> List[str]:
        prompt = f"""
        Given the following thoughts and the current thinking mode ({self.thinking_mode}),
        adjust or expand on these thoughts to better reflect the thinking mode:

        Thoughts: {thoughts}

        Provide the adjusted thoughts:
        """
        response = next(self.gpt_handler.generate_response(prompt))
        return response.strip().split('\n')

    def get_all_thoughts(self) -> List[str]:
        return self.current_thoughts + self.visible_thoughts

    def process_cognitive_cycle(self, context):
        # Generate conscious thoughts
        self.generate_thoughts(context)

        # Generate subconscious thoughts
        self.generate_subconscious_thoughts()

        # Analyze the current situation
        analysis = self.analyze(context, self.current_thoughts + self.visible_thoughts)

        # Make connections
        connections = self.make_connections(context, self.memory.get_recent_memories(), self.current_thoughts)

        # Evaluate hypotheses
        hypotheses = [thought for thought in self.current_thoughts if '?' in thought]
        evaluated_hypotheses = [self.evaluate_hypothesis(h, []) for h in hypotheses]

        # Perform meta-reasoning
        meta_analysis = self.meta_reasoning(str(analysis))

        return {
            'analysis': analysis,
            'connections': connections,
            'evaluated_hypotheses': evaluated_hypotheses,
            'meta_analysis': meta_analysis
        }


