import numpy as np
from typing import List, Dict, Any
from gpt_handler import GPTHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BicaReasoning:
    def __init__(self, external_kb_api=None):
        self.gpt_handler = GPTHandler()
        self.external_kb_api = external_kb_api
        self.tfidf_vectorizer = TfidfVectorizer()

    def analyze(self, information: str, thoughts: List[str]) -> Dict[str, Any]:
        """
        Analyze new information and thoughts to draw conclusions.
        """
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
        """
        Convert the GPT-generated analysis into a structured dictionary.
        """
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
        """
        Draw new connections between a concept, related memories, and new thoughts.
        """
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
        """
        Perform a Bayesian update to revise a probability estimate.
        """
        return (likelihood * prior) / ((likelihood * prior) + (1 - likelihood) * (1 - prior))

    def evaluate_hypothesis(self, hypothesis: str, evidence: List[str]) -> Dict[str, Any]:
        """
        Evaluate a hypothesis based on given evidence using Bayesian reasoning.
        """
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
        """
        Generate an action plan to achieve a goal given certain constraints.
        """
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
        """
        Apply problem-solving techniques to address a given problem.
        """
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
        """
        Evaluate the ethical implications of an action in a given context.
        """
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
        """
        Analyze and evaluate a reasoning process.
        """
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
        """
        Check the factual accuracy of a statement using an external knowledge base.
        """
        if self.external_kb_api:
            # This is a placeholder. In a real implementation, you would call the API
            # and process its response.
            api_response = self.external_kb_api.check_fact(statement)
            return api_response
        else:
            return {"error": "External knowledge base not available"}

    def find_analogy(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """
        Find analogies between two different domains.
        """
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
        """
        Incorporate emotional context into reasoning about a situation.
        """
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
        """
        Adapt reasoning approach based on the problem, context, and previous approaches.
        """
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