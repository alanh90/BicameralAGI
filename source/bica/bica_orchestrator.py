"""
BicameralAGI Orchestrator Module
================================

Overview:
---------
This module serves as the central coordinator for the BicameralAGI system. It is responsible for initializing and managing
all core components, orchestrating the flow of information between them, and coordinating the overall processing of user
inputs and system responses. As the heart of the BicameralAGI architecture, the BicaOrchestrator integrates context, memory,
affect, cognition, and action execution to simulate a cohesive and dynamic AI system.

Currently, this module provides a simplified version of the orchestrator’s functionality, with basic components like context
management and memory retrieval in place. While much of the core processing pipeline has been outlined (emotions, cognitive
processing, safety checks, and more), several critical parts are under development and will be integrated soon.

Key Features (Current and Upcoming):
------------------------------------
1. **Context Management**: Tracks user interactions and maintains a weighted context to inform responses.
2. **Memory Management**: Retrieves recent memories to enrich responses with historical knowledge of past interactions.
3. **Component Coordination**: Facilitates the interaction between core modules like context, memory, and the action executor.
4. **Action Execution**: Executes actions based on processed input and compiled system states, generating responses to users.
5. **Dynamic Prompt Compilation**: Constructs a prompt based on system state (context, memory, etc.) to generate user-facing
   responses.
6. **Planned Features** (under development):
   - **Affect Processing**: Simulation of emotions and personality traits based on input and past experiences.
   - **Cognitive Processing**: Generate thoughts, conduct deep analysis, and evaluate ethics of decisions.
   - **Safety Checks**: Enforce safety and ethical guidelines on thoughts and actions to ensure responsible AI behavior.
   - **Destiny Prediction**: Simulate and project future AI behavior and goals based on emotions, memories, and experiences.

Usage Example:
--------------
The orchestrator can be used to process user inputs and generate intelligent responses:

    orchestrator = BicaOrchestrator()
    response = orchestrator.process_input("Tell me about the future of AI.")
    print(response)

This will initiate a flow that updates the context, processes recent memories, compiles a prompt, and generates a response.

Author:
-------
Alan Hourmand
Date: 9/23/2024
"""

from bica_action_executor import BicaActionExecutor
from bica_logging import BicaLogging
from bica_context import BicaContext
from bica_memory import BicaMemory
from gpt_handler import GPTHandler

#Temporarily unavailable
from bica_affect import BicaAffect
from bica_cognition import BicaCognition
from bica_safety import BicaSafety
from bica_destiny import BicaDestiny
from bica_utilities import *


class BicaOrchestrator:
    def __init__(self):
        self.logger = BicaLogging("BicaOrchestrator")
        self.action_executor = BicaActionExecutor()
        self.gpt_handler = GPTHandler()

        self.context = BicaContext()
        self.memory = BicaMemory()

        '''
        Will need to make sure these are compatible before implementation
        ----------------------------
        self.affect = BicaAffect("BicaAI")
        self.cognition = BicaCognition(self.memory, self.context)
        self.safety = BicaSafety()
        self.destiny = BicaDestiny("BicaAI")
        '''

    def process_input(self, user_input: str) -> str:
        try:
            print(f"Received user input: {user_input}")
            context_data = {
                "user_input": user_input,
                "system_prompt": "You are BicaAI, a helpful and intelligent assistant."
            }

            # 1. Update context
            print("Updating context...")
            self.context.update_context(user_input)
            context_data["weighted_context"] = self.context.get_weighted_context()
            print(f"Weighted context: {context_data['weighted_context']}")

            # 2. Process memories
            print("Retrieving recent memories...")
            context_data["recent_memories"] = self.memory.get_recent_memories(5)
            print(f"Recent memories: {context_data['recent_memories']}")

            # The following block is commented out for future use:
            # context_data["relevant_memories"] = self.memory.recall_memory(user_input)
            # compiled_data["emotional_memories"] = self.memory.get_emotional_memories("joy", 0.7)
            # compiled_data["important_memories"] = self.memory.get_important_memories(0.8)
            #
            # # 3. Process emotions and personality
            # compiled_data["emotions"] = self.affect.get_top_emotions()
            # compiled_data["personality_summary"] = self.affect.get_personality_summary()
            # self.affect.update_personality_from_emotion()
            # self.affect.alter_traits_from_experience(user_input, intensity=0.5)
            # compiled_data["style_guide"] = self.affect.cog_model["styleGuideValues"]
            #
            # # 4. Determine conversation situation
            # compiled_data["situation"] = self.determine_situation(user_input, compiled_data["weighted_context"])
            # compiled_data["situational_response"] = self.affect.cog_model["situationalConversations"]
            # .get(compiled_data["situation"], {}).get("examples", [""])[0]
            #
            # # 5. Cognitive processing
            # compiled_data["conscious_thoughts"] = self.cognition.generate_thoughts(
            # str(compiled_data["weighted_context"]), [user_input])
            # compiled_data["subconscious_thoughts"] = self.cognition.generate_subconscious_thoughts(
            # str(compiled_data["weighted_context"]))
            # compiled_data["deep_analysis"] = self.cognition.deep_thought(user_input)
            #
            # # 6. Ethical evaluation
            # compiled_data["ethical_evaluation"] = self.cognition._evaluate_thoughts(
            # compiled_data["conscious_thoughts"] + compiled_data["subconscious_thoughts"],
            # str(compiled_data["weighted_context"]))
            #
            # # 7. Safety check
            # safety_threshold = self.determine_safety_threshold(compiled_data["weighted_context"],
            # compiled_data["emotions"])
            # compiled_data["safe_thoughts"] = self.safety.safety_filter(
            # str(compiled_data["conscious_thoughts"] + compiled_data["subconscious_thoughts"]),
            # "thought", safety_threshold)
            #
            # # 8. Update destiny
            # self.destiny.generate_destiny(
            #     personality=compiled_data["personality_summary"],
            #     recent_events=str(compiled_data["recent_memories"]),
            #     emotions=str(compiled_data["emotions"])
            # )
            # compiled_data["current_destinies"] = self.destiny.get_destinies()

            # 9. Compile prompt
            print("Compiling prompt...")
            compiled_data = self.compile_prompt(context_data)
            print(f"Compiled prompt:\n{compiled_data}")

            # 10. Execute action (respond)
            print("Executing action 'respond' with compiled data...")
            response = self.action_executor.execute_action("respond", {"compiled_data": compiled_data})
            print(f"Generated response: {response}")

            # The following block is commented out for future use:
            # safe_response = self.safety.safety_filter(action_result["response"], "content", safety_threshold)
            # self.memory.save_memory(f"User: {user_input}\nAI: {safe_response}", compiled_data["emotions"],
            # importance=0.7)
            # self.memory.simulate_future_situations()
            #
            # self.logger.info(f"Generated response: {safe_response}")
            # return safe_response

            # Save memory of the interaction
            print("Saving memory...")
            self.memory.save_memory(f"User: {user_input}\nAI: {response}", {}, importance=0.5)
            print("Memory saved successfully.")

            return response

        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            print(f"Error: {str(e)}")
            return "I apologize, but I encountered an error. Could you please try again?"

    def compile_prompt(self, compiled_data: dict) -> str:
        print("Compiling the prompt...")
        prompt_parts = []
        for key, value in compiled_data.items():
            if value:  # Only include non-empty values
                if isinstance(value, dict):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for sub_key, sub_value in value.items():
                        prompt_parts.append(f"  {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    prompt_parts.append(f"{key.replace('_', ' ').title()}:")
                    for item in value:
                        prompt_parts.append(f"  - {item}")
                else:
                    prompt_parts.append(f"{key.replace('_', ' ').title()}: {value}")

        prompt = "\n".join(prompt_parts)
        prompt += "\n\nBased on the above context information, generate a final response to the user."
        print(f"Compiled prompt:\n{prompt}")
        return prompt
