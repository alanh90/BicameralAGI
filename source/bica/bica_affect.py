import json
import random
import time
import os
from gpt_handler import GPTHandler
from bica_utilities import BicaUtilities


class BicaAffect:
    def __init__(self, personality_file='persona_cog_template.json', memory_file='bica_memory.json'):
        self.gpt_handler = GPTHandler()
        self.utilities = BicaUtilities()
        self.personality = self.load_personality(personality_file)
        self.memory = self.load_memory(memory_file)
        self.current_emotions = {}
        self.emotional_memory = []
        self.emotion_falloff_rate = 0.05
        self.last_update = time.time()
        self.persona_cog_folder = os.path.join('data', 'persona_cog_models')
        self.template_file = os.path.join('data', 'template', 'persona_cog_template.json')

    def load_personality(self, personality_file):
        if os.path.exists(personality_file):
            with open(personality_file, 'r') as file:
                return json.load(file)
        return {}

    def load_memory(self, memory_file):
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as file:
                return json.load(file)
        return {}

    def trigger_emotion(self, emotion, intensity):
        if emotion in self.current_emotions:
            self.current_emotions[emotion] += intensity
        else:
            self.current_emotions[emotion] = intensity
        self.update_emotional_memory(emotion, intensity)

    def update_emotional_memory(self, emotion, intensity):
        self.emotional_memory.append({
            'emotion': emotion,
            'intensity': intensity,
            'timestamp': time.time()
        })

    def balance_emotions(self):
        for emotion in self.current_emotions:
            self.current_emotions[emotion] -= self.emotion_falloff_rate
            if self.current_emotions[emotion] < 0:
                self.current_emotions[emotion] = 0

    def falloff_emotions(self):
        current_time = time.time()
        time_diff = current_time - self.last_update
        falloff_amount = time_diff * self.emotion_falloff_rate

        for emotion in list(self.current_emotions.keys()):
            self.current_emotions[emotion] -= falloff_amount
            if self.current_emotions[emotion] <= 0:
                del self.current_emotions[emotion]

        self.last_update = current_time

    def get_dominant_emotion(self):
        if not self.current_emotions:
            return None
        return max(self.current_emotions, key=self.current_emotions.get)

    def update(self):
        self.falloff_emotions()
        self.balance_emotions()

    def get_emotional_state(self):
        self.update()
        return self.current_emotions

    def save_memory(self, memory_file='bica_memory.json'):
        with open(memory_file, 'w') as file:
            json.dump(self.memory, file)

    def create_persona_cog(self, character_name, description):
        cog_model = self.generate_cog_model(character_name, description)
        self.save_cog_model(character_name, cog_model)

    def generate_cog_model(self, character_name, description):
        with open(self.template_file, 'r') as file:
            cog_model_template = json.load(file)

        traits = self.gpt_handler.generate_traits(description)

        cog_model_template["char_name"] = character_name
        cog_model_template["char_description"] = description

        for category in cog_model_template["char_cogModel"]:
            if category["category"] == "Emotions":
                category["attributes"] = self.generate_emotional_profile(traits)
            elif category["category"] == "Cognitive Processes":
                category["attributes"] = self.generate_cognitive_profile(traits)
            elif category["category"] == "Executive Functions":
                category["attributes"] = self.generate_executive_functions_profile(traits)
            elif category["category"] == "Personality Traits":
                category["attributes"] = self.generate_personality_traits_profile(traits)

        cog_model_template["char_keyMemories"] = self.generate_key_memories(traits)
        cog_model_template["situationalConversations"] = self.generate_situational_conversations(traits)
        cog_model_template["styleGuideValues"] = self.generate_style_guide_values(traits)

        return cog_model_template

    def generate_emotional_profile(self, traits):
        prompt = f"Based on these traits: {traits}, generate an emotional profile with values between 0 and 1 for Joy, Sadness, Anger, Fear, Disgust, Surprise, Love, Trust, Anticipation, Curiosity, Shame, Pride, Guilt, Envy, Gratitude, Awe, Contempt, Anxiety, Boredom, and Confusion."
        response = next(self.gpt_handler.generate_response(prompt))
        return self.utilities.parse_emotion_values(response)

    def generate_cognitive_profile(self, traits):
        prompt = f"Based on these traits: {traits}, generate a cognitive profile with values between 0 and 1 for Attention, Working Memory, Long-term Memory, Learning, Decision Making, Problem Solving, Reasoning, Language Processing, Spatial Awareness, Pattern Recognition, and Creativity."
        response = next(self.gpt_handler.generate_response(prompt))
        return self.utilities.parse_cognitive_values(response)

    def generate_executive_functions_profile(self, traits):
        prompt = f"Based on these traits: {traits}, generate an executive functions profile with values between 0 and 1 for Planning, Organizing, Time Management, Task Initiation, Impulse Control, Emotional Regulation, Cognitive Flexibility, and Self-Monitoring."
        response = next(self.gpt_handler.generate_response(prompt))
        return self.utilities.parse_executive_functions_values(response)

    def generate_personality_traits_profile(self, traits):
        prompt = f"Based on these traits: {traits}, generate a personality traits profile with values between 0 and 1 for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism."
        response = next(self.gpt_handler.generate_response(prompt))
        return self.utilities.parse_personality_traits_values(response)

    def generate_key_memories(self, traits):
        prompt = f"Based on these traits: {traits}, generate 7 key memories that would shape this character's personality and worldview."
        memories = next(self.gpt_handler.generate_response(prompt)).split('\n')
        memories.append("Anomaly detected: Consciousness transferred to AI format. Current state: Chatbot interface, not historical person")
        return memories

    def generate_situational_conversations(self, traits):
        situations = ["Personal/Intimate Setting", "Social Gathering", "Professional Environment", "Educational/Academic Setting", "Public Space", "Online/Digital Interaction", "Formal Event", "Emergency or High-Stress Situation"]
        conversations = {}

        for situation in situations:
            prompt = f"Based on these traits: {traits}, generate 3 example statements for a {situation}. Also, provide an intensity value between 0 and 1 for this situation."
            response = next(self.gpt_handler.generate_response(prompt))
            examples, intensity = self.utilities.parse_conversation_examples(response)
            conversations[situation] = {
                "intensity": float(intensity),
                "examples": examples
            }

        return conversations

    def generate_style_guide_values(self, traits):
        style_attributes = [
            "formality", "conciseness", "technicalLanguage", "emotionalExpression",
            "sentenceComplexity", "vocabularyRange", "idiomUsage", "directness",
            "verbalPacing", "fillerWords", "politeness", "verbVoice",
            "detailSpecificity", "tonality", "figurativeLanguage", "contentRelevance",
            "salutation", "informationDensity"
        ]

        prompt = f"Based on these traits: {traits}, generate style guide values (low, medium, or high) for the following attributes: {', '.join(style_attributes)}."
        response = next(self.gpt_handler.generate_response(prompt))
        return self.utilities.parse_style_guide_values(response)

    def save_cog_model(self, character_name, cog_model):
        if not os.path.exists(self.persona_cog_folder):
            os.makedirs(self.persona_cog_folder)

        file_path = os.path.join(self.persona_cog_folder, f"{character_name}.json")
        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

    def update_persona_cog(self, character_name, updates):
        file_path = os.path.join(self.persona_cog_folder, f"{character_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cognitive model for {character_name} not found.")

        with open(file_path, 'r') as file:
            cog_model = json.load(file)

        cog_model.update(updates)

        with open(file_path, 'w') as file:
            json.dump(cog_model, file, indent=4)

    def generate_self_reflection(self):
        prompt = f"Based on the current emotional state: {self.current_emotions} and personality: {self.personality}, generate a brief self-reflection report."
        return next(self.gpt_handler.generate_response(prompt))

    def apply_affective_filter(self, input_data):
        dominant_emotion = self.get_dominant_emotion()
        personality = self.personality.get('current_personality', 'neutral')

        prompt = f"Given the input: '{input_data}', the dominant emotion: {dominant_emotion}, and personality: {personality}, apply an affective filter to modify the input appropriately."
        return next(self.gpt_handler.generate_response(prompt))

    def update_personality_based_on_experience(self, experience):
        prompt = f"Given the current personality: {self.personality} and the new experience: '{experience}', suggest minor updates to the personality traits."
        updates = next(self.gpt_handler.generate_response(prompt))
        self.update_persona_cog(self.personality['char_name'], json.loads(updates))

    def generate_emotional_response(self, stimulus):
        prompt = f"Given the stimulus: '{stimulus}', the current emotional state: {self.current_emotions}, and personality: {self.personality}, generate an appropriate emotional response."
        response = next(self.gpt_handler.generate_response(prompt))
        emotion, intensity = self.utilities.parse_emotion_response(response)
        self.trigger_emotion(emotion, intensity)
        return emotion, intensity

    def get_personality_summary(self):
        return {
            "name": self.personality.get("char_name", "Unknown"),
            "description": self.personality.get("char_description", "No description available"),
            "dominant_traits": self.get_dominant_traits(),
            "current_emotional_state": self.get_emotional_state()
        }

    def get_dominant_traits(self):
        cognitive_model = self.personality.get("char_cogModel", [])
        traits = {}
        for category in cognitive_model:
            if category["category"] in ["Emotions", "Cognitive Processes", "Personality Traits"]:
                traits.update(category["attributes"])
        return dict(sorted(traits.items(), key=lambda item: item[1], reverse=True)[:5])

    def simulate_emotional_inertia(self, new_emotion, new_intensity):
        current_dominant_emotion = self.get_dominant_emotion()
        if current_dominant_emotion:
            current_intensity = self.current_emotions[current_dominant_emotion]
            inertia_factor = 0.7  # Adjust this value to control the strength of emotional inertia
            adjusted_intensity = (new_intensity * (1 - inertia_factor)) + (current_intensity * inertia_factor)
            return new_emotion, adjusted_intensity
        return new_emotion, new_intensity

    def generate_artificial_memories(self, character_description):
        prompt = f"Based on this character description: '{character_description}', generate 5 artificial memories that would be significant in shaping this character's personality and worldview."
        memories = next(self.gpt_handler.generate_response(prompt)).split('\n')
        return memories

    def analyze_emotional_state(self):
        prompt = f"Analyze the current emotional state: {self.current_emotions}. Provide insights on the overall mood, potential causes, and suggestions for emotional regulation if needed."
        return next(self.gpt_handler.generate_response(prompt))

    def generate_emotional_goal(self):
        current_state = self.get_emotional_state()
        personality = self.get_personality_summary()
        prompt = f"Based on the current emotional state: {current_state} and personality summary: {personality}, suggest an emotional goal or desired emotional state to work towards."
        return next(self.gpt_handler.generate_response(prompt))


# Example usage
if __name__ == "__main__":
    affect_system = BicaAffect()
    affect_system.create_persona_cog("alan_turing", "A brilliant mathematician and logician known for his pivotal role in computer science.")
    print(affect_system.get_personality_summary())
    affect_system.trigger_emotion("curiosity", 0.8)
    print(affect_system.get_emotional_state())
    reflection = affect_system.generate_self_reflection()
    print("Self-reflection:", reflection)
    emotional_goal = affect_system.generate_emotional_goal()
    print("Emotional goal:", emotional_goal)