import json
import random
import time
import os

class BicaEmotions:
    def __init__(self, personality_file='persona_cog_template.json', memory_file='bica_memory.json'):
        self.personality = self.load_personality(personality_file)
        self.memory = self.load_memory(memory_file)
        self.current_emotions = {}
        self.emotional_memory = []
        self.emotion_falloff_rate = 0.05  # rate at which emotions decay
        self.last_update = time.time()

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

# Example usage
if __name__ == "__main__":
    emotions = BicaEmotions()
    emotions.trigger_emotion('happiness', 0.8)
    emotions.trigger_emotion('sadness', 0.4)
    print(emotions.get_emotional_state())
    emotions.save_memory()
