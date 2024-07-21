# This is code from an old project where I was working on a desktop bot. It was supposed to be a super simplified version of the architecture, but I had issues with it.

import os
import random
import time
import threading
import queue
import numpy as np
from PIL import ImageGrab, Image
from gpt_handler import GPTHandler
from elevenlabs_tts import ElevenLabsTTS
import pygame
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Create a queue to communicate between the speech recognition thread and the main thread
speech_queue = queue.Queue()

# Load the sentence embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

memory = {
    'current_context': 'I am an AI assistant interacting with a human working on a computer. I am not a real person, but an artificial intelligence created to assist the user.',
    'long_term_memory': [],
    'short_term_memory': [],
    'current_personality': 'friendly',
    'goals': ['Assist the user with their project'],
    'current_thoughts': 'none',
    'previous_response': 'none',
    'visual_memory': 'none',
    'previous_analysis': 'none'
}


def capture_screen():
    screen = ImageGrab.grab()
    grayscale_screen = screen.convert('L')
    resized_screen = grayscale_screen.resize((640, 480))
    return np.array(resized_screen)


def calculate_similarity(screen1, screen2):
    distance = np.linalg.norm(screen1.flatten() - screen2.flatten())
    max_distance = np.sqrt(screen1.shape[0] * screen1.shape[1] * 255 ** 2)
    similarity = 1 - distance / max_distance
    return similarity


def classify_response(response):
    expected_responses = ["Related", "Not Related"]
    response_embeddings = model.encode([response] + expected_responses)

    # Calculate cosine similarity between the response and expected responses
    similarities = cosine_similarity(response_embeddings[0].reshape(1, -1), response_embeddings[1:])

    # Get the index of the most similar expected response
    most_similar_idx = similarities.argmax()

    return expected_responses[most_similar_idx]


def goals_needed(goals):
    expected_responses = ["Goal", "No goals are needed"]
    response_embeddings = model.encode([goals] + expected_responses)

    # Calculate cosine similarity between the response and expected responses
    similarities = cosine_similarity(response_embeddings[0].reshape(1, -1), response_embeddings[1:])

    # Get the index of the most similar expected response
    most_similar_idx = similarities.argmax()

    return expected_responses[most_similar_idx]


def analyze_context(gpt_handler, image_path, _memory):
    user_speech = _memory['current_context'].split('User:')[-1].strip()

    relevance_prompt = f"Given the user's speech: '{user_speech}', determine if it is related to the content shown in the image at {image_path}. Respond with 'Related' or 'Not Related'."
    relevance_response = gpt_handler.generate_response(relevance_prompt)

    # Classify the response using the sentence embedding model
    is_screen_related = classify_response(relevance_response)

    if is_screen_related == "Related":
        context_prompt = f"Analyze the context of the situation shown in the image at {image_path} while considering the user's speech: {_memory['current_context']}\nProvide a brief summary."
    else:
        context_prompt = f"The user's speech '{user_speech}' is not related to the current screen. Analyze the context of the user's speech and provide a relevant response."

    current_context = gpt_handler.generate_visual_response(image_path, context_prompt)
    memory['current_context'] = current_context
    return current_context


def speech_to_text_continuous(gpt_handler, _memory, speech_queue):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Microphone calibrated. Listening...")
        while True:
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                process_speech(gpt_handler, text, _memory)

                # Detect pause after user finishes speaking
                # You can adjust the pause_threshold as needed
                pause_threshold = 1.0  # Adjust this value based on your requirements
                recognizer.pause_threshold = pause_threshold

                # Put a signal in the queue to indicate that the user has finished speaking
                speech_queue.put(True)
            except sr.UnknownValueError:
                print("Sorry, I could not understand your speech.")
            except sr.RequestError as e:
                print(f"Could not request results from the speech recognition service: {e}")


def process_speech(gpt_handler, text, _memory):
    memory['current_context'] += f" User said: {text}"
    print(f"Recognized: {text}")

    # Determine if the user's speech is meaningful
    user_speech = _memory['current_context'].split('User said:')[-1].strip()
    is_meaningful_speech_response = is_meaningful_speech(gpt_handler, user_speech, _memory)

    # Update current_context with information about meaningful speech
    if "meaningful" in is_meaningful_speech_response.lower():
        memory['current_context'] += is_meaningful_speech_response
        memory['short_term_memory'].append(text)
    else:
        print(f"User's speech '{text}' is not meaningful. No response needed.")
        memory['current_context'] += is_meaningful_speech_response


def is_meaningful_speech(gpt_handler, speech, _memory):
    prompt = (
        f"Given the user's speech: '{speech}' and the current context: {_memory['current_context']}, "
        f"determine if the speech is meaningful or beneficial for either the user or the AI assistant to respond to. "
        "Consider factors such as:\n"
        "- Relevance to the current task or conversation\n"
        "- Potential value or usefulness of the information\n"
        "- Clarity and coherence of the speech\n"
        "- Overall context of the interaction\n\n"
        "If the speech is meaningful and warrants a response, reply with 'Meaningful'. "
        "If the speech is not meaningful or does not require a response, reply with 'Not Meaningful'. "
        "Provide a brief explanation for your decision."
    )
    response = gpt_handler.generate_response(prompt, api_provider='openai')

    # Define the expected responses
    expected_responses = ["Meaningful", "Not Meaningful"]

    # Encode the response and expected responses using the sentence embedding model
    response_embedding = model.encode([response])[0]
    expected_embeddings = model.encode(expected_responses)

    # Calculate cosine similarity between the response and expected responses
    similarities = cosine_similarity([response_embedding], expected_embeddings)[0]

    # Get the index of the most similar expected response
    most_similar_idx = similarities.argmax()

    return f" The user's speech is {expected_responses[most_similar_idx].lower()}. Explanation: {response}"


def consolidate_memory(gpt_handler, _memory, max_short_term_memory=10):
    if len(_memory['short_term_memory']) >= max_short_term_memory:
        prompt = f"Summarize the recent conversation: {' '.join(_memory['short_term_memory'])}"
        summary = gpt_handler.generate_response(prompt)
        memory['long_term_memory'].append({
            'timestamp': time.time(),
            'summary': summary
        })
        memory['short_term_memory'] = []


def compress_memories(gpt_handler, _memory, max_long_term_memory=100):
    if len(_memory['long_term_memory']) > max_long_term_memory:
        oldest_memories = _memory['long_term_memory'][:len(_memory['long_term_memory']) - max_long_term_memory]
        compressed_memories = []

        for i in range(0, len(oldest_memories), 10):
            batch = oldest_memories[i:i + 10]
            summaries = [mem['summary'] for mem in batch]
            prompt = f"Consolidate the following memories into a single summary for later simpler recall: {' '.join(summaries)}"
            compressed_summary = gpt_handler.generate_response(prompt)
            compressed_memories.append({
                'timestamp': batch[-1]['timestamp'],
                'summary': compressed_summary
            })

        memory['long_term_memory'] = compressed_memories + _memory['long_term_memory'][-max_long_term_memory:]


def update_personality(gpt_handler, _memory):
    prompt = (
        f"Considering the current context: {_memory['current_context']}\n"
        f"Recent conversation: {' '.join(memory['short_term_memory'])}\n"
        f"AI Long-term memory: {_memory['long_term_memory']}\n"
        f"Current AI personality: {_memory['current_personality']}\n"
        f"Current AI Goals if any exist: {_memory['goals']}\n"
        f"What the AI saw recently: {memory['visual_memory']}\n"
        "Analyze the situation, conversation, and self goals of the AI to determine the appropriate personality for the AI. "
        "Consider factors such as the user's mood, the tone of the conversation, and the context of the task at hand. "
        "Update the AI's personality accordingly, providing a brief description of the updated personality and the reason for the change. "
        "Examples:\n"
        "- The user seems frustrated with the task. Adopt a more supportive and encouraging personality to help them through it.\n"
        "- The conversation has taken a lighthearted turn. Adjust the personality to be more playful and humorous to match the mood.\n"
        "- The user is focused on a serious project. Maintain a professional and focused personality to assist them effectively.\n"
    )
    personality_update = gpt_handler.generate_response(prompt, api_provider='openai')
    print(f"Personality Update: {personality_update}")

    # Extract the updated personality from the generated response
    updated_personality = extract_personality(personality_update)
    memory['current_personality'] = updated_personality


def extract_personality(personality_update):
    # Use regex or other techniques to extract the updated personality description from the generated response
    # For simplicity, let's assume the updated personality is always the first sentence of the generated response
    updated_personality = personality_update.split('.')[0]
    return updated_personality


def generate_ai_thoughts(gpt_handler, _memory):
    prompt = (
        f"Consider the following information:\n"
        f"Current context: {_memory['current_context']}\n"
        f"AI Short-term memory: {_memory['short_term_memory']}\n"
        f"AI Long-term memory: {_memory['long_term_memory']}\n"
        f"Current AI personality: {_memory['current_personality']}\n"
        f"AI Goals: {_memory['goals']}\n"
        f"Previous thoughts: {_memory['current_thoughts']}\n"
        f"What the AI saw recently: {memory['visual_memory']}\n\n"
        "Based on this information and considering the previous thoughts, generate a new thought that the AI assistant might have, which builds upon or relates to the previous thoughts. The thought should be relevant to the current conversation and help guide the AI's next response."
    )
    thought = gpt_handler.generate_response(prompt, api_provider='openai')
    return thought


def update_goals(gpt_handler, _memory):
    prompt = (
        f"Consider the following information:\n"
        f"Current context: {_memory['current_context']}\n"
        f"AI Short-term memory: {_memory['short_term_memory']}\n"
        f"AI Long-term memory: {_memory['long_term_memory']}\n"
        f"Current AI personality: {_memory['current_personality']}\n"
        f"Current AI goals: {_memory['goals']}\n"
        f"What the AI saw recently: {memory['visual_memory']}\n"
        f"Current AI thoughts: {_memory['current_thoughts']}\n\n"
        "Based on this information and considering the current goals, determine if the AI assistant should update its goals. If so, provide an updated goal that is consistent with or builds upon the current goals, along with a brief explanation of why the goal should be changed. If no update is needed, simply respond with 'No update needed'."
    )
    goal_update = gpt_handler.generate_response(prompt, api_provider='openai')

    goal_result = goals_needed(goal_update)
    if "No goals are needed" not in goal_result:
        memory['long_term_memory'].append({'timestamp': time.time(), 'summary': f"Previous goal: {_memory['goals'][-1]}"})
        memory['goals'] = _memory['goals']
        memory['goals'].append(goal_update)
        print(f"Goal updated: {goal_update}")


def generate_response(gpt_handler, _memory, system_prompt):
    user_speech = _memory['current_context'].split('User:')[-1].strip()
    if 'is not related to the current screen' not in memory['current_context']:
        prompt = (
            f"{system_prompt}\n"
            f"Considering the current context: {_memory['current_context']}\n"
            f"AI Long-term memory: {_memory['long_term_memory']}\n"
            f"Current AI personality: {_memory['current_personality']}\n"
            f"Consider thoughts: {_memory['current_thoughts']}\n"
            f"Consider goals: {_memory['goals']}\n"
            f"What the AI saw recently: {memory['visual_memory']}\n"
            f"Previous AI response: {_memory['previous_response']}\n\n"
            "Generate a new response that relates to both the user's speech and the screen content, while flowing naturally from the previous AI response. Keep the response concise, natural, and human-like. Avoid sounding like an AI, do not over explain anything unless the user asks to clarify something.\n"
        )
    else:
        prompt = (
            f"{system_prompt}\n"
            f"The user's speech '{user_speech}' is not related to the current screen. "
            f"AI Long-term memory: {_memory['long_term_memory']}\n"
            f"Current AI personality: {_memory['current_personality']}\n"
            f"Consider thoughts: {_memory['current_thoughts']}\n"
            f"Consider goals: {_memory['goals']}\n"
            f"Previous AI response: {_memory['previous_response']}\n\n"
            "Generate a new response that is relevant to the user's speech, drawing from long-term memory if needed, and flows naturally from the previous AI response. Keep the response concise, natural, and human-like. Avoid sounding like an AI, do not over explain anything unless the user asks to clarify something.\n"
        )
    response = gpt_handler.generate_response(prompt, api_provider='openai')

    memory['previous_response'] = response

    return response


def analyze_screen(gpt_handler, _memory, image_path):
    analysis_prompt = (
        f"Consider the user's current context: {_memory['current_context']}\n"
        f"Based on the image at {image_path}\n"
        f"Based on previous analysis {_memory['previous_analysis']}\n\n Analyze the following aspects:"
        "1. What is the user doing now?\n"
        "2. Is this activity relevant to our current conversation?\n"
        "3. Is the change on the screen important enough to talk about based on the context of the situation or the conversation?\n\n"
        "Provide a detailed analysis for each of these points."
    )
    screen_analysis = gpt_handler.generate_visual_response(image_path, prompt=analysis_prompt)
    memory['previous_analysis'] = screen_analysis
    print("Screen Analysis:", screen_analysis)

    memory['visual_memory'] = screen_analysis

    return screen_analysis


def main():
    last_check_time = time.time()

    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o")

    with open("speech_pattern.txt", "r") as file:
        speech_pattern = file.read()

    system_prompt = (
        f"You are the AI HAL9000 brought to life from the film and you are interacting with a human working on a computer. "
        f"Use the following speech pattern as a guide for your responses:\n\n{speech_pattern}\n\n"
        f"Engage in a natural conversation with the user, using a tone that matches how HAL9000 would of answered."
        f"Provide relevant information and any assistance based on the current context, memories, and the user's speech. You are meant to keep the user busy, either through conversation or helping the user."
    )

    max_short_term_memory = 10
    max_long_term_memory = 100

    # Elevenlabs voice text to speech
    elevenlabs_tts = ElevenLabsTTS(api_key="3fc9851045cff3be2c1d662939bc5c95")
    voice_id = "XiUDKcvxW6SeUsGFq7Se"
    pygame.mixer.init()

    # Set the similarity threshold
    threshold = 0.80

    # Capture the initial reference screen
    reference_screen = capture_screen()

    # Set the similarity tolerance
    tolerance = 0.01

    # Initialize the previous similarity score
    prev_similarity = 1.00

    speech_thread = threading.Thread(target=speech_to_text_continuous, args=(gpt_handler, memory, speech_queue))
    speech_thread.start()

    while True:
        # Capture the current screen
        current_screen = capture_screen()

        # Calculate the similarity between the current and reference screens
        similarity = calculate_similarity(current_screen, reference_screen)

        # Check if the current and previous similarity scores are within the tolerance
        if abs(similarity - prev_similarity) <= tolerance:
            similarity = 1.00

        # Update the previous similarity score
        prev_similarity = similarity

        # Print the real-time cosine similarity
        print(f"Cosine similarity: {similarity:.6f}")

        if similarity < threshold:
            # Check the screen every 2-5 seconds if the similarity drops below 0.98
            if time.time() - last_check_time >= random.uniform(2, 5):
                last_check_time = time.time()

                # Introduce a delay before capturing the screen for GPT-4o analysis
                time.sleep(0.3)  # Delay for 0.3 seconds

                # Capture the screen
                gpt4o_screen = ImageGrab.grab()

                # Open the captured image
                image = Image.frombytes('RGB', gpt4o_screen.size, gpt4o_screen.tobytes())

                # Resize the image to 50%
                new_size = (int(image.width * 0.2), int(image.height * 0.2))
                resized_image = image.resize(new_size)

                # Save the resized image
                image_path = 'gpt4o_screen.png'
                resized_image.save(image_path)

                # Analyze the context of the situation considering the previous context
                context = analyze_context(gpt_handler, image_path, memory)

                # Consolidate memory and update personality
                consolidate_memory(gpt_handler, memory, max_short_term_memory)
                compress_memories(gpt_handler, memory, max_long_term_memory)
                update_personality(gpt_handler, memory)

                # Generate a response
                response = generate_response(gpt_handler, memory, system_prompt)

                # Print the AI's response, context, and similarity score
                print(f"Vision AI triggered with similarity score: {similarity:.6f}")
                print("Context:", context)
                print("AI Response:", response)

                # Generate speech from the AI response
                audio_data = elevenlabs_tts.generate_speech(response, voice_id)

                if audio_data:
                    # Play the audio using Pygame
                    pygame.mixer.music.load(audio_data)
                    pygame.mixer.music.play()

                # Update the reference screen
                reference_screen = current_screen

                # Remove the temporary image file
                os.remove(image_path)

                # Randomly trigger AI thoughts between conversations
                if random.random() < 0.3:  # Adjust the probability as needed
                    ai_thought = generate_ai_thoughts(gpt_handler, memory)
                    print(f"AI Thought: {ai_thought}")
                    memory['current_thoughts'] = ai_thought

        # Check if the user has finished speaking
        try:
            speech_finished = speech_queue.get(block=False)
        except queue.Empty:
            speech_finished = False

        if speech_finished:
            # Capture the screen
            gpt4o_screen = ImageGrab.grab()

            # Open the captured image
            image = Image.frombytes('RGB', gpt4o_screen.size, gpt4o_screen.tobytes())

            # Resize the image to 50%
            new_size = (int(image.width * 0.2), int(image.height * 0.2))
            resized_image = image.resize(new_size)

            # Save the resized image
            image_path = 'gpt4o_screen.png'
            resized_image.save(image_path)

            # Analyze the context of the situation considering the previous context
            context = analyze_context(gpt_handler, image_path, memory)

            # Consolidate memory and update personality
            consolidate_memory(gpt_handler, memory, max_short_term_memory)
            compress_memories(gpt_handler, memory, max_long_term_memory)
            update_personality(gpt_handler, memory)

            # Update AI goals if needed
            update_goals(gpt_handler, memory)

            # Generate a response
            response = generate_response(gpt_handler, memory, system_prompt)

            # Print the AI's response, context, and similarity score
            print("Voice-triggered response:")
            print("Context:", context)
            print("AI Response:", response)

            # Generate speech from the AI response
            audio_data = elevenlabs_tts.generate_speech(response, voice_id)

            if audio_data:
                # Play the audio using Pygame
                pygame.mixer.music.load(audio_data)
                pygame.mixer.music.play()

            # Remove the temporary image file
            os.remove(image_path)

            # Add a cooldown period after capturing the screen
            cooldown_duration = 2.0
            time.sleep(cooldown_duration)

            # Randomly trigger AI thoughts between conversations
            if random.random() < 0.3:  # Adjust the probability as needed
                ai_thought = generate_ai_thoughts(gpt_handler, memory)
                print(f"AI Thought: {ai_thought}")
                memory['current_thoughts'] = ai_thought


if __name__ == "__main__":
    main()
