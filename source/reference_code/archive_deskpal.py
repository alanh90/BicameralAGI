# This is code from an old project where I was working on a desktop bot. It was supposed to be a super simplified version of the architecture, but I had issues with it.
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the sentence embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


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
