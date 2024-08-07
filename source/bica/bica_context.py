from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from gpt_handler import GPTHandler


class BicaContext:
    def __init__(self, max_length=1000):
        self.context_viewpoints = {"positive": "", "neutral": "", "negative": ""}
        self.weights = {"positive": "med", "neutral": "med", "negative": "med"}  # Initial equal weights
        self.max_length = max_length
        self.gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Load the embedding model

    def update_weights(self, new_info):
        # Generate embeddings for new_info
        new_info_embedding = self.model.encode(new_info, convert_to_tensor=True)

        weights = {}
        for viewpoint in self.context_viewpoints:
            prompt = f"Given the current conversation context: {self.context_viewpoints[viewpoint]}\n\nNew information: {new_info}\n\nProvide a brief interpretation of the {viewpoint} aspects of this information."
            response = next(self.gpt_handler.generate_response(prompt)).strip()
            response_embedding = self.model.encode(response, convert_to_tensor=True)
            similarity = 1 - cosine(new_info_embedding, response_embedding)  # Cosine similarity

            print(f"{viewpoint.capitalize()} value: {similarity}")
            if similarity < 0.5:
                weights[viewpoint] = "low"
            elif 0.5 <= similarity < 0.80:
                weights[viewpoint] = "med"
            else:
                weights[viewpoint] = "high"

        # Ensure caution is slightly more valued
        if weights["negative"] == "med":
            weights["negative"] = "high"
        elif weights["negative"] == "low":
            weights["negative"] = "med"

        self.weights = weights

    def update_context(self, new_info):
        # Reset context viewpoints before updating
        self.context_viewpoints = {"positive": "", "neutral": "", "negative": ""}
        self.update_weights(new_info)
        for viewpoint in self.context_viewpoints:
            if viewpoint == "positive":
                prompt = f"Given the current conversation context: {self.context_viewpoints[viewpoint]}\n\nNew information: {new_info}\n\nProvide a brief interpretation of the positive aspects or benefits of this information."
            elif viewpoint == "neutral":
                prompt = f"Given the current conversation context: {self.context_viewpoints[viewpoint]}\n\nNew information: {new_info}\n\nProvide a brief neutral interpretation of this information."
            elif viewpoint == "negative":
                prompt = f"Given the current conversation context: {self.context_viewpoints[viewpoint]}\n\nNew information: {new_info}\n\nProvide a brief interpretation of the potential negative aspects, such as manipulation, danger, deception, of this information."

            updated_context = next(self.gpt_handler.generate_response(prompt))

            self.context_viewpoints[viewpoint] = updated_context.strip()
            if len(self.context_viewpoints[viewpoint]) > self.max_length:
                self.compress_context(viewpoint)

    def compress_context(self, viewpoint):
        prompt = f"Summarize the following {viewpoint} context briefly, retaining key information:\n\n{self.context_viewpoints[viewpoint]}"

        compressed_context = next(self.gpt_handler.generate_response(prompt))

        self.context_viewpoints[viewpoint] = compressed_context.strip()

    def get_context(self):
        return self.context_viewpoints

    def get_weighted_context(self):
        weighted_context = {}
        for viewpoint, context in self.context_viewpoints.items():
            weighted_context[viewpoint] = f"Weight: {self.weights[viewpoint]}\n{context}"
        return weighted_context

    def wipe_context(self):
        for viewpoint in self.context_viewpoints:
            self.context_viewpoints[viewpoint] = ""


def generate_ai_response(memory, context_viewpoints, weights):
    # Construct the prompt based on memory and weighted context viewpoints
    prompt = f"Memory: {' '.join(memory[-5:])}\n\n"  # Consider only the last 5 memory entries for brevity
    for viewpoint, context in context_viewpoints.items():
        prompt += f"{viewpoint.capitalize()} context (Weight: {weights[viewpoint]}):\n{context}\n\n"
    prompt += "Based on the memory and contexts, generate an appropriate response."

    # Generate response using the GPT handler
    gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")
    response = next(gpt_handler.generate_response(prompt))

    return response.strip()


def chatbot():
    context = BicaContext()
    memory = []

    print("Start chatting with the AI (type 'exit' to end the conversation):\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Update memory with user input
        memory.append(f"You: {user_input}")

        # Update context with new information
        context.update_context(user_input)

        # Get updated context viewpoints and weights
        contexts = context.get_context()
        weights = context.weights

        # Generate AI response based on memory and weighted context
        ai_response = generate_ai_response(memory, contexts, weights)

        # Update memory with AI response
        memory.append(f"AI: {ai_response}")

        print(f"AI: {ai_response}\n")


if __name__ == "__main__":
    chatbot()
