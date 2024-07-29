from gpt_handler import GPTHandler
import random


class Thoughts:
    def __init__(self, api_provider="openai", model="gpt-4o-mini"):
        self.gpt_handler = GPTHandler(api_provider=api_provider, model=model)

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
        Context: {context}
        Thoughts:
        """
        stream = self.gpt_handler.generate_response(prompt, temperature=temperature, stream=True)
        thoughts = ""
        for chunk in stream:
            thoughts += chunk
        print(f"Temperature used: {temperature}")
        return thoughts.strip()


def main():
    thoughts_generator = Thoughts()
    context = input("Enter a context for thought generation: ")
    thoughts = thoughts_generator.generate_thoughts(context)
    print("\nGenerated thoughts:")
    print(thoughts)


if __name__ == "__main__":
    main()
