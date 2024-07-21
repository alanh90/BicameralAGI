# gpt_handler.py
# This file contains the implementation for handling interactions with the GPT models.
# It includes functionalities for sending prompts, receiving responses, and managing the conversation context.
# This handler is crucial for integrating language model capabilities into the BiCA framework.

import openai
import util
import base64
from groq import Groq
from IPython.display import Image, display, Audio, Markdown


class GPTHandler:
    def __init__(self, api_provider="openai", model="gpt-4o"):
        self.api_provider = api_provider
        self.model = model

        if self.api_provider == "openai":
            self.ai_client = openai.OpenAI()
            self.ai_client.api_key = util.get_environment_variable("OPENAI_API_KEY")
        elif self.api_provider == "groq":
            self.groq_client = Groq(api_key=util.get_environment_variable("GROQ_API_KEY"))
        else:
            raise ValueError("Invalid API provider. Choose either 'openai' or 'claude'.")

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_visual_response(self, image_path, prompt="What do you feel when looking at this image?", temperature=0.7):
        if self.api_provider == "openai":
            base64_image = self.encode_image(image_path)
            response = self.ai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                temperature=temperature,
                max_tokens=350,
            )
            print("Thinking...")
            return response.choices[0].message.content.strip()

    def generate_response(self, prompt, api_provider="openai", temperature=0.7, max_tokens=350, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None, stream=False):
        if self.api_provider == "openai":
            response = self.ai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            print("Thinking...")
            return response.choices[0].message.content.strip()
        elif api_provider == "groq":
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI."},
                    {"role": "user", "content": prompt}
                ],
                model='Mixtral 8x7b',
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                stream=stream
            )
            print("Thinking...")
            return chat_completion.choices[0].message.content

    def extract_text_from_response(self, response_content):
        if isinstance(response_content, list):
            text_blocks = [block.text.strip() for block in response_content if hasattr(block, 'text')]
            return ' '.join(text_blocks)
        return str(response_content).strip()



