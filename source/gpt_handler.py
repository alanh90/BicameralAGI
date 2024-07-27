import openai
import util


class GPTHandler:
    def __init__(self, api_provider="openai", model="gpt-4o-mini"):
        self.api_provider = api_provider
        self.model = model

        if self.api_provider == "openai":
            openai.api_key = util.get_environment_variable("OPENAI_API_KEY")
        else:
            raise ValueError("Invalid API provider")

    async def generate_response(self, prompt, temperature=0.7, max_tokens=350, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
        if self.api_provider == "openai":
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            return response.choices[0].message.content.strip()

    def extract_text_from_response(self, response_content):
        if isinstance(response_content, list):
            text_blocks = [block.text.strip() for block in response_content if hasattr(block, 'text')]
            return ' '.join(text_blocks)
        return str(response_content).strip()
