from gpt_handler import GPTHandler


class Context:
    def __init__(self, max_length=1000):
        self.context = ""
        self.max_length = max_length
        self.gpt_handler = GPTHandler(api_provider="openai", model="gpt-4o-mini")

    def update_context(self, new_info):
        prompt = f"Current context: {self.context}\n\nNew information: {new_info}\n\nAnalyze and integrate this new information into the context."
        updated_context = next(self.gpt_handler.generate_response(prompt))
        self.context = updated_context
        if len(self.context) > self.max_length:
            self.compress_context()

    def compress_context(self):
        prompt = f"Summarize the following context, retaining key information. Try not to summarize the latest information because it may be relevant for the current active situation.:\n\n{self.context}"
        compressed_context = next(self.gpt_handler.generate_response(prompt))
        self.context = compressed_context

    def get_context(self):
        return self.context

    def wipe_context(self):
        self.context = ""
