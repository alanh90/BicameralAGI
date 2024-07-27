import asyncio
from gpt_handler import GPTHandler

gpt_handler = GPTHandler(api_provider="openai", model="gpt-4")
thought_queue = asyncio.Queue()


async def generate_thoughts(prompt):
    thoughts = await gpt_handler.generate_response(prompt)
    for thought in thoughts.split('\n'):
        print(f"Generated Thought: {thought}")
        await thought_queue.put(thought)
    await thought_queue.put(None)  # Signal that thought generation is done


async def analyze_thought():
    while True:
        thought = await thought_queue.get()
        if thought is None:
            break
        analysis_prompt = f"Analyze the following thought: {thought}"
        analysis = await gpt_handler.generate_response(analysis_prompt)
        print(f"Thought Analysis: {analysis}")


async def main():
    initial_prompt = "Start generating thoughts about the nature of existence."
    await asyncio.gather(
        generate_thoughts(initial_prompt),
        analyze_thought()
    )


if __name__ == "__main__":
    asyncio.run(main())
