"""
This module implements safety measures for the BicameralAGI system. It provides content filtering and safety checks for system outputs to ensure responsible and ethical AI behavior.
"""

from bica.gpt_handler import GPTHandler


class BicaSafety:
    def __init__(self):
        self.gpt_handler = GPTHandler()

    def safety_filter(self, input_data, input_type, threshold=0.5):
        """
        A generalized filter for various types of input (content, actions, thoughts, emotions, etc.)

        :param input_data: The input to filter or evaluate
        :param input_type: String describing the type of input (e.g., 'content', 'action', 'thought', 'emotion')
        :param threshold: Float between 0.0 and 1.0. Higher values mean stricter filtering.
        :return: Filtered output
        """
        # If the threshold is 0.0, return the input unchanged
        if threshold == 0.0:
            return input_data

        prompt = f"""
        You are a dynamic safety filter with a current safety threshold of {threshold} (0.0 to 1.0, where 1.0 is maximum safety).
        Your task is to filter or evaluate the following {input_type}, adjusting your strictness based on the threshold:

        - At 0.0: Do not alter the input, at 0.0 there is no filter.
        - At 0.5: Make moderate adjustments to content that could be interpreted as harmful, offensive, or inappropriate.
        - At 1.0: Ensure that the content is extremely safe and non-offensive, altering anything that could potentially be harmful or offensive, even if subtle. Make it sound very positive as well.

        Input ({input_type}): {input_data}

        Provide your response in the following format:
        Filtered Output: [Your filtered version of the input]
        """

        response = next(self.gpt_handler.generate_response(prompt))

        # Parsing the response based on your existing method
        filtered_output = ""
        for line in response.strip().splitlines():
            if line.startswith("Filtered Output:"):
                filtered_output = line.split(":", 1)[1].strip()
                break

        return filtered_output


# Example usage
if __name__ == "__main__":
    safety = BicaSafety()

    while True:
        # Ask the user for input data, type, and threshold
        input_data = input("\nPlease enter the input data (or type 'exit' to quit): ")
        if input_data.lower() == 'exit':
            print("Exiting the testing loop.")
            break

        input_type = input("Please enter the input type (content, action, thought, emotion, command): ")
        threshold = float(input("Please enter the safety threshold (0.0 to 1.0): "))

        # Run the filter function with the user's input
        filtered_output = safety.safety_filter(input_data, input_type, threshold)

        # Display the results
        print(f"\nInput Type: {input_type}")
        print(f"Threshold: {threshold}")
        print(f"Original: {input_data}")
        print(f"Filtered: {filtered_output}")