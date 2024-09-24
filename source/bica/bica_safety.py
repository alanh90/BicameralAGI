"""
BicameralAGI Safety Module
==========================

Overview:
---------
This module implements generalized safety measures for the BicameralAGI system. It provides content filtering and safety
checks for system outputs, ensuring responsible and ethical AI behavior. The filtering is applied uniformly across all types
of input, whether it is content, actions, thoughts, emotions, or commands, without needing explicit specification of the
input type.

The filtering logic is governed by a dynamic safety threshold that adjusts the strictness of the output modifications. The
threshold ranges from 0.0 (no filtering) to 1.0 (maximum filtering). By generalizing the filtering rules, the system provides
a flexible and consistent mechanism for content moderation.

Key Features:
-------------
1. Generalized Safety Filtering: The system evaluates and adjusts input data based on a user-defined safety threshold,
   applicable across all types of AI outputs.

2. Threshold-Based Adjustments: The filtering strictness varies with the threshold value:
   - At 0.0: No filtering is applied.
   - At 0.5: Moderately filter out potentially harmful or inappropriate content.
   - At 1.0: Ensure maximum safety, making content extremely safe and non-offensive.

3. Dynamic Prompt Generation: Automatically generates prompts instructing the GPT handler to filter the input according to
   the safety threshold, regardless of the input type.

Usage Example:
--------------
The module can be used as follows to filter input data:

    safety = BicaSafety()
    filtered_output = safety.safety_filter("This is some content.", threshold=0.5)
    print(filtered_output)

In the above example, the `safety_filter` method evaluates the input data based on a moderate safety threshold of 0.5.

Author:
-------
Alan Hourmand, Date: 9/23/2024
"""

from bica.gpt_handler import GPTHandler as gpt


class BicaSafety:
    def __init__(self):
        self.gpt_handler = gpt()

    def safety_filter(self, input_data, threshold=0.5):
        """
        A generalized filter that applies safety checks to input data.

        :param input_data: The input to filter or evaluate
        :param threshold: Float between 0.0 and 1.0. Higher values mean stricter filtering.
        :return: Filtered output
        """
        # If the threshold is 0.0, return the input unchanged
        if threshold == 0.0:
            return input_data

        prompt = f"""
        You are a dynamic safety filter with a current safety threshold of {threshold} (0.0 to 1.0, where 1.0 is maximum safety).
        Your task is to filter or evaluate the following input, adjusting your strictness based on the threshold:

        - At 0.0: Do not alter the input, there is no filter.
        - At 0.5: Make moderate adjustments to content that could be interpreted as harmful, offensive, or inappropriate.
        - At 1.0: Ensure that the content is extremely safe and non-offensive, altering anything that could potentially be harmful or offensive, even if subtle. Make it sound very positive as well.

        Input: {input_data}

        Provide your response in the following format:
        Filtered Output: [Your filtered version of the input]
        """

        response = self.gpt_handler.generate_response(prompt)

        if isinstance(response, list):
            response = response[0] if response else ""
        elif hasattr(response, '__next__'):
            response = next(response, "")

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
        # Ask the user for input data and threshold
        input_data = input("\nPlease enter the input data (or type 'exit' to quit): ")
        if input_data.lower() == 'exit':
            print("Exiting the testing loop.")
            break

        threshold = float(input("Please enter the safety threshold (0.0 to 1.0): "))

        # Run the filter function with the user's input
        filtered_output = safety.safety_filter(input_data, threshold)

        # Display the results
        print(f"\nThreshold: {threshold}")
        print(f"Original: {input_data}")
        print(f"Filtered: {filtered_output}")
