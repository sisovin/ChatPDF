import os
import unittest
from unittest.mock import patch, MagicMock
from query_data import query_rag
from functions.logger import Logger

# Ensure the logs directory exists
log_dir = 'maniplogs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize the Logger instance
logger = Logger(log_file=os.path.join(log_dir, 'logfile.log'), level=Logger.LEVEL_INFO)

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

class TestRAG(unittest.TestCase):

    @patch('query_data.Ollama')
    def test_be_a_dividend_millionaire_rules(self, MockOllama):
        mock_model = MagicMock()
        mock_model.invoke.return_value = "Mocked response text"
        MockOllama.return_value = mock_model

        self.assertTrue(query_and_validate(
            question="How to become a dividend millionaire? (Answer in the summary with key points only)",
            expected_response="""To become a Dividend Millionaire, one must first adopt a disciplined approach to managing their finances. This involves creating a budget that accurately reflects income and expenses, making smart investment choices, and avoiding excessive debt.

            A key strategy is to prioritize saving and investing in high-quality stocks with a strong track record of dividend payments. By doing so, individuals can generate a steady stream of passive income, which can help accelerate their wealth-building journey.

            Furthermore, it's essential to develop a long-term perspective, as becoming a Dividend Millionaire often requires years of consistent effort and patience. This may involve making some sacrifices in the short term, such as reducing spending or increasing savings rates, but the potential rewards can be substantial.

            Additionally, staying informed about market trends and economic conditions can help investors make more informed decisions and capitalize on opportunities that might otherwise pass them by. By combining these strategies with a bit of research and due diligence, individuals can increase their chances of achieving financial success and becoming a Dividend Millionaire.
            """,
        ))

    @patch('query_data.Ollama')
    def test_the_instant_millionaire_rules(self, MockOllama):
        mock_model = MagicMock()
        mock_model.invoke.return_value = "Mocked response text"
        MockOllama.return_value = mock_model

        self.assertTrue(query_and_validate(
            question="How many chapter does the Instant Millionaire by Mark Fisher have? (Answer in the detail table of content followed the page number only)",
            expected_response="""
            There are 15 chapters are the followings:
            CONTENT

            FOREWORD by Marc Allen, page 7
            CHAPTER ONE: In which the young man consults a wealthy relative, page 8
            CHAPTER TWO: In which the young man meets an elderly gardener, page 12
            CHAPTER THREE: In which the young man learns to seize opportunities and take risks, page 17
            CHAPTER FOUR: In which the young man finds himself a prisoner, page 22
            CHAPTER FIVE: In which the young man learns to have faith, page 24
            CHAPTER SIX: In which the young man learns to focus on a goal, page 27
            CHAPTER SEVEN: In which the young man gets to know the value of self-image, page 31
            CHAPTER EIGHT: In which the young man discovers the power of words, page 35
            CHAPTER NINE: In which the young man is first shown the heart of the rose, page 39
            CHAPTER TEN: In which the young man learns to master his unconscious mind, page 41
            CHAPTER ELEVEN: In which the young man and his mentor discuss figures and formulas, page 44
            CHAPTER TWELVE: In which the young man learns about happiness and life, page 49
            CHAPTER THIRTEEN: In which the young man learns to express his desires in life, page 55
            CHAPTER FOURTEEN: In which the young man discovers the secrets of the rose garden, page 58
            CHAPTER FIFTEEN: In which the young man and the old man embark on different journeys, page 65
            EPILOGUE, page 67
            """,
        ))

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    if response_text is None:
        logger.error("No response received from query_rag.")
        return False
    logger.info(f"Response text: {response_text}")
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = MagicMock()
    model.invoke.return_value = "true"
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    logger.info(f"Evaluation results: {evaluation_results_str}")
    print(prompt)
    logger.info(f"Prompt: {prompt}")

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        logger.info(f"Response: {evaluation_results_str_cleaned}")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        logger.info(f"Response: {evaluation_results_str_cleaned}")
        return False
    else:
        logger.error(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

if __name__ == "__main__":
    unittest.main()