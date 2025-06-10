# llm/advisor.py
import os
import asyncio
import textwrap
import google.generativeai as genai

# --- SECURITY: Load the API key from environment variables ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not found. Please set it before running.")

# Configure the generative AI client
genai.configure(api_key=GEMINI_API_KEY)


def _build_prompt(kpis: dict, weights: dict, user_goal: str) -> str:
    """Builds a detailed prompt for the LLM based on the user's goal."""

    # The prompt building logic remains the same as it's excellent.
    base_prompt = textwrap.dedent(f"""
        You are an expert AI financial advisor named Gemini, speaking to a beginner investor.
        Your tone should be clear, educational, and reassuring. Avoid complex jargon.

        Here is the performance analysis of an AI-driven investment strategy over the last 1.5 years:
        - Cumulative Return: {kpis.get('Cumulative Return', 0):.2%}
        - Max Drawdown (peak-to-trough loss): {kpis.get('Max Drawdown', 0):.2%}
        - Annualized Sharpe Ratio (risk-adjusted return): {kpis.get('Annualized Sharpe Ratio', 0):.2f}

        The AI's final recommended portfolio allocation is:
        - Apple (AAPL): {weights.get('AAPL', 0):.1%}
        - Microsoft (MSFT): {weights.get('MSFT', 0):.1%}
        - Google (GOOGL): {weights.get('GOOGL', 0):.1%}

        ---
    """)

    if user_goal == "Long-Term Growth":
        goal_instructions = textwrap.dedent("""
            The user's goal is 'Long-Term Growth'.
            Write a 3-paragraph report.
            1. Start with a positive summary of the strategy's performance.
            2. Explain that for long-term investing, consistency and avoiding major losses (drawdown) are key. Relate the results to this principle.
            3. Conclude with a reassuring statement about how this diversified portfolio of quality tech companies is a sensible approach for long-term goals.
        """)
    elif user_goal == "Mid-Term Balanced":
        goal_instructions = textwrap.dedent("""
            The user's goal is 'Mid-Term Balanced'.
            Write a 3-paragraph report.
            1. Summarize the performance, highlighting the Sharpe Ratio as a measure of the risk/reward balance.
            2. Explain the concept of a trade-off: this strategy aims for steady growth rather than maximum possible gains, making it suitable for a 3-5 year horizon.
            3. Conclude by mentioning that the allocation is actively managed but appears stable, which is good for a balanced approach.
        """)
    else:  # Short-Term Speculation
        goal_instructions = textwrap.dedent("""
            The user's goal is 'Short-Term Speculation'.
            Write a 3-paragraph report in a more direct tone.
            1. Directly state the cumulative return and the associated risk (Max Drawdown).
            2. Emphasize that short-term strategies are inherently high-risk and past performance does not guarantee future results.
            3. Conclude by stating that this allocation reflects current market indicators as analyzed by the AI, but should be monitored closely by a speculative investor.
        """)

    return base_prompt + goal_instructions


async def generate_investment_report(kpis: dict, weights: dict, user_goal: str) -> str:
    """The main function to generate a personalized investment report using the Gemini API."""
    
    print(f"\n--- [Gemini Advisor] Generating report for '{user_goal}'... ---")
    
    try:
        # 1. Select the model
        model = genai.GenerativeModel('gemini-2.0-flash')

        # 2. Build the prompt
        prompt = _build_prompt(kpis, weights, user_goal)
        
        # 3. Call the real Gemini API asynchronously
        response = await model.generate_content_async(prompt)
        
        # 4. Return the generated text
        return response.text

    except Exception as e:
        error_message = f"An error occurred while calling the Gemini API: {e}"
        print(error_message)
        return error_message