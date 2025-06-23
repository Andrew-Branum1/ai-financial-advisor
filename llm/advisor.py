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


# In llm/advisor.py, replace the existing _build_prompt function

def _build_prompt(kpis: dict, weights: dict, user_goal: str) -> str:
    """Builds a structured, data-driven prompt for the LLM."""

    base_prompt = textwrap.dedent(f"""
        You are an expert AI financial advisor named Gemini, speaking to a beginner investor.
        Your tone should be clear, educational, and reassuring. Use simple language and avoid jargon.

        Here is the performance data for an AI-driven investment strategy:
        - Cumulative Return: {kpis.get('Cumulative Return', 0):.2%}
        - Max Drawdown: {kpis.get('Max Drawdown', 0):.2%}
        - Annualized Sharpe Ratio: {kpis.get('Annualized Sharpe Ratio', 0):.2f}

        The AI's final recommended portfolio allocation is:
        - Apple (AAPL): {weights.get('AAPL', 0):.1%}
        - Microsoft (MSFT): {weights.get('MSFT', 0):.1%}
        - Google (GOOGL): {weights.get('GOOGL', 0):.1%}

        ---
        Based on the user's goal, provide a report in the following format:
        **Quick Summary:** (A 1-2 sentence overview of the result.)
        **Key Metrics Explained:** (Use bullet points to simply explain what the Cumulative Return, Max Drawdown, and Sharpe Ratio mean in this context.)
        **Final Takeaway:** (A 1-2 sentence concluding thought.)
        ---
    """)

    if user_goal == "Long-Term Growth":
        goal_instructions = "The user's goal is **Long-Term Growth**. Emphasize safety, diversification, and consistency in your explanations."
    elif user_goal == "Mid-Term Balanced":
        goal_instructions = "The user's goal is a **Mid-Term Balanced** approach. Emphasize the balance between risk (drawdown) and reward (return) in your explanations."
    else:  # Short-Term Speculation
        goal_instructions = "The user's goal is **Short-Term Speculation**. Use a direct tone. Emphasize that the strategy is high-risk and that past performance is not a guarantee of future results."

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