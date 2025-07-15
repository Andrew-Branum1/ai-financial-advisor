import os
import textwrap
import google.generativeai as genai
class FinancialAdvisor:
    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash") 

    def _build_prompt(self, user_profile: dict, portfolio_allocation: dict) -> str:
        user_context = f"""
            **User Profile:**
            - Age: {user_profile.get('age', 'N/A')}
            - Initial Investment: {user_profile.get('investment_amount', 'N/A')}
            - Stated Time Horizon: {user_profile.get('time_horizon', 'N/A')}
            - Stated Risk Tolerance: {user_profile.get('risk_tolerance', 'N/A')}
        """

        if not portfolio_allocation:
            portfolio_lines = ["- 100% Cash (No investment recommended at this time)."]
        else:
            portfolio_lines = [f"- {ticker}: {weight}" for ticker, weight in portfolio_allocation.items()]
        
        portfolio_context = "**AI Recommended Portfolio Allocation:**\n" + "\n".join(portfolio_lines)

        prompt = f"""
            You are an expert AI financial advisor named Gemini. Your tone should be clear, educational, and reassuring. Avoid overly complex jargon. Your audience is a beginner investor.

            Here is the user's profile and the portfolio recommended by our quantitative model:

            {textwrap.dedent(user_context)}
            ---
            {portfolio_context}
            ---
            **Your Task:**
            Based on all the information above, provide a brief, easy-to-understand report. Structure your response in the following format, using markdown for formatting:

            **1. Quick Summary:**
            (A 1-2 sentence overview of the strategy and why it fits the user's profile.)

            **2. Portfolio Rationale:**
            (In a few bullet points, explain *why* this mix of assets makes sense for the user. For example, if it's aggressive, explain that it's geared towards growth. If it's conservative, explain that it prioritizes stability.)

            **3. Important Considerations:**
            (Provide 1-2 brief, important reminders for any investor, such as the nature of market risk and the importance of diversification.)
        """
        return textwrap.dedent(prompt)

    def generate_advice(self, user_profile: dict, portfolio_allocation: dict) -> str:
        try:
            prompt = self._build_prompt(user_profile, portfolio_allocation)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return "There was an issue generating the financial rationale. Please try again later."

financial_advisor = FinancialAdvisor()