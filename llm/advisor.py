import os
import google.generativeai as genai
class FinancialAdvisor:
    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash") 

    def _build_prompt(self, user_profile: dict, portfolio_allocation: dict) -> str:
        
        user_context = "User Profile:\n"
        user_context += f"Age: {user_profile.get('age', 'N/A')}\n"
        user_context += f"Initial Investment: {user_profile.get('investment_amount', 'N/A')}\n"
        user_context += f"Stated Time Horizon: {user_profile.get('time_horizon', 'N/A')}\n"
        user_context += f"Stated Risk Tolerance: {user_profile.get('risk_tolerance', 'N/A')}"

        portfolio_lines = []
        for ticker, weight in portfolio_allocation.items():
            portfolio_lines.append(f"- {ticker}: {weight}")
        portfolio_context = "AI Recommended Portfolio Allocation:\n" + "\n".join(portfolio_lines)

        prompt = (
            "You are an expert AI financial advisor named Gemini. Your tone should be clear, educational, and reassuring. Avoid overly complex jargon. Your audience is a beginner investor.\n\n"
            "Here is the user's profile and the portfolio recommended by our quantitative model:\n\n"
            f"{user_context}\n"
            "---\n"
            f"{portfolio_context}\n"
            "---\n"
            "Your Task:\n"
            "Based on all the information above, provide a brief, easy-to-understand report. Structure your response in the following format:\n\n"
            "1. Quick Summary:\n"
            "(A 1-2 sentence overview of the strategy and why it fits the user's profile.)\n\n"
            "2. Portfolio Rationale:\n"
            "(In a few bullet points, explain why this mix of assets makes sense for the user. For example, if it's aggressive, explain that it's geared towards growth. If it's conservative, explain that it prioritizes stability.)\n\n"
            "3. Important Considerations:\n"
            "(Provide 1-2 brief, important reminders for any investor, such as the nature of market risk and the importance of diversification.)"
        )
        
        return prompt

    def generate_advice(self, user_profile: dict, portfolio_allocation: dict) -> str:
        try:
            prompt = self._build_prompt(user_profile, portfolio_allocation)
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"LLM Error: {e}")
            return "LLM Error"

financial_advisor = FinancialAdvisor()