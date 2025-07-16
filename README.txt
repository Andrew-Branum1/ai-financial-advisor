AI Financial Advisor
====================

Generates financial advice using a Reinforcement Learning model for portfolio allocation and Google Gemini's Large Language Model (gemini-2.0-flash) for the explanation.


Setup
-----

1. Clone & Install
   git clone https://github.com/your-username/ai-financial-advisor.git
   cd ai-financial-advisor
   pip install -r requirements.txt

2. API Key
   Set the GEMINI_API_KEY environment variable.
   export GEMINI_API_KEY="your_api_key_here"

3. Build Database
   python src/data_manager.py

4. Train Models
   python train.py

5. Evaluate Models 
   python eval/validate_all.py
   python eval/visualize_results.py

6. Run Web App
   python app.py


Acknowledgements
----------------

This project was developed with assistance from Google's AI.
