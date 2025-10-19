import pandas as pd
import google.generativeai as genai
import json

def generate_financial_summary(df: pd.DataFrame, metrics: dict) -> str:
    """
    Uses a generative AI to create a textual summary of the financial data.
    """
    if df.empty:
        return "No transaction data available to analyze."

    try:
        # 1. Prepare a concise summary of the data for the prompt
        cat_summary = df[df['Debit'] > 0].groupby('Category')['Debit'].sum().nlargest(5).to_dict()
        top_expenses = df[df['Debit'] > 0].nlargest(3, 'Debit')[['Description', 'Debit']].to_dict('records')
        need_vs_want = df.groupby('Classification')['Debit'].sum().to_dict()

        data_summary = {
            "overall_metrics": metrics,
            "top_spending_categories": cat_summary,
            "largest_individual_expenses": top_expenses,
            "need_vs_want_split": need_vs_want
        }

        # 2. Engineer the prompt
        prompt = f"""
        You are a friendly and helpful financial analyst. Your task is to provide a concise, easy-to-understand summary of a user's spending habits based on the following data.

        **Data Summary:**
        {json.dumps(data_summary, indent=2)}

        **Instructions:**
        1.  Start with a brief, encouraging overall summary.
        2.  Analyze the 'Top Spending Categories'. Point out the main areas of spending.
        3.  Comment on the 'Largest Individual Expenses'. Mention if any large one-time purchases stand out.
        4.  Analyze the 'Need vs. Want' split.
        5.  Conclude with one simple, actionable tip for better financial awareness based *only* on the data provided.
        6.  Use markdown for formatting. Use headers like '## Overall Summary', '## Spending Habits', etc. Do not use overly complex financial jargon. Keep the tone positive.
        """

        # 3. Call the Gemini API
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(prompt)
        
        return response.text

    except Exception as e:
        return f"An error occurred while generating the AI summary: {e}"