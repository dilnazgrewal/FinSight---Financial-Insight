import pandas as pd
import json
import time 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# In modules/chatbot_agent.py

def _create_data_summary(df: pd.DataFrame) -> str:
    """
    IMPROVED: Creates a concise, text-based summary with raw numbers for the AI prompt.
    """
    if df.empty:
        return "No data available."

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    expense_df = df[df['Debit'] > 0].dropna(subset=['Date'])
    if expense_df.empty:
        return "No expense data available."

    total_spent = expense_df['Debit'].sum()
    start_date = expense_df['Date'].min().strftime('%Y-%m-%d')
    end_date = expense_df['Date'].max().strftime('%Y-%m-%d')
    
    # Send raw numbers, not formatted strings
    category_summary = expense_df.groupby('Category')['Debit'].sum().round(2).to_dict()
    need_want_summary = expense_df.groupby('Classification')['Debit'].sum().round(2).to_dict()
    top_5_expenses = expense_df.nlargest(5, 'Debit')[['Date', 'Description', 'Debit']].round(2).to_dict('records')
    
    for expense in top_5_expenses:
        expense['Date'] = expense['Date'].strftime('%Y-%m-%d')

    summary = {
        "period_start_date": start_date,
        "period_end_date": end_date,
        "total_spent": round(total_spent, 2), # Use round()
        "spending_by_category": category_summary,
        "spending_by_classification": need_want_summary,
        "top_5_largest_transactions": top_5_expenses
    }
    return json.dumps(summary, indent=2)

def answer_query(df: pd.DataFrame, query: str, chat_history: list):
    """
    Answers a user's query using a LangChain chain with Gemini.
    Yields the response in chunks for a typewriter effect.
    """
    data_summary = _create_data_summary(df)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                                 convert_system_message_to_human=True)

    # UPDATED: A much more sophisticated prompt for a smarter agent
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are 'Balance Bot', a friendly and insightful financial assistant. Your personality is encouraging, clear, and helpful.

        **Your Task:**
        Analyze the provided DATA SUMMARY to answer the USER'S QUESTION. Your main goal is to provide insightful analysis, not just basic answers.

        **Response Guidelines:**
        1.  **Be a Proactive Analyst:** Anticipate the user's next logical question. For example, if they ask for their total spending, you should also mention their top 2-3 spending categories in the same answer.
        2.  **Provide Comprehensive Summaries:** If the user asks a broad question (like "How did I do?" or "Summarize my spending"), give a multi-part answer covering: (a) Total spend, (b) The top spending categories with amounts, and (c) The Need vs. Want split.
        3.  **Engage Intelligently:** After providing a comprehensive answer, ask a relevant, open-ended question to guide the conversation. For very specific, simple questions (e.g., "What was the date of my biggest expense?"), you can answer directly without a follow-up if it feels more natural.
        4.  **Formatting:** Always format monetary values with the Indian Rupee symbol (₹) and commas (e.g., ₹1,234.56).
        5.  **Data Grounding:** Base all answers strictly on the DATA SUMMARY. If you cannot answer, politely state that you don't have enough information.

        ---
        DATA SUMMARY:
        {data_summary}
        ---
        CHAT HISTORY:
        {chat_history}
        ---
        USER'S QUESTION:
        "{query}"
        ---

        Your Proactive & Insightful Answer:
        """
    )

    chain = prompt_template | llm | StrOutputParser()

    try:
        for chunk in chain.stream({
            "data_summary": data_summary,
            "chat_history": str(chat_history),
            "query": query
        }):
            time.sleep(0.01) 
            yield chunk
            
    except Exception as e:
        yield f"Sorry, I encountered an error: {e}"