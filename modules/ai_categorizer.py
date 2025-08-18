import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from fuzzywuzzy import fuzz

# Load environment variable
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

category_keywords = {
    "Shopping": ["amazon", "flipkart", "myntra", "ajio", "meesho"],
    "Food": ["swiggy", "zomato", "dominos", "kfc", "mcdonald", "pizza", "blinkit", "bigbasket"],
    "Fuel": ["petrol", "diesel", "fuel", "indianoil", "bharat petroleum", "hpcl"],
    "Bills": ["electricity", "water bill", "gas bill", "broadband", "wifi", "internet", "mobile recharge"],
    "Travel": ["ola", "uber", "irctc", "makemytrip", "airlines", "bus ticket"],
    "Entertainment": ["netflix", "prime video", "spotify", "hotstar", "sonyliv"],
}

def categorize_transaction(desc):
    for category, keywords in category_keywords.items():
        for kw in keywords:
            if fuzz.partial_ratio(desc, kw) > 80:
                return category
    return "Miscellaneous"

def categorize_transactions_batch(transactions):
    """
    Takes a list of transaction descriptions and returns a list of dicts
    with Category and Classification.
    """
    # Clean transactions: ensure all are strings, strip extra spaces
    transactions = [str(t).strip() if t else "" for t in transactions]

    # If no valid transactions, return empty categories
    if not any(transactions):
        return [{"Category": "Uncategorized", "Classification": None} for _ in transactions]

    prompt = f"""
You are a financial transaction categorizer.
For each transaction description below, respond with a JSON array.
Each element should have:
- "Category": One of [Food & Dining, Groceries, Transport, Shopping, Subscriptions, Utilities, Rent, Entertainment, Medical, Miscellaneous]
- "Classification": "Need" or "Want"

Example Output:
[
    {{"Category": "Food & Dining", "Classification": "Want"}},
    {{"Category": "Transport", "Classification": "Need"}}
]

Transactions:
{chr(10).join([f"- {t}" for t in transactions])}
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Extract text safely
        ai_text = response.text.strip() if hasattr(response, "text") else ""
        
        # Try to locate JSON inside response
        if "```" in ai_text:
            ai_text = ai_text.split("```")[1].replace("json", "").strip()

        results = json.loads(ai_text)

        # Validate output structure
        if isinstance(results, list) and all(isinstance(x, dict) for x in results):
            # Ensure output length matches input length
            if len(results) != len(transactions):
                results = results[:len(transactions)] + [
                    {"Category": "Uncategorized", "Classification": None}
                    for _ in range(len(transactions) - len(results))
                ]
            return results
        else:
            raise ValueError("Unexpected output format")

    except Exception as e:
        print("Error parsing AI output:", e)
        # Fallback: return "Uncategorized" for all
        return [{"Category": "Uncategorized", "Classification": None} for _ in transactions]
