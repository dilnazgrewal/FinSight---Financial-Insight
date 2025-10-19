import os
import json
import time
import hashlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file")
else:
    genai.configure(api_key=api_key)

CACHE_FILE = "ai_cache.json"

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

AI_CACHE = _load_cache()


def preprocess_description(desc: str) -> str:
    """Clean and normalize the description before sending to AI."""
    desc = str(desc or "").lower().strip()
    for phrase in [
        "paid to", "received from", "transaction id", "utr no.",
        "paid by", "credited to", "debited from", "via upi"
    ]:
        desc = desc.replace(phrase, "")
    return desc.strip()


def _categorize_with_gemini(descriptions: list, debug=False, retries=2):
    """Send a batch of descriptions to Gemini for categorization."""
    if not descriptions:
        return []

    # Remove duplicates by cache lookup
    results = []
    to_query = []
    for desc in descriptions:
        key = hashlib.md5(desc.encode()).hexdigest()
        if key in AI_CACHE:
            results.append(AI_CACHE[key])
        else:
            results.append(None)
            to_query.append(desc)

    if not to_query:
        return results  # all from cache

    prompt = f"""
    You are a finance categorization assistant.
    Assign each transaction a single best-fitting "Category" 
    from this list:
    [Food & Dining, Groceries, Transport, Shopping, Subscriptions, Utilities,
    Rent, Entertainment, Medical, Education, Bank & UPI Transfers, Investments,
    Bank Charges, Interest/Dividends, Miscellaneous].

    Respond ONLY with a JSON array of objects, where each object has a single "Category" key.
    For example: [{{"Category": "Food & Dining"}}, {{"Category": "Shopping"}}]

    Transactions: {json.dumps(to_query, ensure_ascii=False)}
    """

    model = genai.GenerativeModel("gemini-2.5-flash-lite")

    for attempt in range(retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                },
            )

            if debug:
                st.write("🔍 Raw AI response:", response.text)

            batch_results = json.loads(response.text)

            if isinstance(batch_results, list) and len(batch_results) == len(to_query):
                # Save new results to cache
                for desc, res in zip(to_query, batch_results):
                    key = hashlib.md5(desc.encode()).hexdigest()
                    AI_CACHE[key] = res
                _save_cache(AI_CACHE)
                return batch_results

        except Exception as e:
            if debug:
                st.warning(f"Attempt {attempt + 1} failed: {e}")
            if "quota" in str(e).lower():
                st.error("🚨 Gemini quota reached. Using defaults for this batch.")
                return [{"Category": "Miscellaneous", "Classification": "Other"} for _ in to_query]
            time.sleep(2)

    return [{"Category": "Miscellaneous", "Classification": "Other"} for _ in to_query]


def categorize_transactions_with_ai(df: pd.DataFrame, debug=False):
    """Categorize only rows that are unclassified or marked as Other/Miscellaneous."""
    if df.empty or "Description" not in df.columns:
        return df

    # Clean text
    df["Cleaned_Description"] = df["Description"].astype(str).apply(preprocess_description)

    # Find rows that need AI help
    to_fix = df[
        df["Category"].isna() |
        df["Category"].isin(["Other", "Miscellaneous"])
    ].copy()

    if to_fix.empty:
        if debug:
            st.info("✅ All transactions already categorized — skipping Gemini.")
        return df

    descriptions = to_fix["Cleaned_Description"].tolist()

    BATCH_SIZE = 15
    progress = st.progress(0)

    for i in range(0, len(descriptions), BATCH_SIZE):
        batch = descriptions[i:i + BATCH_SIZE]
        indices = list(to_fix.index[i:i + len(batch)])

        batch_results = _categorize_with_gemini(batch, debug=debug)

        if not isinstance(batch_results, list):
            batch_results = []

        while len(batch_results) < len(batch):
            batch_results.append({"Category": "Miscellaneous", "Classification": "Other"})

        for idx, result in zip(indices, batch_results):
            df.at[idx, "Category"] = result.get("Category", "Miscellaneous")
            df.at[idx, "Classification"] = result.get("Classification", "Other")

        # Add the st.write debug line inside the loop
        for idx, result in zip(indices, batch_results):
            st.write(f"🐛 DEBUG: Index={idx}, Result='{result}', Type={type(result)}") 
        
            if isinstance(result, dict):
                # This handles the case where the result is a dictionary as expected
                df.at[idx, "Category"] = result.get("Category", "Miscellaneous")
                df.at[idx, "Classification"] = result.get("Classification", "Other")
            elif isinstance(result, str):
                # This handles the case where the result is just a string
                df.at[idx, "Category"] = result
                df.at[idx, "Classification"] = "Other" # Assign a default classification

        progress.progress((i + len(batch)) / len(descriptions))

    return df
