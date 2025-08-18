import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
from modules.detect_format import detect_pdf_type
from modules.parse_upi import parse_upi_text
from modules.parse_bank import parse_bank_text
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from modules.ai_categorizer import categorize_transactions_batch  # Batch version
import plotly.express as px
import numpy as np
from langchain.schema import SystemMessage
from modules.ai_categorizer import categorize_transaction  
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Microsoft VS Code\tesseract.exe"

# hi 
def extract_text_with_ocr(pdf_bytes):
    """Extract text from image-based PDFs using OCR."""
    images = convert_from_bytes(pdf_bytes)
    text_pages = []
    for img in images:
        text = pytesseract.image_to_string(img)
        text_pages.append(text)
    return "\n".join(text_pages)

RULE_KEYWORDS = {
    "Transport": ["fuel", "petrol", "diesel", "petrol pump", "fuel purchase", "uber", "ola", "cab", "taxi", "uber ride", "ola ride"],
    "Utilities": ["electricity", "electricity bill", "water bill", "phone recharge", "recharge", "bill payment", "gas bill"],
    "Subscriptions": ["netflix", "spotify", "youtube premium", "prime video", "google play", "play store", "amazon prime", "disney","subscriptions"],
    "Food & Dining": ["zomato", "swiggy", "mcdonald", "domino", "domino's", "pizza", "restaurant", "kfc"],
    "Groceries": ["bigbasket", "reliance fresh", "local kirana", "grocery", "dmart"],
    "Shopping": ["amazon", "flipkart", "myntra", "online shopping", "pos purchase"],
    "Entertainment": ["bookmyshow", "movie", "movie tickets"],
    "Medical": ["clinic", "pharmacy", "hospital", "medicine"],
}

def apply_rule_based_override(description, model_cat, model_class):
    desc = (str(description) if description is not None else "").lower()
    for cat, kwlist in RULE_KEYWORDS.items():
        for kw in kwlist:
            if kw in desc:
                classification = "Need" if cat in ["Groceries", "Utilities", "Transport", "Rent", "Medical"] else "Want"
                return {"Category": cat, "Classification": classification}
    if isinstance(model_cat, str) and model_cat.strip():
        return {"Category": model_cat, "Classification": (model_class if model_class and str(model_class).strip() else "Want")}
    return {"Category": "Miscellaneous", "Classification": "Want"}


st.set_page_config(page_title="Project Balance", layout="wide")
st.title("📄 Project Balance - Upload All Your PDFs")

# --- Init session state ---
if "final_df" not in st.session_state:
    st.session_state.final_df = pd.DataFrame()

if "pdfs_parsed" not in st.session_state:
    st.session_state.pdfs_parsed = False

# --- PDF Upload ---
uploaded_files = st.file_uploader(
    "Upload Bank + UPI PDFs (GPay, PhonePe, Paytm, etc.)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.pdfs_parsed:
    all_dfs = []

    for file in uploaded_files:
        st.subheader(f"📂 Processing: {file.name}")

        # Read PDF normally
        file_bytes = file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)

        # OCR fallback if text is too short or missing
        if len(text.strip()) < 50:
            st.warning(f"⚠ {file.name} seems scanned. Using OCR...")
            text = extract_text_with_ocr(file_bytes)

        # --- Always parse after text is ready ---
        pdf_type = detect_pdf_type(text)
        st.info(f"Detected Type: *{pdf_type}*")

        if pdf_type == "UPI":
            df = parse_upi_text(text)
        elif pdf_type == "BANK":
            df = parse_bank_text(text)
        else:
            st.warning("❌ Unknown format. Skipping this file.")
            continue

        df["Source File"] = file.name
        all_dfs.append(df)

        st.success(f"✅ Extracted {len(df)} transactions from {file.name}")
        st.dataframe(df)

    # Combine all parsed data
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if "Source File" not in combined_df.columns:
            combined_df["Source File"] = "Unknown"


        # Ensure required columns exist
        for col in ["Description", "Category", "Classification"]:
            if col not in combined_df.columns:
                combined_df[col] = None

        # --- Ensure Credit & Debit columns ---
        if "Credit" not in combined_df.columns:
            combined_df["Credit"] = 0.0
        if "Debit" not in combined_df.columns:
            combined_df["Debit"] = 0.0

        # If only Amount column exists, classify into Credit/Debit
        if "Amount" in combined_df.columns:
            for i, row in combined_df.iterrows():
                desc = str(row.get("Description", row.get("Merchant", ""))).lower()
                amt = row.get("Amount", 0.0)

                if any(word in desc for word in ["received", "credit", "refund", "deposit"]):
                    combined_df.at[i, "Credit"] = amt
                    combined_df.at[i, "Debit"] = 0.0
                elif any(word in desc for word in ["paid", "debit", "sent", "transfer", "purchase", "bill"]):
                    combined_df.at[i, "Debit"] = amt
                    combined_df.at[i, "Credit"] = 0.0
                else:
                    combined_df.at[i, "Debit"] = amt
                    combined_df.at[i, "Credit"] = 0.0
        # Remove failed/pending/cancelled/reversed transactions
        if "Status" in combined_df.columns:
            combined_df = combined_df[~combined_df["Status"].str.lower().isin(
                ["failed", "pending", "cancelled", "reversed"]
            )]

        # Append to final_df
        st.session_state.final_df = pd.concat(
            [st.session_state.final_df, combined_df],
            ignore_index=True
        )

        st.session_state.final_df["Category"] = st.session_state.final_df["Description"].apply(categorize_transaction)

        # AI fallback for Miscellaneous
        misc_mask = st.session_state.final_df["Category"] == "Miscellaneous"
        if misc_mask.any():
            misc_descs = st.session_state.final_df.loc[misc_mask, "Description"].tolist()
            ai_results = categorize_transactions_batch(misc_descs)  # your existing AI batch categorizer

            for idx, res in zip(st.session_state.final_df.loc[misc_mask].index, ai_results):
                st.session_state.final_df.at[idx, "Category"] = res.get("Category", "Miscellaneous")
                st.session_state.final_df.at[idx, "Classification"] = res.get("Classification", "Want")

    

# ---- Batch AI Categorization ----
    def batch_categorize_with_chunks(descriptions, chunk_size=30):
        results = []
        for i in range(0, len(descriptions), chunk_size):
            chunk = descriptions[i:i+chunk_size]
            results.extend(categorize_transactions_batch(chunk))
        return results

    with st.spinner("🔍 Auto-categorizing uncategorized transactions using AI..."):
        mask = (
            st.session_state.final_df["Source File"] != "Manual Entry"
        ) & (
            st.session_state.final_df["Category"].isna() |
            (st.session_state.final_df["Category"].str.strip() == "") |
            (st.session_state.final_df["Category"].str.lower() == "uncategorized") |
            st.session_state.final_df["Classification"].isna()
        )

        if mask.any():
            if "Description" not in st.session_state.final_df.columns:
                st.session_state.final_df["Description"] = ""

            batch_data = (
                st.session_state.final_df.loc[mask, "Description"]
                .fillna("")
                .astype(str)
                .tolist()
            )

            results = batch_categorize_with_chunks(batch_data, chunk_size=30)
            mask_indexes = list(st.session_state.final_df.loc[mask].index)

            if len(results) == len(mask_indexes):
                for idx, res in zip(mask_indexes, results):
                    st.session_state.final_df.at[idx, "Category"] = res.get("Category", "Miscellaneous")
                    st.session_state.final_df.at[idx, "Classification"] = res.get("Classification", "Need")
            else:
                st.warning("⚠ Mismatch between AI results and rows — uncategorized rows kept as Miscellaneous.")

            # Fill remaining NaNs
            st.session_state.final_df["Category"] = st.session_state.final_df["Category"].fillna("Miscellaneous")
            st.session_state.final_df["Classification"] = st.session_state.final_df["Classification"].fillna("Need")


        st.session_state.final_df.to_csv("all_combined_transactions.csv", index=False)
        st.session_state.pdfs_parsed = True

# --- Show current transactions ---
if not st.session_state.final_df.empty:
    st.subheader("📊 Current Transactions")
    st.dataframe(st.session_state.final_df)

# --- Manual Entry ---
st.markdown("---")
st.subheader("➕ Add a Manual Transaction")

categories = {
    "Food & Dining": "Want",
    "Groceries": "Need",
    "Transport": "Need",
    "Shopping": "Want",
    "Subscriptions": "Want",
    "Utilities": "Need",
    "Rent": "Need",
    "Entertainment": "Want",
    "Medical": "Need",
    "Miscellaneous": "Want"
}

with st.form("manual_transaction_form", clear_on_submit=True):
    manual_date = st.date_input("Date of Transaction")
    manual_desc = st.text_input("Description (e.g., Plumber, Movie Tickets)")
    manual_amt = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
    manual_cat = st.selectbox("Category", list(categories.keys()))
    submitted = st.form_submit_button("➕ Add Transaction")

if submitted:
    manual_row = {
        "Date": manual_date.strftime("%d-%m-%Y"),
        "Time": "",
        "Merchant": manual_desc,
        "Amount": manual_amt,
        "Status": "Success",
        "Transaction ID": "MANUAL",
        "UPI ID": "N/A",
        "Note": "Manual Entry",
        "Credit": 0.0,
        "Debit": manual_amt,
        "Balance": None,
        "Description": manual_desc,
        "Category": manual_cat,
        "Classification": categories[manual_cat],
        "Source File": "Manual Entry"
    }

    manual_df = pd.DataFrame([manual_row])
    st.session_state.final_df = pd.concat(
        [manual_df, st.session_state.final_df],
        ignore_index=True
    )

    st.success("✅ Manual transaction added!")
    st.session_state.final_df.to_csv("all_combined_transactions.csv", index=False)


if not st.session_state.final_df.empty:
    st.markdown("---")
    st.subheader("📊 All Transactions (Updated)")
    st.dataframe(st.session_state.final_df)

df = st.session_state.final_df.copy()
# Ensure Date column exists
if "Date" not in df.columns:
    df["Date"] = pd.NaT  # Fill with NaT (Not a Time) if missing

df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
# Ensure Credit & Debit columns always exist
if "Credit" not in df.columns:
    df["Credit"] = 0.0
if "Debit" not in df.columns:
    df["Debit"] = 0.0

# If only Amount column exists, split into Credit/Debit
if "Amount" in df.columns and (df["Credit"].sum() == 0 and df["Debit"].sum() == 0):
    for i, row in df.iterrows():
        desc = str(row.get("Description", row.get("Merchant", ""))).lower()
        amt = row.get("Amount", 0.0)

        if any(word in desc for word in ["received", "credit", "refund", "deposit"]):
            df.at[i, "Credit"] = amt
            df.at[i, "Debit"] = 0.0
        elif any(word in desc for word in ["paid", "debit", "sent", "transfer", "purchase", "bill"]):
            df.at[i, "Debit"] = amt
            df.at[i, "Credit"] = 0.0
        else:
            df.at[i, "Debit"] = amt
            df.at[i, "Credit"] = 0.0

if "Status" in df.columns:
    df = df[df["Status"].str.lower() != "failed"]

# --- Summary Section ---
st.markdown("---")
st.header("📊 Summary Overview")

if not df.empty:
    total_expense = df["Debit"].sum() if "Debit" in df.columns else 0.0
    avg_daily_expense = df.groupby(df["Date"].dt.date)["Debit"].sum().mean()

    col1, col2 = st.columns(2)
    col1.metric("💸 Total Expenses", f"₹{total_expense:,.2f}")
    col2.metric("📆 Avg Daily Spend", f"₹{avg_daily_expense:,.2f}")

    # Top spending category
    if "Category" in df.columns and not df["Category"].isna().all():
        category_totals = df.groupby("Category")["Debit"].sum().reset_index()
        top_category = category_totals.loc[category_totals["Debit"].idxmax()]
        st.write(f"🏆 Top Spending Category:** {top_category['Category']} (₹{top_category['Debit']:,.2f})")

    # Most expensive transaction
    if "Description" in df.columns and not df["Debit"].isna().all():
        max_txn = df.loc[df["Debit"].idxmax()]
        st.write(f"💎 Most Expensive Transaction:** {max_txn['Description']} - ₹{max_txn['Debit']:,.2f}")

    # Needs vs Wants Ratio
    if "Classification" in df.columns:
        needs_spend = df.loc[df["Classification"] == "Need", "Debit"].sum()
        wants_spend = df.loc[df["Classification"] == "Want", "Debit"].sum()
        total_spend = needs_spend + wants_spend
        if total_spend > 0:
            needs_percent = (needs_spend / total_spend) * 100
            wants_percent = (wants_spend / total_spend) * 100
            st.write(f"⚖ Needs vs Wants:** {needs_percent:.1f}% Needs / {wants_percent:.1f}% Wants")


# --- Visual Analysis ---
if not st.session_state.final_df.empty:
    st.markdown("---")
    st.header("📊 Visual Analysis")

    # ---- Toggle for Monthly View ----
    monthly_view = st.toggle("📅 Show Monthly View Only")

    if monthly_view:
        # Let user pick month/year
        selected_month = st.selectbox(
            "Select Month",
            sorted(df["Date"].dt.to_period("M").dropna().astype(str).unique())
        )
        df = df[df["Date"].dt.to_period("M").astype(str) == selected_month]

    # Monthly Spending Trend
    monthly_spend = df.groupby(df["Date"].dt.to_period("M"))["Debit"].sum().reset_index()
    monthly_spend["Date"] = monthly_spend["Date"].astype(str)
    fig_monthly = px.line(monthly_spend, x="Date", y="Debit", markers=True,
                          title="Monthly Spending Trend",
                          labels={"Date": "Month", "Debit": "Amount (₹)"},
                          template="plotly_white")
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Category-wise Spending
    category_spend = df.groupby("Category")["Debit"].sum().reset_index()
    fig_category = px.pie(category_spend, values="Debit", names="Category",
                          hole=0.4, title="Spending by Category",
                          color_discrete_sequence=px.colors.qualitative.Set3)
    fig_category.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_category, use_container_width=True)

    # Needs vs Wants
    needs_wants = df.groupby("Classification")["Debit"].sum().reset_index()
    fig_needs = px.bar(needs_wants, x="Classification", y="Debit",
                       color="Classification",
                       title="Needs vs Wants",
                       labels={"Debit": "Amount (₹)"},
                       template="plotly_white")
    st.plotly_chart(fig_needs, use_container_width=True)

    # Top 10 Expenses
    top_expenses = df.sort_values(by="Debit", ascending=False).head(10)
    fig_top10 = px.bar(top_expenses, x="Description", y="Debit",
                       color="Category",
                       title="Top 10 Expenses",
                       labels={"Debit": "Amount (₹)", "Description": "Transaction"},
                       template="plotly_white")
    fig_top10.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_top10, use_container_width=True)

# --- AI Q&A Section (Multi-turn) ---
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time

# Load .env file
load_dotenv()

st.markdown("---")
st.header("💬 Ask Project Balance AI")

# Typewriter effect for responses
def typewriter(text, delay=0.02):
    placeholder = st.empty()
    typed = ""
    for char in text:
        typed += char
        placeholder.markdown(typed)
        time.sleep(delay)

if not st.session_state.final_df.empty:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ Google API key not found. Please set it in your .env file.")
    else:
        # Initialize conversation history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Create LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.6,
            convert_system_message_to_human=True
        )
        system_message = SystemMessage(
            content=(
                "You are Project Balance AI, a friendly personal finance assistant. "
                "You answer based on the user's transaction data and remember the conversation history. "
                "Be clear, conversational, and offer helpful insights."
    )
)

        # Create agent
        agent = create_pandas_dataframe_agent(
            llm,
            st.session_state.final_df,
            verbose=False,
            allow_dangerous_code=True,
            agent_executor_kwargs={"system_message": system_message}
        )


        # Chat input
        user_query = st.chat_input("Ask about your spending (e.g., 'Where did I spend the most this month?')")

        # Display previous conversation
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        if user_query:
            # Show user message
            st.session_state.chat_history.append(("user", user_query))
            with st.chat_message("user"):
                st.markdown(user_query)

            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        raw_response = agent.run(user_query)
                        final_response = f"Here's what I found:\n\n{raw_response}"
                        typewriter(final_response)
                        st.session_state.chat_history.append(("assistant", final_response))
                    except Exception as e:
                        st.error(f"Error: {e}")

