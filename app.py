import streamlit as st
import pandas as pd
import time
import html
import speech_recognition as sr
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import fitz
import re
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from streamlit_mic_recorder import speech_to_text
from modules.chatbot_agent import answer_query
from modules.parsers import parse_pdf
from modules.detect_format import detect_pdf_type
from modules.data_handler import clean_transactions_dataframe
from modules.charts import (
    plot_expense_by_category, plot_need_vs_want,
    plot_top_expenses, plot_monthly_trends
)
from modules.misc_analyzer import refine_miscellaneous_transactions
from modules.insights import compute_financial_metrics
from modules.ai_insights import generate_financial_summary
from modules.report_generator import create_report
from modules.ai_categorizer import categorize_transactions_with_ai
from modules.misc_analyzer import refine_miscellaneous_transactions, refine_bank_transfers

st.set_page_config(page_title="FinSight", layout="wide")
st.title("💸 FinSight- Financial Insight")
st.write("Understand. Analyze. Optimize your spending.")

uploaded_files = st.file_uploader("Upload Bank or UPI PDF statements", type=["pdf"], accept_multiple_files=True)

def categorize_transaction(Description: str) -> str:
    """
    Categorizes a transaction using a hierarchical, multi-layered logic to ensure accuracy.
    """
    desc_lower = str(Description or "").lower().strip()

    # Tier 1: High-Confidence Brands & Services (Unambiguous)
    HIGH_CONFIDENCE = {
        "Food & Dining": ["swiggy", "zomato", "dominos", "kfc", "mcdonald's", "pizzahut"],
        "Groceries": ["jiomart", "blinkit", "zepto", "bigbasket"],
        "Transport": ["uber", "ola", "rapido", "metro"],
        "Shopping": ["amazon", "flipkart", "myntra", "ajio", "meesho"],
        "Entertainment": ["netflix", "spotify", "hotstar", "prime video", "bookmyshow", "pvr", "inox"],
        "Investments": ["zerodha", "groww", "upstox", "wintwealth"],
        "Utilities": ["vodafone", "airtel", "jio", "recharge"]
    }

    # Tier 2: High-Confidence Business Types (Strong Indicators)
    BUSINESS_TYPES = {
        "Groceries": ["grocery", "karyana", "supermarket", "hypermarket", "alu shop"],
        "Health": ["pharmacy", "drug store", "medicose", "medical", "medicals", "hospital", "clinic", "chemist"],
        "Food & Dining": ["restaurant", "cafe", "sweets", "confectionery", "bakery", "eatery", "tiffin", "tiffin center", "stall", "hotel"],
        "Shopping": ["fashions", "handloom", "emporium", "gift", "variety store", "book store", "puja bhandar", "samagri"],
        "Transport": ["petrol", "fuel", "h p center", "filling station"],
        "Education": ["school", "college", "tuition", "udemy", "coursera"],
        "Utilities": ["electrical", "net for you", "communication"],
        "Rent": ["rent"]
    }

    # --- Categorization Logic ---

    # 1. Check High-Confidence Brands first for a quick and accurate match.
    for category, keywords in HIGH_CONFIDENCE.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category

    # 2. Check for specific business types.
    for category, keywords in BUSINESS_TYPES.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
            
    # 3. Handle Generic Peer-to-Peer or Unclear Business Payments.
    generic_payment_pattern = r'^(gpay transaction:|paid to)'
    if re.search(generic_payment_pattern, desc_lower):
        all_keywords = [kw for sublist in list(HIGH_CONFIDENCE.values()) + list(BUSINESS_TYPES.values()) for kw in sublist]
        if not any(keyword in desc_lower for keyword in all_keywords):
            return "Bank & UPI Transfers"

    # 4. Fallback for generic bank terms if no other category fits.
    BANK_KEYWORDS = ["upi", "imps", "neft", "rtgs", "atm", "withdrawal", "deposit", "bank charge"]
    if any(word in desc_lower for word in BANK_KEYWORDS):
        return "Bank & UPI Transfers"

    # 5. If no rules match after all checks, label it for the AI.
    return "Other"

def classify_need_or_want(category: str) -> str:
    """Classifies a category as a 'Need', 'Want', or 'Other'."""
    needs = [
        "Food & Dining", "Groceries", "Transport", "Utilities", "Rent",
        "Medical", "Education", "Health", "Loans", "Bank Charges"
    ]
    wants = [
        "Shopping", "Entertainment", "Subscriptions"
    ]
    if category in needs:
        return "Need"
    if category in wants:
        return "Want"
    # Everything else (like Bank & UPI Transfers, Investments, Interest) is 'Other'.
    return "Other"

if 'transactions_df' not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()

if uploaded_files:
    if st.button("Process uploaded files"):
        all_dfs = []
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    raw_bytes = uploaded_file.read()
                    doc = fitz.open(stream=raw_bytes, filetype="pdf")
                    text = "".join(page.get_text() for page in doc)
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    pdf_type = detect_pdf_type(text)
                    df_parsed, detected_type, full_text = parse_pdf(pdf_type, uploaded_file, debug=False)

                    if df_parsed is None or df_parsed.empty:
                        st.warning(f"No transactions found in {uploaded_file.name} (detected type: {pdf_type}).")
                    else:
                        df_parsed['Source'] = uploaded_file.name
                        all_dfs.append(df_parsed)
                        st.success(f"Parsed {len(df_parsed)} transactions from {uploaded_file.name} (detected {detected_type}).")
                except Exception as e:
                    st.error(f"Failed to parse {uploaded_file.name}: {e}")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            df = clean_transactions_dataframe(combined)

            # Local categorization first
            st.info("⚙️ Applying local keyword categorization...")
            text_col = "RawLine" if "RawLine" in df.columns and df["RawLine"].notna().any() else "Description"
            if text_col not in df.columns:
                st.warning("⚠️ No suitable text column found for keyword categorization.")
            else:
                df["Category"] = df[text_col].astype(str).apply(categorize_transaction)
            df["Classification"] = df["Category"].apply(classify_need_or_want)

            # AI for only the ambiguous rows
            ambiguous_mask = df["Category"].isin(["Other", "Miscellaneous"]) | df["Category"].isnull()
            num_ambiguous = ambiguous_mask.sum()
            if num_ambiguous > 0:
                st.info(f"🤖 Sending {num_ambiguous} ambiguous rows to Gemini for AI categorization...")
                need_ai = df.loc[ambiguous_mask].copy()  # keep original indices
                ai_result = categorize_transactions_with_ai(need_ai, debug=False)
                # Update original df using index alignment
                df.loc[ai_result.index, ["Category", "Classification"]] = ai_result.loc[:, ["Category", "Classification"]].values
                st.success("✅ AI categorization completed for ambiguous rows.")
            else:
                st.success("✅ All transactions categorized locally (no AI calls).")

            df = refine_miscellaneous_transactions(df, debug=True)
            df = refine_bank_transfers(df, debug=True)
            st.session_state.transactions_df = df
            st.success("🎉 All files processed and categorized!")
            st.rerun()

 
# Main UI when data exists
if not st.session_state.transactions_df.empty:
    df = st.session_state.transactions_df.copy()

    # Manual entry
    st.markdown("---")
    st.subheader("✍️ Add a Manual Transaction")

    with st.form("manual_txn", clear_on_submit=True):
        date = st.date_input("Date")
        amt = st.number_input("Amount (expense)", min_value=0.0, value=0.0, step=1.0)
        category_options = [
        "Food & Dining", "Groceries", "Transport", "Shopping", "Subscriptions",
        "Utilities", "Rent", "Entertainment", "Medical", "Education",
        "Bank & UPI Transfers", "Investments",
        "Interest/Dividends", "Miscellaneous"
    ]
        cat= st.selectbox("Category", category_options)
        cls = st.selectbox("Classification", ["Need", "Want", "Other"])
        if st.form_submit_button("Add Transaction"):
            new_row = {
                "Date": pd.to_datetime(date),
                "Debit": float(amt),
                "Credit": 0.0,
                "Category": cat,
                "Classification": cls,
                "Source": "Manual Entry"
            }
            st.session_state.transactions_df = pd.concat(
                [st.session_state.transactions_df, pd.DataFrame([new_row])],
                ignore_index=True
            )
            st.success("✅ Manual transaction added.")
            st.rerun()

#---------------------Transactions table--------------------------
    st.markdown("---")
    st.subheader("📋 Transactions Table")
    st.dataframe(df)

#---------------------Snapshot------------------------------------
    st.markdown("---")
    st.subheader("📈 Expense Analysis Snapshot")
    metrics = compute_financial_metrics(df)
    cols = st.columns(5)

    cols[0].metric(
        "Total Spent",
        metrics["Total Expense"]
    )
    cols[1].metric(
        "Spending Period",
        metrics["Spending Period (Days)"]
    )
    cols[2].metric(
        "Daily Average Spend",
        metrics["Daily Average Spend"]
    )
    cols[3].metric(
        "No. of Transactions",
        metrics["Total Transactions"]
    )
    # This metric dynamically shows the user's #1 spending category
    cols[4].metric(
        f"Top Category: {metrics['Top Category Name']}",
        metrics["Top Category Value"]
    )

    # Charts
    st.markdown("---")
    st.subheader("📊 Visual Analysis")
    left, right = st.columns(2)
    with left:
        fig1 = plot_expense_by_category(df)
        if fig1:
            st.plotly_chart(fig1, use_container_width=True)
        fig2 = plot_top_expenses(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    with right:
        fig3 = plot_need_vs_want(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        fig4 = plot_monthly_trends(df)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)

    # Downloads
    st.markdown("---")
    st.subheader("⬇️ Download Data")

    # 1. Download CSV Button (remains the same)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Export All Transactions (CSV)", 
        csv, 
        file_name="transactions.csv", 
        mime="text/csv"
    )

# 2. Download PDF Report Button (updated logic)
if st.button("Generate PDF Report"):
    with st.spinner("🤖 AI Analyst is writing your summary..."):
        raw_metrics = compute_financial_metrics(df) 
        metrics_for_ai = {
            "Total Expense": df['Debit'].sum(),
            "Total Transactions": len(df)
        }
        ai_summary = generate_financial_summary(df, metrics_for_ai)
        st.session_state.ai_summary = ai_summary

    with st.spinner("🎨 Assembling your report with charts..."):
        pdf_bytes = create_report(df, raw_metrics, st.session_state.ai_summary)
        st.session_state.pdf_bytes = pdf_bytes
        st.success("✅ Your PDF report is ready!")

if 'pdf_bytes' in st.session_state:
    st.download_button(
        "Download PDF Report", 
        st.session_state.pdf_bytes, 
        file_name="financial_report.pdf", 
        mime="application/pdf"
    )


#------------------Balance Bot-------------------------
load_dotenv()
def answer_query(df, query):
    """
    Uses Gemini API to analyze user's transactions and provide concise financial insights.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        csv_preview = df.head(10).to_string(index=False)

        prompt = f"""
        You are a concise and analytical AI financial assistant.
        The user asked: "{query}"

        Here are their recent transactions (first 10 rows):
        {csv_preview}

        Provide a short, structured response including:
        - Total spending (if inferable)
        - Top 3 spending categories with approximate amounts
        - A brief Needs vs Wants summary
        - 1-2 observations for saving opportunities

        Use bullet points or short formatted text.
        Avoid long paragraphs — prioritize clarity and brevity.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error generating response: {e}"


# ------------------ Session State Init ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "transactions_df" not in st.session_state:
    st.session_state.transactions_df = pd.DataFrame()

# ------------------ Voice Input Function ------------------
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please speak clearly")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
            st.info("🔍 Processing your voice input...")
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.WaitTimeoutError:
            st.warning("⚠️ Listening timed out. Please try again.")
        except sr.UnknownValueError:
            st.warning("⚠️ Sorry, I couldn’t understand your voice.")
        except Exception as e:
            st.error(f"Error: {e}")
    return None

# ------------------ Chat Display ------------------
st.markdown("## 💬 Chat with Balance Bot")

# Display message history
for m in st.session_state.messages:
    role = m.get("role", "")
    raw_content = m.get("content", "")
    content = html.escape(str(raw_content)).replace("\n", "<br>")

    if role == "user":
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #BBDEFB 0%, #E3F2FD 100%);
                padding: 12px 14px;
                border-radius: 15px;
                margin: 8px 0;
                text-align: right;
                color: #0D47A1;
                font-weight: 500;
                font-size: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08);
                width: fit-content;
                max-width: 80%;
                margin-left: auto;
            ">{content}</div>
            """,
            unsafe_allow_html=True,
        )

    elif role == "assistant":
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1565C0 0%, #1E88E5 100%);
                padding: 12px 14px;
                border-radius: 15px;
                margin: 8px 0;
                text-align: left;
                color: white;
                font-weight: 500;
                font-size: 15px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.12);
                width: fit-content;
                max-width: 80%;
                margin-right: auto;
            ">{content}</div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ------------------ Input Section ------------------
col1, col2 = st.columns([5, 1])

with col1:
    prompt = st.chat_input("Type your question or use the mic...")

with col2:
    if st.button("🎤 Speak"):
        spoken_text = recognize_speech()
        if spoken_text:
            prompt = spoken_text 

# ------------------ Handle User Query ------------------
if prompt:
    final_prompt = prompt.strip()

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": final_prompt,
        "ts": datetime.utcnow().isoformat()
    })

    # Generate assistant response
    df = st.session_state.get("transactions_df")
    if df is not None and not df.empty:
        try:
            assistant_reply = answer_query(df, final_prompt)
        except Exception as e:
            assistant_reply = f"Error while answering the query: {e}"
    else:
        assistant_reply = "Please upload and process a file first."

    # --- Animated intro bubble ---
    summary = "Here's your spending summary and insights 💰👇"
    message_placeholder = st.empty()
    typed_text = ""
    for char in summary:
        typed_text += char
        message_placeholder.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #BBDEFB 0%, #E3F2FD 100%);
                padding: 12px;
                border-radius: 12px;
                margin: 8px 0;
                text-align: left;
                color: #0D47A1;
                font-weight: 600;
                font-size: 16px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            ">{typed_text}</div>
            """,
            unsafe_allow_html=True,
        )
        time.sleep(0.006)  # faster animation speed

    # --- Detailed report below in collapsible expander ---
    with st.expander("📊 View Detailed Breakdown", expanded=True):
        st.markdown(
            f"""
            <div style="
                background-color: #E3F2FD;
                padding: 15px;
                border-radius: 10px;
                margin-top: 8px;
                text-align: left;
                color: #0D47A1;
                font-size: 15px;
                line-height: 1.6;
            ">{assistant_reply}</div>
            """,
            unsafe_allow_html=True,
        )

    # Save assistant message in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply,
        "ts": datetime.utcnow().isoformat()
    })

    st.rerun()