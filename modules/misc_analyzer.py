import pandas as pd
import re
import streamlit as st

def refine_miscellaneous_transactions(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Refines transactions categorized as 'Miscellaneous' or 'Other'.
    This function is generally safe and does not need changes.
    """

    if df.empty or "Description" not in df.columns:
        return df

    df = df.copy()
    if "Reasoning" not in df.columns:
        df["Reasoning"] = ""

    misc_df = df[df["Category"].isin(["Miscellaneous", "Other"])].copy()
    if misc_df.empty:
        return df

    recurring_counts = misc_df["Description"].value_counts()
    recurring_payees = recurring_counts[recurring_counts >= 3].index.tolist()

    rent_keywords = ["rent", "pg", "flat", "room", "landlord", "tenant"]

    for idx, row in misc_df.iterrows():
        desc = str(row["Description"]).lower().strip()
        new_cat = row["Category"]
        reason = ""

        # Simple rule: If it's a recurring payment to the same description and doesn't contain other keywords,
        # it's likely a personal transfer.
        if row["Description"] in recurring_payees and not any(kw in desc for kw in rent_keywords):
            new_cat, reason = "Bank & UPI Transfers", "Recurring miscellaneous transaction"
        
        if reason:
            df.at[idx, "Category"] = new_cat
            df.at[idx, "Reasoning"] = reason
            
    return df

def refine_bank_transfers(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    REVISED: Safely refines 'Bank & UPI Transfers' only when there is high confidence.
    Removes aggressive amount-based guessing.
    """

    if df.empty:
        return df

    desc_col = next((col for col in ["RawLine", "Description"] if col in df.columns), None)
    if not desc_col:
        return df

    df = df.copy()
    if "Reasoning" not in df.columns:
        df["Reasoning"] = ""

    bank_df = df[df["Category"] == "Bank & UPI Transfers"].copy()
    if bank_df.empty:
        return df

    refined_count = 0
    for idx, row in bank_df.iterrows():
        desc = str(row[desc_col]).lower()
        credit = row.get("Credit", 0)
        
        new_cat, reason = row["Category"], ""

        # --- High-Confidence Keyword Checks ---

        # Case 1: Bank interest (safe)
        if "interest" in desc and credit > 0:
            new_cat, reason = "Interest/Dividends", "Bank interest credited"

        # Case 2: Salary or payroll (safe)
        elif any(word in desc for word in ["salary", "payroll"]):
            new_cat, reason = "Income", "Salary credit detected"

        # Case 3: Rent-related (safe)
        elif "rent" in desc:
            new_cat, reason = "Rent", "Rent-related transaction"

        # Case 4: EMI or loan repayments (safe)
        elif "emi" in desc or "loan" in desc:
            new_cat, reason = "Loans", "EMI or loan repayment detected"
            
        # --- Aggressive Rules REMOVED ---
        # REMOVED: Case for 'Small UPI debit' - This assumption was incorrect.
        # REMOVED: Case for 'Large outgoing transfer' - This assumption was also incorrect.

        # --- Confirmation Logic ---
        # If a payment is recurring, we add a note but KEEP the correct category.
        elif row[desc_col] in bank_df[desc_col].value_counts()[bank_df[desc_col].value_counts() >= 3].index:
            reason = "Recurring peer transfer"

        # --- Update DataFrame ---
        if new_cat != row["Category"] or reason:
            if new_cat != row["Category"]:
                 refined_count += 1
            df.at[idx, "Category"] = new_cat
            df.at[idx, "Reasoning"] = reason
            
    if debug:
        st.success(f"🏦 Safely refined {refined_count} of {len(bank_df)} Bank/UPI transfers.")

    return df