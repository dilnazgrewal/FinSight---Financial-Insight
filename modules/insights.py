import pandas as pd
import numpy as np

def compute_financial_metrics(df: pd.DataFrame):
    """
    Computes a robust expense analysis from the transaction data.
    FIXED: Counts all transactions, not just expenses.
    """
    if df.empty:
        return {
            "Total Expense": "₹0", "Spending Period (Days)": 0,
            "Daily Average Spend": "₹0", "Total Transactions": 0,
            "Top Category Name": "N/A", "Top Category Value": "₹0"
        }

    # --- FIX: Count all transactions from the original DataFrame ---
    total_transactions = len(df)

    # Ensure date column is in the correct format
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    expense_df = df[df['Debit'] > 0].dropna(subset=['Date'])

    if expense_df.empty:
        # Still return the total transaction count even if there are no expenses
        return {
            "Total Expense": "₹0", "Spending Period (Days)": 0,
            "Daily Average Spend": "₹0", "Total Transactions": total_transactions,
            "Top Category Name": "N/A", "Top Category Value": "₹0"
        }

    # Core Calculations
    total_expense = expense_df['Debit'].sum()
    start_date = expense_df['Date'].min()
    end_date = expense_df['Date'].max()
    num_days = (end_date - start_date).days + 1 if pd.notna(start_date) else 1
    daily_avg_spend = total_expense / num_days
    
    # Top Category Calculation
    top_cat_series = expense_df.groupby('Category')['Debit'].sum().nlargest(1)
    if not top_cat_series.empty:
        top_category_name = top_cat_series.index[0]
        top_category_value = top_cat_series.iloc[0]
    else:
        top_category_name = "N/A"
        top_category_value = 0

    # Format for display
    return {
        "Total Expense": f"₹{total_expense:,.2f}",
        "Spending Period (Days)": f"{num_days} days",
        "Daily Average Spend": f"₹{daily_avg_spend:,.2f}",
        "Total Transactions": total_transactions,
        "Top Category Name": top_category_name,
        "Top Category Value": f"₹{top_category_value:,.2f}"
    }