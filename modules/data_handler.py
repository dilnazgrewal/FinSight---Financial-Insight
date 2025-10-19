import pandas as pd

def clean_transactions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns, parse dates, convert numeric values and remove empty records."""
    if df is None or df.empty:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit', 'Credit', 'Category', 'Classification', 'Source'])

    df = df.copy()
    # Ensure columns
    for col in ['Date', 'Description', 'Debit', 'Credit', 'Category', 'Classification', 'Source']:
        if col not in df.columns:
            df[col] = '' if col in ['Date', 'Description', 'Category', 'Classification', 'Source'] else 0.0

    # Dates
    try:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Numeric
    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0.0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0.0)

    # Fill descriptions
    df['Description'] = df['Description'].fillna('').astype(str).str.strip()

    # Drop records without a date and without any money movement
    df = df[ (df['Date'].notna()) & ((df['Debit'] != 0.0) | (df['Credit'] != 0.0)) ]

    df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    return df
