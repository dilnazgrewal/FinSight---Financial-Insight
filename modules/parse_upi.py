import re
import pandas as pd

def parse_upi_text(text: str) -> pd.DataFrame:
    pattern = r"Date:\s*(.*?)\s*Time:\s*(.*?)\s*Merchant:\s*(.*?)\s*Amount \(INR\):\s*Rs\. (.*?)\s*Status:\s*(.*?)\s*Transaction ID:\s*(.*?)\s*UPI ID:\s*(.*?)\s*Note:\s*(.*?)\s*(?=Date:|$)"

    matches = re.findall(pattern, text, re.DOTALL)

    transactions = []
    for match in matches:
        transactions.append({
            "Date": match[0].strip(),
            "Time": match[1].strip(),
            "Merchant": match[2].strip(),
            "Amount": float(match[3].replace(",", "")),
            "Status": match[4].strip(),
            "Transaction ID": match[5].strip(),
            "UPI ID": match[6].strip(),
            "Note": match[7].strip()
        })

    df = pd.DataFrame(transactions)

    # --- Ensure Description Column Exists and is Filled ---
    if "Description" not in df.columns or df["Description"].isna().all() or (df["Description"].str.strip() == "").all():
        if "Merchant" in df.columns and df["Merchant"].notna().any():
            df["Description"] = df["Merchant"]
        elif "Note" in df.columns and df["Note"].notna().any():
            df["Description"] = df["Note"]
        else:
            df["Description"] = ""

    return df
