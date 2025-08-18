def detect_pdf_type(text: str) -> str:
    """Detect if the PDF is UPI, BANK, or UNKNOWN based on keywords."""
    text_lower = text.lower()

    # Strong signals for UPI statements
    upi_keywords = ["upi id", "@ok", "@ybl", "@paytm", "vpa", "upi transaction", "upi ref no"]
    upi_count = sum(kw in text_lower for kw in upi_keywords)

    # Strong signals for BANK statements
    bank_keywords = [
        "account number", "statement period", "ifsc", "branch", 
        "credit", "debit", "withdrawal", "deposit", "balance"
    ]
    bank_count = sum(kw in text_lower for kw in bank_keywords)

    # Classification logic
    if upi_count >= 2 and bank_count < 2:
        return "UPI"
    elif bank_count >= 2:
        return "BANK"
    else:
        return "UNKNOWN"
