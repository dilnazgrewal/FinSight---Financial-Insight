def detect_pdf_type(text: str) -> str:
    """Detect PDF type from text. Returns SBI, PhonePe, GPay, Paytm, UPI (generic), Bank (generic), or UNKNOWN"""
    if not text:
        return "UNKNOWN"
    t = text.lower()
    if "indian bank" in t:
        return "Indian Bank"
    if "state bank of india" in t or "sbi" in t:
        return "SBI"
    if "phonepe" in t:
        return "PhonePe"
    if "google pay" in t or "gpay" in t:
        return "GPay"
    if "paytm" in t:
        return "Paytm"

    upi_keywords = ["upi id", "@ok", "@ybl", "@paytm", "vpa", "upi transaction", "upi"]
    bank_keywords = ["account number", "statement period", "ifsc", "branch", "account no", "account"]

    if any(kw in t for kw in upi_keywords):
        return "UPI (Generic)"
    if any(kw in t for kw in bank_keywords):
        return "Bank (Generic)"
    return "UNKNOWN"
