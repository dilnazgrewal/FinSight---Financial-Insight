# modules/parse_bank.py

import re
import pandas as pd
from typing import List, Dict, Optional

# -----------------------------
# Helpers
# -----------------------------

CURRENCY = r"(?:₹|INR|Rs\.?|£)?"
NUM = rf"{CURRENCY}\s*[-+]?\s*\d[\d,]*\.?\d*"
DATE_SBI = r"\d{2}\s+\w{3}\s+\d{4}"          # 26 Nov 2020
DATE_SLASH = r"\d{2}/\d{2}/\d{4}"            # 26/11/2020
DATE_DASH_MON = r"\d{2}-\w{3}-\d{4}"         # 26-Nov-2020
DATE_DASH = r"\d{2}-\d{2}-\d{4}"             # 26-11-2020

def _to_float(x: Optional[str]) -> float:
    if x is None:
        return 0.0
    s = str(x)
    s = re.sub(r"[^\d\.\-]", "", s)  # keep digits, dot, minus
    try:
        return float(s) if s else 0.0
    except:
        return 0.0

def _strip(x: str) -> str:
    return re.sub(r"\s+", " ", x or "").strip()

def _join_wrapped_lines(lines: List[str]) -> List[str]:
    """Join lines that obviously belong to the previous one (common after OCR)."""
    joined = []
    for line in lines:
        line = line.rstrip()
        if not joined:
            joined.append(line)
            continue
        # If the line doesn't start with a date and previous line didn't end with full stop,
        # it might be a continuation of description.
        if not re.match(rf"^\s*({DATE_SBI}|{DATE_SLASH}|{DATE_DASH_MON}|{DATE_DASH})\b", line) \
           and (not joined[-1].endswith((".", ":", ";"))):
            joined[-1] = joined[-1] + " " + line.strip()
        else:
            joined.append(line)
    return joined

def _clean_description(desc: str) -> str:
    """Remove UPI/IMPS/NEFT codes, numeric refs, noisy tokens."""
    desc = desc or ""
    desc = desc.replace("\n", " ")
    # remove common rails/prefixes
    desc = re.sub(r"\b(UPI|IMPS|NEFT|RTGS|ACH|ACHCr|ACHDr|NACH|ECS|POS|PG|INB|IB|CMS)\b[:/\\-]?", " ", desc, flags=re.I)
    desc = re.sub(r"\b(TRANSF(?:ER)?(?:RED)?|DEBITED|CREDITED|FROM|TO)\b", " ", desc, flags=re.I)
    # remove long numeric / alphanumeric ids
    desc = re.sub(r"[A-Z0-9]{6,}", " ", desc)
    desc = re.sub(r"\b\d{6,}\b", " ", desc)
    # collapse spaces
    desc = re.sub(r"\s+", " ", desc).strip()
    return desc

def _finalize_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["Date", "Description", "Debit", "Credit", "Balance"]:
        if col not in df.columns:
            df[col] = None

    # coerce
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    for c in ["Debit", "Credit", "Balance"]:
        df[c] = df[c].apply(_to_float)

    # clean desc
    df["Description"] = df["Description"].astype(str).apply(_clean_description)

    # drop rows that are completely empty
    df = df[(df["Date"].notna()) | (df["Debit"] != 0) | (df["Credit"] != 0) | (df["Description"].str.strip() != "")]
    df = df.reset_index(drop=True)
    return df

# -----------------------------
# SBI
# -----------------------------

def parse_sbi_bank_statement(text: str) -> pd.DataFrame:
    """
    Handles lines like:

    26 NOV 2020 TRANSFER FROM 5099076162094 - UPI/CR/...  136.00   3000.00
    25 NOV 2020 TRANSFER TO ... UPI/DR/...                 66.00    3000.00

    Also handles OCR where columns may break into multiple lines.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = _join_wrapped_lines(lines)

    rows: List[Dict] = []

    # Pattern 1: typical table row with date at start and ends with amounts
    row_re = re.compile(
        rf"^({DATE_SBI})\s+(.*?)\s+({NUM})?\s*({NUM})?\s*$", re.I
    )
    # Many SBI statements show either Debit then Balance OR Credit then Balance;
    # We’ll infer via CR/DR hints inside description, else assume the second last is amount, last is balance.
    for line in lines:
        m = row_re.match(line)
        if not m:
            continue
        date, desc, maybe_amt, maybe_bal = m.groups()
        desc = _strip(desc)

        debit = credit = 0.0
        balance = 0.0

        if maybe_bal and maybe_amt:
            balance = _to_float(maybe_bal)
            amount = _to_float(maybe_amt)

            # decide debit/credit by markers in description
            mark = desc.upper()
            if " CR" in mark or " CREDIT" in mark or "/CR/" in mark:
                credit = amount
            elif " DR" in mark or " DEBIT" in mark or "/DR/" in mark:
                debit = amount
            else:
                # if not marked, heuristic: if "TRANSFER TO"/"POS"/"BILL" → debit, else credit
                if re.search(r"\b(TRANSFER TO|POS|BILL|DEBIT|PURCHASE|PAYMENT)\b", desc, flags=re.I):
                    debit = amount
                else:
                    credit = amount

        rows.append({
            "Date": date,
            "Description": desc,
            "Debit": debit,
            "Credit": credit,
            "Balance": balance
        })

    # Pattern 2: Sometimes amount columns appear as: ...  Debit   Credit   Balance
    # Try a second pass if we got nothing meaningful:
    if not rows:
        tab_re = re.compile(
            rf"^({DATE_SBI})\s+(.*?)\s+({NUM}|-)\s+({NUM}|-)\s+({NUM}|-)\s*$", re.I
        )
        for line in lines:
            m = tab_re.match(line)
            if not m:
                continue
            date, desc, debit_s, credit_s, bal_s = m.groups()
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": _to_float(debit_s if debit_s != "-" else "0"),
                "Credit": _to_float(credit_s if credit_s != "-" else "0"),
                "Balance": _to_float(bal_s if bal_s != "-" else "0"),
            })

    df = pd.DataFrame(rows)
    return _finalize_df(df)

# -----------------------------
# HDFC
# -----------------------------

def parse_hdfc_bank_statement(text: str) -> pd.DataFrame:
    """
    Common HDFC format (CSV-like in PDFs):
    26/11/2020  AMAZON PAY INDIA  66.00  0.00  3000.00
    or
    26/11/2020  AMAZON PAY INDIA  DR  66.00  3000.00
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = _join_wrapped_lines(lines)

    rows = []

    # Pattern A: date desc debit credit balance (all numeric)
    p_a = re.compile(
        rf"^({DATE_SLASH})\s+(.*?)\s+({NUM}|-)\s+({NUM}|-)\s+({NUM}|-)\s*$", re.I
    )
    # Pattern B: date desc (Cr|Dr) amount balance
    p_b = re.compile(
        rf"^({DATE_SLASH})\s+(.*?)\s+(Cr|DR|CR|Dr)\s+({NUM})\s+({NUM})\s*$", re.I
    )

    for line in lines:
        m = p_a.match(line)
        if m:
            date, desc, debit_s, credit_s, bal_s = m.groups()
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": _to_float(debit_s if debit_s != "-" else "0"),
                "Credit": _to_float(credit_s if credit_s != "-" else "0"),
                "Balance": _to_float(bal_s if bal_s != "-" else "0"),
            })
            continue

        m = p_b.match(line)
        if m:
            date, desc, crdr, amount_s, bal_s = m.groups()
            amount = _to_float(amount_s)
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": amount if crdr.lower() == "dr" else 0.0,
                "Credit": amount if crdr.lower() == "cr" else 0.0,
                "Balance": _to_float(bal_s),
            })
            continue

    df = pd.DataFrame(rows)
    return _finalize_df(df)

# -----------------------------
# ICICI
# -----------------------------

def parse_icici_bank_statement(text: str) -> pd.DataFrame:
    """
    ICICI often uses 26-Nov-2020 and Cr/Dr markers:
    26-Nov-2020  ZOMATO  350.00 Dr  10,500.55
    26-Nov-2020  SALARY  50,000.00 Cr  60,500.55
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = _join_wrapped_lines(lines)

    rows = []
    p = re.compile(
        rf"^({DATE_DASH_MON})\s+(.*?)\s+({NUM})\s+(Cr|CR|Dr|DR)\s+({NUM})\s*$", re.I
    )

    # Tabular variant: date desc debit credit balance
    tab = re.compile(
        rf"^({DATE_DASH_MON})\s+(.*?)\s+({NUM}|-)\s+({NUM}|-)\s+({NUM}|-)\s*$", re.I
    )

    for line in lines:
        m = p.match(line)
        if m:
            date, desc, amt_s, crdr, bal_s = m.groups()
            amt = _to_float(amt_s)
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": amt if crdr.lower() == "dr" else 0.0,
                "Credit": amt if crdr.lower() == "cr" else 0.0,
                "Balance": _to_float(bal_s),
            })
            continue

        m = tab.match(line)
        if m:
            date, desc, debit_s, credit_s, bal_s = m.groups()
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": _to_float(debit_s if debit_s != "-" else "0"),
                "Credit": _to_float(credit_s if credit_s != "-" else "0"),
                "Balance": _to_float(bal_s if bal_s != "-" else "0"),
            })
            continue

    df = pd.DataFrame(rows)
    return _finalize_df(df)

# -----------------------------
# Generic fallback (column headers)
# -----------------------------

def parse_generic_bank_statement(text: str) -> pd.DataFrame:
    """
    A header-driven fallback:
    Detects lines after a header that includes some of:
    Date | Description/Particulars | Debit/Withdrawal | Credit/Deposit | Balance
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = _join_wrapped_lines(lines)

    header_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"\bDate\b", line, flags=re.I) and \
           re.search(r"\b(Description|Particulars|Details)\b", line, flags=re.I) and \
           (re.search(r"\b(Debit|Withdrawal)\b", line, flags=re.I) or re.search(r"\b(Credit|Deposit)\b", line, flags=re.I)):
            header_idx = i
            break

    rows = []
    if header_idx != -1:
        for line in lines[header_idx+1:]:
            # try: date  desc  debit  credit  balance (some may be missing)
            m = re.match(
                rf"^\s*({DATE_SBI}|{DATE_SLASH}|{DATE_DASH}|{DATE_DASH_MON})\s+(.*?)\s+({NUM}|-)?\s+({NUM}|-)?\s+({NUM}|-)?\s*$",
                line, flags=re.I
            )
            if not m:
                continue
            date, desc, d_s, c_s, b_s = m.groups()
            rows.append({
                "Date": date,
                "Description": desc,
                "Debit": _to_float(d_s if d_s and d_s != "-" else "0"),
                "Credit": _to_float(c_s if c_s and c_s != "-" else "0"),
                "Balance": _to_float(b_s if b_s and b_s != "-" else "0"),
            })

    df = pd.DataFrame(rows)
    return _finalize_df(df)

# -----------------------------
# Detect + Route
# -----------------------------

def parse_bank_text(text: str) -> pd.DataFrame:
    tl = text.lower()

    # Quick bank detection
    if "state bank of india" in tl or re.search(r"\bsbi\b", tl):
        df = parse_sbi_bank_statement(text)
    elif "hdfc bank" in tl or "hdfc" in tl:
        df = parse_hdfc_bank_statement(text)
    elif "icici bank" in tl or "icici" in tl:
        df = parse_icici_bank_statement(text)
    else:
        df = parse_generic_bank_statement(text)

    # If parser produced empty df, attempt generic fallback anyway
    if df.empty:
        fallback = parse_generic_bank_statement(text)
        if not fallback.empty:
            df = fallback

    return _finalize_df(df)
