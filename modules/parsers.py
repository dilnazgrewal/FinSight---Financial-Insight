import io
import re
from datetime import datetime
import pandas as pd
import pdfplumber
import streamlit as st
from pdf2image import convert_from_bytes
import pytesseract

MONTHS_PATTERN = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"

def _extract_text_from_bytes(file_bytes: bytes) -> str:
    """
    Extracts text from a PDF. Tries a fast method first, then falls back
    to a powerful but slower OCR method if the first one fails.
    """
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text(x_tolerance=1)
                if ptext:
                    text += ptext + "\n"
    except Exception:
        text = "" 
    if len(text.strip()) < 100:
        st.info("📄 Standard text extraction failed. Attempting slower, more powerful OCR scan...")
        try:
            # Convert PDF to a list of images
            images = convert_from_bytes(file_bytes)
            ocr_text = ""
            # Process each image with Tesseract
            for img in images:
                ocr_text += pytesseract.image_to_string(img) + "\n"
            
            # If OCR found more text, use it. Otherwise, keep the original.
            if len(ocr_text.strip()) > len(text.strip()):
                st.success("✅ OCR scan completed successfully.")
                return ocr_text

        except Exception as e:
            st.warning(f"OCR process failed. This may be a scanned PDF that is difficult to read. Error: {e}")

    return text

def _parse_phonepe_from_bytes(file_bytes: bytes, full_text: str, debug: bool = False) -> pd.DataFrame:
    """
    Robust line-based PhonePe parser.
    Returns a DataFrame with Date, Description, Debit, Credit, RawLine.
    """
    transactions = []
    if not full_text:
        full_text = _extract_text_from_bytes(file_bytes)
    if not full_text:
        return pd.DataFrame()

    # Fix glued patterns like "₹10Paid" -> "₹10 Paid"
    normalized_text = re.sub(r'(₹\s*[\d,]+(?:\.\d{1,2})?)(?=[A-Za-z])', r'\1 ', full_text)

    lines = [ln.strip() for ln in normalized_text.splitlines() if ln.strip()]
    if debug:
        st.write("===== PHONEPE - RAW TEXT SAMPLE (first 1200 chars) =====")
        st.text(normalized_text[:1200])

    skip_indices = set()
    n = len(lines)
    i = 0
    while i < n:
        if i in skip_indices:
            i += 1
            continue

        line = lines[i]

        # Find date at start or anywhere in the line
        date_search = re.search(rf'({MONTHS_PATTERN}\s+\d{{1,2}},\s+\d{{4}})', line, re.IGNORECASE)
        if date_search:
            date_str = date_search.group(1)

            # Get time token safely from same line or next line (only the time token, not the whole line)
            time_token = None
            # check same line for time token
            tmatch = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', line, re.IGNORECASE)
            if tmatch:
                time_token = tmatch.group(1)
            else:
                # look ahead a little: next line might start with time (but may also contain Transaction ID)
                if i + 1 < n:
                    next_line = lines[i + 1]
                    tmatch2 = re.match(r'^\s*(\d{1,2}:\d{2}\s*(?:AM|PM))', next_line, re.IGNORECASE)
                    if tmatch2:
                        time_token = tmatch2.group(1)
                        # mark the next line to skip later (we consumed the time token)
                        skip_indices.add(i + 1)

            # parse datetime (safe: only date_str and time_token)
            try:
                if time_token:
                    txn_dt = datetime.strptime(f"{date_str} {time_token}", "%b %d, %Y %I:%M %p")
                else:
                    txn_dt = datetime.strptime(date_str, "%b %d, %Y")
            except Exception:
                # fallback: parse date only (ignore time)
                try:
                    txn_dt = datetime.strptime(date_str, "%b %d, %Y")
                except Exception:
                    txn_dt = pd.to_datetime("today")

            # The transaction detail is usually after the date in the same line
            # e.g., "Jun 29, 2024 Paid to NAGENDRA VARIETY STORE DEBIT ₹10"
            detail_part = line[date_search.end():].strip()

            # If detail_part seems empty, try next non-skip line (but not Transaction ID)
            if not detail_part and i + 1 < n and (i + 1) not in skip_indices:
                lookahead = lines[i + 1]
                if not re.search(r'Transaction ID|UTR|Paid by|Credited to', lookahead, re.IGNORECASE):
                    detail_part = lookahead
                    skip_indices.add(i + 1)

            # If still empty, skip
            if not detail_part:
                i += 1
                continue

            # Extract amount: look in detail_part first, else search nearby lines
            amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', detail_part)
            j = i + 1
            while not amt_match and j < min(n, i + 4):
                if j not in skip_indices:
                    amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', lines[j])
                j += 1

            if not amt_match:
                # no amount found -> skip block
                i += 1
                continue
            amount = float(amt_match.group(1).replace(",", ""))

            # Determine DEBIT / CREDIT
            txn_type = None
            if re.search(r'\bDEBIT\b', detail_part, re.IGNORECASE):
                txn_type = "DEBIT"
            elif re.search(r'\bCREDIT\b', detail_part, re.IGNORECASE):
                txn_type = "CREDIT"
            else:
                # check nearby lines for DEBIT/CREDIT tokens
                for k in range(i, min(n, i + 4)):
                    if k in skip_indices: 
                        continue
                    if re.search(r'\bDEBIT\b', lines[k], re.IGNORECASE):
                        txn_type = "DEBIT"; break
                    if re.search(r'\bCREDIT\b', lines[k], re.IGNORECASE):
                        txn_type = "CREDIT"; break
                # fallback by keywords
                if txn_type is None:
                    if re.search(r'\b(paid to|paid|sent to)\b', detail_part, re.IGNORECASE):
                        txn_type = "DEBIT"
                    elif re.search(r'\b(received from|credited|deposit)\b', detail_part, re.IGNORECASE):
                        txn_type = "CREDIT"
                    else:
                        txn_type = "DEBIT"

            # Build description: remove amount, DEBIT/CREDIT, and junk tokens
            desc = detail_part
            # remove the amount token occurrences
            desc = re.sub(r'₹\s*[\d,]+(?:\.\d{1,2})?', '', desc)
            # remove explicit DEBIT/CREDIT tokens
            desc = re.sub(r'\bDEBIT\b|\bCREDIT\b', '', desc, flags=re.IGNORECASE)
            # remove Transaction ID / UTR / Paid by / Credited to fragments and anything after them
            desc = re.sub(r'(Transaction ID|UTR|UTR No|Ref|Ref No|Paid by|Credited to).*', '', desc, flags=re.IGNORECASE)
            # strip generic labels like "Paid to", "Received from"
            desc = re.sub(r'^\s*(Paid to|Received from)\s*', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'\s+', ' ', desc).strip()

            # If description is empty after cleaning, try to find a likely merchant line nearby
            if not desc:
                candidate = None
                for k in range(i + 1, min(n, i + 6)):
                    if k in skip_indices:
                        continue
                    line_k = lines[k]
                    if re.search(r'(Transaction ID|UTR|Paid by|Credited to)', line_k, re.IGNORECASE):
                        continue
                    # prefer lines that contain letters and not just time/ids
                    if re.search(r'[A-Za-z]', line_k):
                        candidate = line_k
                        break
                desc = candidate.strip() if candidate else "Miscellaneous"

            # Add transaction row
            transactions.append({
                "Date": pd.to_datetime(txn_dt),
                "Description": desc,
                "Debit": float(amount) if txn_type == "DEBIT" else 0.0,
                "Credit": float(amount) if txn_type == "CREDIT" else 0.0,
                "RawLine": detail_part
            })

            # mark some nearby indices to skip (time line, transaction ID line) to avoid reprocessing
            # if we used next line as time or description we already added it to skip_indices
            # also mark subsequent line(s) if they are Transaction ID or UTR lines
            if i + 1 < n and re.search(r'(Transaction ID|UTR|UTR No|Ref|Ref No)', lines[i + 1], re.IGNORECASE):
                skip_indices.add(i + 1)

        i += 1  # move to next line

    df = pd.DataFrame(transactions)
    if debug:
        st.write("===== Parsed PhonePe transactions (first 10) =====")
        if not df.empty:
            st.dataframe(df.head(10))
        else:
            st.write("No transactions parsed from the PhonePe blocks. (See raw text sample above)")
    return df


def _parse_gpay_from_bytes(file_bytes: bytes, full_text: str, debug: bool = False) -> pd.DataFrame:
    """
    Robust line-based Google Pay parser that mirrors the PhonePe parser's behavior.
    Returns a DataFrame with Date, Description, Debit, Credit, RawLine.
    """
    transactions = []
    if not full_text:
        full_text = _extract_text_from_bytes(file_bytes)
    if not full_text:
        return pd.DataFrame()

    normalized_text = re.sub(r'([A-Za-z])₹', r'\1 ₹', full_text)
    normalized_text = re.sub(r'(₹\s*[\d,]+(?:\.\d{1,2})?)(?=[A-Za-z])', r'\1 ', normalized_text)

    lines = [ln.strip() for ln in normalized_text.splitlines() if ln.strip()]
    if debug:
        st.write("===== GPAY - RAW TEXT SAMPLE (first 1200 chars) =====")
        st.text(normalized_text[:1200])

    skip_indices = set()
    n = len(lines)
    i = 0

    while i < n:
        if i in skip_indices:
            i += 1
            continue

        line = lines[i]

        # Find date pattern like "02 Sep, 2025" or "2 Sep, 2025"
        date_search = re.search(rf'(\d{{1,2}}\s+{MONTHS_PATTERN},\s+\d{{4}})', line)
        if date_search:
            date_str = date_search.group(1)

            # safe parse date (GPay uses day-month-year like "02 Sep, 2025")
            try:
                txn_date = datetime.strptime(date_str, "%d %b, %Y")
            except Exception:
                txn_date = pd.to_datetime("today")

            # Get time token from same or next line if present (e.g. "06:10 PM")
            time_token = None
            tmatch = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM))', line, re.IGNORECASE)
            if tmatch:
                time_token = tmatch.group(1)
            else:
                if i + 1 < n:
                    tmatch2 = re.match(r'^\s*(\d{1,2}:\d{2}\s*(?:AM|PM))', lines[i + 1], re.IGNORECASE)
                    if tmatch2:
                        time_token = tmatch2.group(1)
                        skip_indices.add(i + 1)

            # combine into full datetime if possible
            if time_token:
                try:
                    txn_dt = datetime.strptime(f"{date_str} {time_token}", "%d %b, %Y %I:%M %p")
                except Exception:
                    txn_dt = txn_date
            else:
                txn_dt = txn_date

            # The detail is usually after the date in the same line
            detail_part = line[date_search.end():].strip()

            # If detail_part empty, try next non-skipped line (but avoid UPI/ID lines)
            if not detail_part and i + 1 < n and (i + 1) not in skip_indices:
                lookahead = lines[i + 1]
                if not re.search(r'UPI Transaction ID|Transaction ID|Paid by|Paid to HDFC|UTR', lookahead, re.IGNORECASE):
                    detail_part = lookahead
                    skip_indices.add(i + 1)

            if not detail_part:
                i += 1
                continue

            # Extract amount: prefer detail_part, else nearby lines
            amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', detail_part)
            j = i + 1
            while not amt_match and j < min(n, i + 5):
                if j not in skip_indices:
                    amt_match = re.search(r'₹\s*([\d,]+(?:\.\d{1,2})?)', lines[j])
                    if amt_match:
                        skip_indices.add(j)
                j += 1

            if not amt_match:
                i += 1
                continue

            # parse amount safely
            try:
                amount = float(amt_match.group(1).replace(",", ""))
            except Exception:
                i += 1
                continue

            # Determine DEBIT / CREDIT (GPay uses "Paid to" and "Received from", but also "Paid by")
            txn_type = None
            if re.search(r'\b(Paid to|Paid)\b', detail_part, re.IGNORECASE):
                txn_type = "DEBIT"
            elif re.search(r'\b(Received from|Received)\b', detail_part, re.IGNORECASE):
                txn_type = "CREDIT"
            else:
                # look nearby for explicit tokens
                for k in range(i, min(n, i + 4)):
                    if k in skip_indices:
                        continue
                    if re.search(r'\b(Paid to|Paid)\b', lines[k], re.IGNORECASE):
                        txn_type = "DEBIT"; break
                    if re.search(r'\b(Received from|Received)\b', lines[k], re.IGNORECASE):
                        txn_type = "CREDIT"; break
                if txn_type is None:
                    txn_type = "DEBIT"  # safe default

            # Build description: remove amount, tokens, and IDs
            desc = detail_part
            desc = re.sub(r'₹\s*[\d,]+(?:\.\d{1,2})?', '', desc)
            desc = re.sub(r'\b(Paid to|Received from|Paid|Received)\b', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'(UPI Transaction ID|Transaction ID|UTR|Paid by|Paid to HDFC Bank|Paid to Axis Bank|Paid to Bank).*', '', desc, flags=re.IGNORECASE)
            desc = re.sub(r'\s+', ' ', desc).strip()

            # fallback: if desc empty, find nearby merchant-like line
            if not desc:
                candidate = None
                for k in range(i + 1, min(n, i + 6)):
                    if k in skip_indices:
                        continue
                    if re.search(r'[A-Za-z]', lines[k]) and not re.search(r'UPI|Transaction ID|UTR|Paid by', lines[k], re.IGNORECASE):
                        candidate = lines[k]
                        break
                desc = (candidate or "Miscellaneous").strip()

            transactions.append({
                "Date": pd.to_datetime(txn_dt),
                "Description": desc,
                "Debit": float(amount) if txn_type == "DEBIT" else 0.0,
                "Credit": float(amount) if txn_type == "CREDIT" else 0.0,
                "RawLine": detail_part
            })

            # skip UPI lines if they immediately follow
            if i + 1 < n and re.search(r'(UPI Transaction ID|Transaction ID|UTR)', lines[i + 1], re.IGNORECASE):
                skip_indices.add(i + 1)

        i += 1

    # Final normalization to match PhonePe parser outputs exactly
    df = pd.DataFrame(transactions)
    # Ensure columns exist
    for col in ["Date", "Description", "Debit", "Credit", "RawLine"]:
        if col not in df.columns:
            df[col] = 0.0 if col in ["Debit", "Credit"] else ""

    # Coerce types (very important: ai_categorizer expects numbers/datetimes/strings)
    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").fillna(pd.to_datetime("today"))
    except Exception:
        df["Date"] = pd.to_datetime("today")

    try:
        df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0.0)
        df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0.0)
    except Exception:
        df["Debit"] = df["Credit"] = 0.0

    df["Description"] = df["Description"].astype(str).fillna("Miscellaneous").str.strip()
    df["RawLine"] = df["RawLine"].astype(str).fillna("")

    df = df.reset_index(drop=True)[["Date", "Description", "Debit", "Credit", "RawLine"]]

    if debug:
        st.write("===== Parsed GPay transactions (first 10) =====")
        if not df.empty:
            st.dataframe(df.head(10))
            st.write("dtypes:", df.dtypes)
        else:
            st.warning("No transactions parsed from the GPay file. See raw text sample above.")
            st.text(normalized_text[:1200])
    df["Description"] = (
    "GPay Transaction: " + df["Description"].astype(str)).str.strip()
    
    return df


def _parse_sbi_from_bytes(file_bytes: bytes, full_text: str, debug: bool = False) -> pd.DataFrame:
    """
    SBI parser using table extraction (best-effort).
    """
    transactions = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if not table or len(table) < 2:
                            continue
                        header = [str(c).strip().lower() if c else "" for c in table[0]]
                        joined = " ".join(header)
                        if "date" in joined and ("detail" in joined or "description" in joined or "particular" in joined):
                            # Detect indices
                            date_idx = next((i for i, h in enumerate(header) if "date" in h), 0)
                            desc_idx = next((i for i, h in enumerate(header) if any(x in h for x in ["detail", "description", "particular"])), 1)
                            debit_idx = next((i for i, h in enumerate(header) if "debit" in h or "withdraw" in h or "dr" == h.strip()), None)
                            credit_idx = next((i for i, h in enumerate(header) if "credit" in h or "deposit" in h or "cr" == h.strip()), None)

                            for row in table[1:]:
                                row = [c if c is not None else "" for c in row] + [""] * max(0, (len(header) - len(row)))
                                date_raw = str(row[date_idx]).strip() if date_idx < len(row) else ""
                                desc_raw = str(row[desc_idx]).strip() if desc_idx < len(row) else ""
                                debit_raw = str(row[debit_idx]).strip() if (debit_idx is not None and debit_idx < len(row)) else ""
                                credit_raw = str(row[credit_idx]).strip() if (credit_idx is not None and credit_idx < len(row)) else ""

                                # parse date
                                parsed_date = None
                                for fmt in ("%d %b %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
                                    try:
                                        parsed_date = datetime.strptime(date_raw, fmt)
                                        break
                                    except Exception:
                                        continue
                                if parsed_date is None:
                                    m = re.search(r'(\d{1,2}\s[A-Za-z]{3}\s\d{4})', date_raw)
                                    if m:
                                        try:
                                            parsed_date = datetime.strptime(m.group(1), "%d %b %Y")
                                        except Exception:
                                            parsed_date = None

                                if parsed_date is None:
                                    continue

                                # amounts
                                def _extract_amount(s):
                                    if not s:
                                        return 0.0
                                    mm = re.search(r'([\d,]+(?:\.\d{1,2})?)', s.replace(',', ''))
                                    return float(mm.group(1)) if mm else 0.0

                                transactions.append({
                                    "Date": pd.to_datetime(parsed_date),
                                    "Description": re.sub(r'\s+', ' ', desc_raw).strip() or "Miscellaneous",
                                    "Debit": _extract_amount(debit_raw),
                                    "Credit": _extract_amount(credit_raw)
                                })
                except Exception:
                    continue
    except Exception:
        pass

    df = pd.DataFrame(transactions)
    if debug:
        st.write("===== Parsed SBI transactions (first 10) =====")
        st.dataframe(df.head(10) if not df.empty else pd.DataFrame())
    return df

# Add this entire new function to your parsers.py file

# In modules/parsers.py, replace the indian_bank parser with this one:

def _parse_indian_bank_from_bytes(file_bytes: bytes, full_text: str, debug: bool = False) -> pd.DataFrame:
    """
    FINAL VERSION: A robust parser for Indian Bank statements.
    It uses a prioritized, regex-based approach to clean complex descriptions.
    """
    transactions = []
    try:
        # We start with the base table extraction from the SBI parser
        df_base = _parse_sbi_from_bytes(file_bytes, full_text, debug)
        if df_base.empty:
            return pd.DataFrame()

        for _, row in df_base.iterrows():
            raw_desc = row['Description']
            clean_desc = "Bank Transaction" # Start with a safe default

            # --- Prioritized Cleaning Rules ---

            # Priority 1: Check for high-confidence, specific keywords first
            if "zerodha" in raw_desc.lower():
                clean_desc = "Zerodha"
            elif "atm cash" in raw_desc.lower():
                clean_desc = "ATM Cash Withdrawal"
            elif "sms charges" in raw_desc.lower():
                clean_desc = "Bank SMS Charges"
            elif "credit interest" in raw_desc.lower():
                clean_desc = "Interest Credited"
            
            # Priority 2: Try to find a standard UPI merchant or name
            # This regex looks for content between the 3rd and 4th slashes
            else:
                upi_match = re.search(r'UPI/[^/]+/[^/]+/(.*?)/', raw_desc)
                if upi_match:
                    # Clean up the extracted name
                    extracted_name = upi_match.group(1).strip()
                    if extracted_name: # Ensure it's not empty
                        clean_desc = extracted_name.title()

                # Priority 3: Try to find merchant names in other complex strings
                # This looks for a sequence of 2 or more capitalized words, often the merchant
                else:
                    merchant_match = re.search(r'\b([A-Z][A-Z\s]{3,})\b', raw_desc)
                    if merchant_match:
                        clean_desc = merchant_match.group(1).strip().title()

            transactions.append({
                "Date": row['Date'],
                "Description": clean_desc,
                "Debit": row['Debit'],
                "Credit": row['Credit'],
                "RawLine": raw_desc # Always populate RawLine
            })
        
        return pd.DataFrame(transactions)

    except Exception as e:
        if debug:
            st.error(f"Indian Bank parser failed: {e}")
        return pd.DataFrame()
    

def parse_pdf(pdf_type: str, uploaded_file, debug: bool = False):
    """
    Router that accepts:
      - pdf_type: string like 'PhonePe', 'SBI', or None
      - uploaded_file: a Streamlit UploadedFile or bytes or filepath
    Returns: (df, detected_type, full_text)
    """
    # Read bytes robustly
    try:
        if hasattr(uploaded_file, "read"):
            file_bytes = uploaded_file.read()
            try:
                uploaded_file.seek(0)
            except Exception:
                pass
        elif isinstance(uploaded_file, (bytes, bytearray)):
            file_bytes = bytes(uploaded_file)
        else:
            with open(uploaded_file, "rb") as f:
                file_bytes = f.read()
    except Exception as e:
        raise RuntimeError(f"Could not read uploaded_file bytes: {e}")

    full_text = _extract_text_from_bytes(file_bytes)

    # If pdf_type was not provided, try to detect using simple heuristics
    detected_type = pdf_type
    if not detected_type:
        if re.search(r'phonepe', full_text, re.IGNORECASE):
            detected_type = "PhonePe"
        elif re.search(r'state bank of india|sbi', full_text, re.IGNORECASE):
            detected_type = "SBI"
        elif re.search(r'google pay|gpay', full_text, re.IGNORECASE):
            detected_type = "GPay"
        else:
            detected_type = "UNKNOWN"

    # Route to parser(s)
    df = pd.DataFrame()
    if detected_type and detected_type.lower().startswith("phone"):
        df = _parse_phonepe_from_bytes(file_bytes, full_text, debug=debug)
    elif detected_type and detected_type.lower() == "indian bank":
        df = _parse_indian_bank_from_bytes(file_bytes, full_text, debug=debug)
    elif detected_type and detected_type.lower().startswith("sbi"):
        df = _parse_sbi_from_bytes(file_bytes, full_text, debug=debug)
    elif detected_type and detected_type.lower().startswith("gpay"):
        df = _parse_gpay_from_bytes(file_bytes, full_text, debug=debug)
    else:
        # Try phonepe first (text style); if empty, try SBI style table parser
        df = _parse_phonepe_from_bytes(file_bytes, full_text, debug=debug)
        if df.empty:
            df = _parse_sbi_from_bytes(file_bytes, full_text, debug=debug)

    # Final normalization
    if df is None:
        df = pd.DataFrame()
    for col in ["Date", "Description", "Debit", "Credit"]:
        if col not in df.columns:
            df[col] = 0.0 if col in ["Debit", "Credit"] else ""

    try:
        df["Debit"] = pd.to_numeric(df["Debit"], errors="coerce").fillna(0.0)
        df["Credit"] = pd.to_numeric(df["Credit"], errors="coerce").fillna(0.0)
    except Exception:
        pass

    return df.reset_index(drop=True), detected_type, full_text