import re

def clean_description(desc):
    desc = desc.lower()
    # Remove UPI IDs, numbers, and extra slashes
    desc = re.sub(r'upi/\w+/\d+', '', desc)
    desc = re.sub(r'\b\d{6,}\b', '', desc)  # long numbers
    desc = re.sub(r'[\|/\\_-]+', ' ', desc)
    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip()
