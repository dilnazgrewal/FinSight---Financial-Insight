import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variable from .env
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

try:
    # Use a supported model
    model = genai.GenerativeModel("models/gemini-1.5-flash")

    # Sample input
    prompt = "Classify this transaction: Paid ₹500 to Swiggy for dinner."

    # Make the request
    response = model.generate_content(prompt)

    # Show result
    print("✅ Gemini API is working.")
    print("Response:\n", response.text.strip())

except Exception as e:
    print("❌ Gemini API call failed.")
    print("Error:", e)
