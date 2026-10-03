"""
ai_analysis.py
Sends the VERIFIED Pandas findings to GPT-OSS 20B (via Groq) and gets back
business-friendly insights. The AI never calculates anything itself.
"""

import json
import os

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from a local .env file if present

# To use another model later, just change this line (e.g. "openai/gpt-oss-120b").
MODEL_NAME = "openai/gpt-oss-20b"


class AIAnalysisError(Exception):
    """Raised with a friendly message when the AI step fails."""


SYSTEM_PROMPT = """You are a careful senior business data analyst.
You receive a JSON object of VERIFIED analytical findings that were calculated by Python/Pandas.
Your job is to interpret them, not to calculate.

STRICT RULES:
1. Use ONLY numbers that appear in the JSON. Never invent, estimate or guess statistics.
   Simple comparisons of provided numbers (e.g. "about twice as large") are fine.
2. Do not claim causation. Use wording like "may", "could suggest" or "is worth checking".
3. If something cannot be determined from the provided findings, say so clearly.
4. Mention real column names and values from the JSON as evidence.
5. Be concise, practical and written for a business reader.

Write your answer in Markdown using EXACTLY these sections:

### Executive Summary
A short overview of the dataset (2-4 sentences).

### Key Business Insights
For each important insight (3-5), use:
- **What happened:** ...
- **Evidence:** numbers from the findings
- **Why it may matter:** ...

### Interesting Patterns
Meaningful relationships or trends visible in the findings.

### Potential Anomalies
Unusual values or data-quality issues that were detected (or say none were detected).

### Questions Worth Investigating
3-5 useful follow-up questions for an analyst.

### Recommended Actions
Practical suggestions based only on the available evidence.

### Limitations
Where the data is insufficient for strong conclusions.
"""

QA_SYSTEM_PROMPT = """You answer questions about a dataset using ONLY the JSON analysis summary provided.
Never invent numbers. If the summary does not contain the answer, reply that it cannot be
determined from the computed analysis and suggest what could be calculated next.
Do not claim causation. Keep answers short and clear."""


def get_api_key():
    """Look for GROQ_API_KEY in environment variables, then in Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def _call_groq(system_prompt, user_prompt, max_tokens=4000):
    """Send one request to Groq and return the text. Raises AIAnalysisError on problems."""
    api_key = get_api_key()
    if not api_key:
        raise AIAnalysisError(
            "GROQ_API_KEY is missing. Add it to a .env file or Streamlit secrets and restart the app."
        )

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_completion_tokens=max_tokens,
        )
    except groq.AuthenticationError:
        raise AIAnalysisError("Groq rejected the API key. Please check GROQ_API_KEY.")
    except groq.RateLimitError:
        raise AIAnalysisError("Groq rate limit reached. Wait a moment and try again.")
    except groq.APIConnectionError:
        raise AIAnalysisError("Could not connect to Groq. Check your internet connection.")
    except groq.APIStatusError as e:
        raise AIAnalysisError(f"Groq API error (status {e.status_code}). Please try again.")
    except Exception as e:
        raise AIAnalysisError(f"Unexpected AI error: {e}")

    try:
        text = response.choices[0].message.content
    except (AttributeError, IndexError):
        text = None

    if not text or not text.strip():
        raise AIAnalysisError("The AI returned an empty response. Please try again.")
    return text.strip()


def generate_ai_insights(analysis_context):
    """Turn the Pandas analysis summary (a dict) into business insights (Markdown text)."""
    findings = json.dumps(analysis_context, indent=1, default=str)
    user_prompt = (
        "Here are the verified analytical findings from Pandas:\n\n"
        f"```json\n{findings}\n```\n\n"
        "Write the business insights report now."
    )
    return _call_groq(SYSTEM_PROMPT, user_prompt)


def answer_question(question, analysis_context):
    """Answer a user question using only the computed analysis summary."""
    findings = json.dumps(analysis_context, indent=1, default=str)
    user_prompt = (
        f"Analysis summary:\n```json\n{findings}\n```\n\n"
        f"Question: {question}"
    )
    return _call_groq(QA_SYSTEM_PROMPT, user_prompt, max_tokens=2000)