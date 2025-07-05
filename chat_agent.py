"""
Tiny wrapper around the Groq Python SDK.
• Requires:  pip install groq
• Set your API key once:
      export GROQ_API_KEY="sk-..."
"""

from groq import Groq
import os

# Choose any production model from https://console.groq.com/docs/models
MODEL_ID = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = Groq()            # uses GROQ_API_KEY env var


def chat(messages: list[dict]) -> str:
    """
    messages – list like OpenAI spec, e.g.
        [{"role":"system","content":"You are ..."},
         {"role":"user","content":"Hello"}]
    returns assistant reply str
    """
    resp = _client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return resp.choices[0].message.content
