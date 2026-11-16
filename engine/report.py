import json

import anthropic

SYSTEM = (
    "You are a risk analyst writing for a beginner investor. "
    "Use ONLY the numbers in the JSON you are given and never invent new numbers. "
    "Explain the 3 biggest risks and why each matters, in short paragraphs. "
    "Then list general ways people reduce these kinds of risk. "
    "Never tell the user to buy or sell any specific asset. "
    "End with: 'Analytics only, not investment advice. Past data does not predict the future.'"
)


def ai_report(metrics: dict, api_key: str, model: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model, max_tokens=900, system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(metrics, default=str)}],
    )
    return msg.content[0].text