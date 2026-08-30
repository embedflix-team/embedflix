from anthropic import Anthropic

client = Anthropic(api_key="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")

response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=10,
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

print(response)