import ollama

response = ollama.chat(
    model="llama2",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue? Answer in 3 words.",
        },
    ],
)
print(response["message"]["content"])
