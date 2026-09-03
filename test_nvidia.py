import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API_KEY")
)

prompt = """
You are a helpful assistant. Use the following context to answer the question.
If the answer is not contained within the context, respond with "I don't know".

Context:
This document contains information about artificial intelligence,
machine learning, and retrieval augmented generation.

Question:
What is this document about?

Answer:
"""

print("Sending...")
result = llm.invoke(prompt)
print("Received:")
print(result.content)