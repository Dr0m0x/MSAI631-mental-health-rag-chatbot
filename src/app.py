"""
app.py — Gradio chat interface for the Mental Health Support Chatbot.

Run locally:   python app.py
Then open the local URL it prints (usually http://127.0.0.1:7860).
"""

import gradio as gr
from rag import generate


def respond(message, history):
    if not message or not message.strip():
        return "Please type a question about stress, sleep, anxiety, or general wellbeing."
    try:
        return generate(message)
    except Exception as e:  # keep the UI alive if a call fails
        return (
            "Sorry, I ran into a problem answering that. Please try again.\n\n"
            f"(Technical detail: {e})"
        )


demo = gr.ChatInterface(
    fn=respond,
    title="💚 Mental Health Support Chatbot",
    description=(
        "Ask general questions about stress, sleep, anxiety, and wellbeing. "
        "This bot shares educational information from trusted sources — "
        "it is not a substitute for professional care."
    ),
    examples=[
        "How can I manage everyday stress?",
        "What are some healthy sleep habits?",
        "What is cognitive behavioural therapy?",
    ],
)

if __name__ == "__main__":
    demo.launch()
