import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- Hugging Face Client ----------------
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

# ---------------- Streamlit App ----------------
st.title("💬 Kemet – Discover Egypt Through Conversation")

# Sidebar settings
st.sidebar.header("⚙ Settings")
thinking_mode = st.sidebar.checkbox("Enable Thinking Mode", value=False)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Type your message here...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages for the API
    api_messages = []
    for m in st.session_state.messages:
        api_messages.append({"role": m["role"], "content": m["content"]})

    if thinking_mode:
        api_messages.append({"role": "system", "content": "Think step by step before answering."})

    # Call Hugging Face OpenAI-compatible API
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 Thinking...")

        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b:fireworks-ai",  # ⚠ غير الاسم لو عندك موديل تاني
                messages=api_messages,
                max_tokens=500,
                temperature=0.7,
            )
            model_response = completion.choices[0].message.content.strip()
        except Exception as e:
            model_response = f"❌ API Error: {e}"

        message_placeholder.markdown(model_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": model_response})





