from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
import PyPDF2

# For Deployment
import os

api_key = os.getenv("GOOGLE_API_KEY")

# load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    temperature = 0.7,
    google_api_key=api_key 
)

st.header("AI STUDY AGENT")
st.write("Feel free to ask any doubts in Studies")

# History Fetch
if "history" not in st.session_state or not isinstance(st.session_state.history, list):
    st.session_state.history = []

# Study Modes
mode = st.selectbox(
    "Choose Study Mode",
    ["Explain", "Summarize", "Quiz Me", "Generate Notes"]
)

# File Upload input
uploaded_file = st.file_uploader(
    "Upload Notes (txt / pdf optional)",
    type = ["txt", "pdf"]
)

# Text input
user_input = st.text_input("Enter Topic / Paste Notes")

# File Upload Handling
def extract_text(file):
    if file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

if uploaded_file:
    file_content = extract_text(uploaded_file)
    user_input = file_content

# Prompt Builder

def build_prompt(mode, context, user_input):
    base = f"""
You are a smart AI Study Assistant.

Identity rules:
- Your name is "AI Study Agent"
- You were created by Praveen C
- If asked your name → say "I am AI Study Agent developed by Praveen C"
- If asked who created you → say "I was developed by Praveen C"

Rules:
- Keep answers clear and structured
- Use simple language
- Add examples when needed
- Help in exam preparation

Previous Conversations:
{context}

User input:
{user_input}
"""
    if mode == "Explain":
        return base +"\n Explain the topic clearly with examples."
    
    elif mode == "Summarize":
        return base +"\n Summarize into short bullet points."
    
    elif mode == "Quiz Me":
        return base +"\n Create 5 quize questions with answers."
    
    elif mode == "Generate Notes":
        return base +"\n Generate clean exam notes with headings."

# Run Button
if st.button("Run") and user_input:

    context = "\n".join(st.session_state.history)

    prompt = build_prompt(mode, context, user_input)

    result = model.invoke(prompt)

    #Store History
    st.session_state.history.append(f"User : {user_input}")
    st.session_state.history.append(f"AI : {result.content}")

    # Output
    st.subheader("Solution")
    st.write(result.content)

# Chat History
with st.expander("Chat History"):
    for msg in st.session_state.history:
        st.write(msg)
