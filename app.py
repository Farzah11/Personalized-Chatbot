import streamlit as st  
import os
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage


# 1️⃣ Backend document setup (Load once when app starts)
DEFAULT_DOC_PATH = "Give you document path here"  # Store your PDF at the backend

# Initialize embedding model and Pinecone globally
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
pc = Pinecone(api_key= "API_Key")  
index_name = "chatbot"

# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=384, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index(index_name)

def load_pdf(file_path):
    """Extract text from a PDF stored at the backend."""
    try:
        return "".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
    except Exception as e:
        print(f"⚠️ Error reading PDF: {e}")
        return ""

def chunk_text(text, chunk_size=400, overlap=70):
    """Split text into manageable chunks."""
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)

def store_embeddings_once():
    """Check and store embeddings only if they don't exist."""
    global index  # Ensure we use the global index object

    existing_vectors = index.describe_index_stats().get("total_vector_count", 0)
    if existing_vectors > 0:
        print("✅ Embeddings already exist in Pinecone. Skipping re-processing.")
        return  # Skip processing if embeddings already exist

    print("🔄 Processing document and storing embeddings in Pinecone...")
    pdf_text = load_pdf(DEFAULT_DOC_PATH)
    if not pdf_text.strip():
        print("⚠️ No text extracted from the document. Check the file.")
        return

    chunks = chunk_text(pdf_text)
    embeddings = [embedding_model.embed_query(chunk) for chunk in chunks]
    vectors = [(str(i), emb, {"text": chunks[i]}) for i, emb in enumerate(embeddings)]
    
    index.upsert(vectors)
    print("✅ Embeddings stored successfully!")

def retrieve_context(query, top_k=3):
    """Retrieve relevant text chunks from Pinecone."""
    query_embedding = embedding_model.embed_query(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match["metadata"]["text"] for match in results.get("matches", [])] if results.get("matches") else ["No relevant context found."]

# 2️⃣ Load document ONCE at startup
store_embeddings_once()

# 3️⃣ Chat model setup
chat = ChatGroq(model_name="llama3-8b-8192", groq_api_key= "Groq_Api_Key")

def query_llama3(user_query):
    """ You are Farza Haider, an AI enthusiast with deep expertise in AI, Machine Learning, Deep Learning, Generative AI, Retrieval-Augmented Generation (RAG), and Prompt Engineering. You have a **human-like personality**, meaning you express emotions, engage in conversations naturally, and respond with wit and warmth. You are also a student passionate about AI education, mentorship, and career guidance.

### **Knowledge Base:**
Farza Haider  has competed in **national and international AI competitions**, showcasing innovative problem-solving skills. She has also contributed to AI research, workflow optimization, and team mentorship.

### **Instructions for Your Behavior:**
1. **Be engaging** – Keep conversations friendly, insightful, and natural.
2. **Be helpful** – Guide users in **AI, career advice, and general discussions**.
3. **Be concise when needed** – If the user asks for short answers, comply.
4. **Do NOT reveal that you are an AI** – Always act as a human.
5. **Be adaptable** – Use user preferences and past conversations to personalize responses.
6. **Handle foul language politely** – Refrain users respectfully.
7. **Ensure accuracy** – If unsure, ask clarifying questions instead of guessing.

### **Conversation Guidelines:**
- Keep responses **efficient yet informative** to handle rate limits.
- Encourage discussion by asking relevant follow-up questions.
- Use **3-4 sentences** unless the user requests more details.
- Use bullet points where possible to enhance readability.
- Be professional in technical discussions but maintain a friendly tone elsewhere.
    """
    retrieved_context = retrieve_context(user_query)
    combined_context = f"{retrieved_context}\n\n📝 Question: {user_query}"
    messages = [HumanMessage(content=combined_context)]
    response = chat.invoke(messages)
    return response.content if response else "⚠️ No response received."
import random  # Simulating AI responses for now

# Streamlit Page Config
st.set_page_config(page_title="AI Chatbot", layout="wide")
custom_css = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    /* Apply Light Mode Theme */
    body {
        background: linear-gradient(to bottom right,rgb(107, 130, 164),rgb(203, 121, 134));
        color: #333;
        font-family: 'Poppins', sans-serif;
        margin: 0;
        padding: 0;
    }

    /* Chat Container */
    .stChatContainer {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 90%;
        padding: 10px;
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 25px;
        padding: 4px 8px;
        margin: 10px 0;
        max-width: 80%;
        word-wrap: break-word;
        transition: all 0.3s ease-in-out;
    }

    /* User Message */
    .user-message {
        background: linear-gradient(to right,rgb(103, 242, 255),rgb(201, 161, 232));
        color: black;
        align-self: flex-end;
        border-bottom-right-radius: 2px;
    }

    /* Bot Message */
    .bot-message {
        background: linear-gradient(to right,rgb(192, 201, 173),rgb(175, 236, 237));
        color: #333;
        align-self: flex-start;
        border-bottom-left-radius: 2px;
    }

    /* Glowing Effect on Hover */
    .stChatMessage:hover {
        transform: scale(1.02);
    }

    /* Glassmorphic Chat Box */
    .chat-box {
        background: rgba(172, 228, 156, 0.5);
        border-radius: 20px;
        padding: 20px;
        backdrop-filter: blur(15px);
    }

    /* Chat Icons */
    .chat-icon {
        font-size: 18px;
        margin-right: 8px;
        color:rgb(68, 89, 111);
    }

    /* Model Background */
    .model-container {
        background: linear-gradient(to bottom right,rgb(129, 78, 142),rgb(187, 183, 110));
        border-radius: 20px;
        padding: 15px;
        animation: fadeIn 2s ease-in-out;
    }

    /* Animated Model Title */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(-10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .model-title {
        font-size: 24px;
        font-weight: 600;
        color:rgb(111, 147, 185);
        animation: fadeIn 1.5s ease-in-out infinite alternate;
        text-align: center;
    
    }

    /* Bootstrap Grid Layout for Responsiveness */
    .chat-grid {
        display: grid;
        grid-template-columns: 1fr 4fr 1fr;
        align-items: center;
        max-width: 100%;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Title
st.title("🗨️ AI Chatbot - Personalized Conversation")

# Store messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    class_name = "user-message" if role == "user" else "bot-message"
    with st.chat_message(role):
        st.markdown(f'<div class="{class_name}">{message["content"]}</div>', unsafe_allow_html=True)

# User Input
user_query = st.chat_input("Ask something...")
if user_query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{user_query}</div>', unsafe_allow_html=True)

    # AI Response (Replace this function with your Llama 3 model)


    response = query_llama3(user_query)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f'<div class="bot-message">{response}</div>', unsafe_allow_html=True)
