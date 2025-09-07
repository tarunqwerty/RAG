import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os, asyncio, sys

# 🔑 Setup Gemini API Key
os.environ["GOOGLE_API_KEY"] = ""

# ✅ Ensure event loop exists (for Streamlit threads on Windows)
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def get_embeddings():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 🔹 Function to load all PDFs
def load_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files[:5]:
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 RAG Bot with Gemini + Memory")
st.write("Upload up to 5 PDFs and chat with them (context-aware).")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_text = load_pdfs(uploaded_files)

    if not all_text.strip():
        st.error("❌ No text extracted from PDFs. Upload text-based PDFs.")
    else:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(all_text)

        # ✅ Safe embeddings init
        embeddings = get_embeddings()

        # Create Vector DB
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # ✅ Memory for conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Conversational QA Chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )

        # Initialize session state for storing history
        if "history" not in st.session_state:
            st.session_state.history = []

        # User query
        query = st.text_input("Ask me anything from the uploaded PDFs 👇")
        if query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = qa_chain({"question": query})
                    answer = result["answer"]

                    # Store in chat history
                    st.session_state.history.append(("You", query))
                    st.session_state.history.append(("Bot", answer))

                    # Display chat
                    for role, msg in st.session_state.history:
                        if role == "You":
                            st.markdown(f"**🧑 {role}:** {msg}")
                        else:
                            st.markdown(f"**🤖 {role}:** {msg}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
