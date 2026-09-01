import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Streamlit UI Config ---
st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 Document Intelligence Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Chat directly with your PDF documents using RAG (Retrieval-Augmented Generation).</div>', unsafe_allow_html=True)

# --- Sidebar UI ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Check secrets or ask user
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=secret_key,
        type="password",
        placeholder="sk-...",
        help="Provide your OpenAI API key to run the model."
    )
    
    st.markdown("---")
    st.header("📄 Upload Data")
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to index."
    )
    
    st.markdown("---")
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "indexed_file_names" not in st.session_state:
    st.session_state.indexed_file_names = []


# --- Helper Functions ---
def format_docs(docs):
    """Combines chunk contents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_pipeline(files, api_key):
    """Processes uploaded documents and builds the LangChain LCEL pipeline."""
    docs = []
    
    # Read files temporarily
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
            os.remove(tmp.name)

    if not docs:
        return None

    # 1. Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # 2. Vector Store Setup
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Model Setup
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key
    )

    # 4. History-aware rephraser prompt
    rephrase_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if "
            "needed and otherwise return it as is."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    rephrase_chain = rephrase_prompt | llm | StrOutputParser() | retriever | format_docs

    # 5. QA Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful and precise assistant. Answer the user's question based strictly on the "
            "provided retrieved context below. If the context doesn't contain the answer, state that "
            "the document does not provide enough information.\n\n"
            "Retrieved Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # 6. End-to-End Chain
    rag_chain = (
        RunnablePassthrough.assign(context=rephrase_chain)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


# --- Handle Document Ingestion ---
current_uploaded_names = [f.name for f in uploaded_files] if uploaded_files else []

if uploaded_files and openai_api_key:
    if st.session_state.indexed_file_names != current_uploaded_names:
        with st.status("Indexing documents...", expanded=True) as status:
            st.write("Extracting PDF contents...")
            st.write("Building vector embeddings with FAISS...")
            st.session_state.rag_chain = build_rag_pipeline(uploaded_files, openai_api_key)
            st.session_state.indexed_file_names = current_uploaded_names
            status.update(label="✅ Indexing Complete! Ready to chat.", state="complete", expanded=False)

# Empty state notices
if not openai_api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to get started.")
elif not uploaded_files:
    st.info("👈 Please upload one or more PDF files to begin.")

# --- Display Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Interaction & Streaming ---
if user_query := st.chat_input("Ask a question about your files..."):
    if not openai_api_key:
        st.warning("Please provide your OpenAI API key.")
    elif st.session_state.rag_chain is None:
        st.warning("Please upload PDF documents first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Build message history for LangChain
        formatted_history = [
            (
                ("human", m["content"])
                if m["role"] == "user"
                else ("assistant", m["content"])
            )
            for m in st.session_state.messages[:-1]
        ]

        # Display assistant streaming response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            # Stream the generated output directly to UI
            for chunk in st.session_state.rag_chain.stream(
                {"question": user_query, "chat_history": formatted_history}
            ):
                full_response += chunk
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
