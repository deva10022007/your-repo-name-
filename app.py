import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Document Assistant (RAG Chat)")
st.write("Upload your PDFs, enter your OpenAI API Key, and start asking questions!")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Try reading key from Streamlit Secrets first; fallback to user text input
    secret_key = st.secrets.get("OPENAI_API_KEY", "")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=secret_key,
        type="password",
        help="Get your key from https://platform.openai.com/api-keys"
    )
    
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Session States ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# --- Helper: Build Vectorstore & RAG Chain ---
def initialize_rag(files, api_key):
    docs = []
    
    # Load PDFs using temporary files
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            loader = PyPDFLoader(tmp_file.name)
            docs.extend(loader.load())
            os.remove(tmp_file.name)

    if not docs:
        return None

    # 1. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # 2. Embed and store in in-memory FAISS Vector Store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. LLM Setup
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=api_key
    )

    # 4. Contextualize query prompt (Memory re-writer)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context "
        "in the chat history, formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just reformulate it if "
        "needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 5. Question Answering prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following retrieved context to answer the question. "
        "If you do not know the answer, say that you do not know. "
        "Keep answers clear and factual.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Combined RAG Chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- Document Processing Logic ---
current_file_names = [f.name for f in uploaded_files] if uploaded_files else []

if uploaded_files and openai_api_key:
    if st.session_state.processed_files != current_file_names:
        with st.spinner("Processing documents and generating embeddings..."):
            st.session_state.rag_chain = initialize_rag(uploaded_files, openai_api_key)
            st.session_state.processed_files = current_file_names
            st.success("✅ Documents indexed! You can now start chatting.")
elif not openai_api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar.")
elif not uploaded_files:
    st.info("👈 Please upload at least one PDF file in the sidebar.")

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & Response Generation ---
if user_input := st.chat_input("Ask a question about your uploaded documents..."):
    if not openai_api_key:
        st.warning("Please provide your OpenAI API key in the sidebar.")
    elif st.session_state.rag_chain is None:
        st.warning("Please upload at least one PDF and ensure it has been indexed.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Convert Streamlit history to LangChain tuple format
        chat_history = [
            (
                ("human", m["content"])
                if m["role"] == "user"
                else ("assistant", m["content"])
            )
            for m in st.session_state.messages[:-1]
        ]

        # Generate LLM response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(
                        {"input": user_input, "chat_history": chat_history}
                    )
                    answer = response.get("answer", "No answer generated.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
