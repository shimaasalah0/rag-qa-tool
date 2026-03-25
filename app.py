import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Q&A Tool", page_icon="📄")
st.title("📄 Ask Your PDF")
st.write("Upload a PDF and ask it anything!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

@st.cache_resource
def create_vectorstore(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore, len(pages), len(chunks)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Processing your PDF..."):
        vectorstore, num_pages, num_chunks = create_vectorstore(tmp_path)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    st.success(f"✅ Ready! PDF has {num_pages} pages and {num_chunks} chunks.")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    # Chat input
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Thinking..."):
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant.
Answer the question based on the provided context.
If the context has partial information, use it to give the best possible answer.
Only say 'I don't know' if the context has absolutely no relevant information.

Context:
{context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )

            def get_response(question):
                docs = retriever.invoke(question)
                context = "\n\n".join([d.page_content for d in docs])
                chain = prompt | llm | StrOutputParser()
                return chain.invoke({
                    "context": context,
                    "question": question,
                    "chat_history": st.session_state.chat_history
                })

            answer = get_response(question)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))