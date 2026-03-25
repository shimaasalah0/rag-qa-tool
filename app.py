import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Q&A Tool", page_icon="📄")
st.title("📄 Ask Your PDF")
st.write("Upload a PDF and ask it anything!")

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
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    st.success(f"✅ Ready! PDF has {num_pages} pages and {num_chunks} chunks.")

    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
""")
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            answer = chain.invoke(question)

        st.markdown("### 📌 Answer:")
        st.write(answer)