import csv
from datetime import datetime
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Asistente Reglamentario Intanis")
st.image("logo_Intanis.png", width=180)
st.title("🧑‍💼 Asistente de Reglamentos Intanis")
st.markdown("Haz preguntas sobre los reglamentos internos de la empresa (Conducta Empresarial, Seguridad de la Información y RIOHS).")

# Cargar clave de OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
def log_interaction(user_question, response):
    with open("chat_logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), user_question, response])
@st.cache_resource
def load_chain():
    loaders = [
        PyPDFLoader("ConductaEmpresarial.PDF"),
        PyPDFLoader("PoliticaSeguridadInformacion.PDF"),
        PyPDFLoader("RIOHS.PDF")
    ]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    custom_prompt = PromptTemplate.from_template(
        "Eres un asistente corporativo experto en regulaciones internas de la empresa Intanis. "
        "Responde con base exclusivamente en los reglamentos disponibles. "
        "Si la información no está contenida en los documentos, responde: 'No tengo información suficiente para responder esa consulta.'\n\n"
        "Pregunta: {question}\n\n"
        "Contexto:\n{context}\n\n"
        "Respuesta:"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

qa_chain = load_chain()

query = st.text_input("Escribe tu pregunta:")
if query:
    response = qa_chain.run(query)
    st.markdown("### ✅ Respuesta:")
    st.write(response)
    log_interaction(query, response)
