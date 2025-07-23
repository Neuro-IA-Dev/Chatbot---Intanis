import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pandas as pd

# Cargar variables de entorno
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---------- CONFIGURACIÓN ----------
st.set_page_config(page_title="Asistente Intanis", layout="wide")
st.image("logo_Intanis.png", width=180)
st.title("🧑‍💼 Asistente de Reglamentos Intanis")
st.markdown("Haz preguntas sobre los reglamentos internos de la empresa (Conducta Empresarial, Seguridad de la Información y RIOHS).")

# ---------- CARGA DE DOCUMENTOS ----------
def cargar_documentos():
    rutas = [
        "ConductaEmpresarial.PDF",
        "PoliticaSeguridadInformacion.PDF",
        "RIOHS.PDF"
    ]
    documentos = []
    for ruta in rutas:
        loader = PyPDFLoader(ruta)
        documentos.extend(loader.load())
    return documentos

# ---------- PROCESAMIENTO ----------
def construir_cadena(documentos):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(documentos)

    vectores = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectores, docs

# ---------- FUNCIONES DE LOG ----------
def log_interaction(query, response):
    log_path = "chat_logs.csv"
    nueva_fila = {"Pregunta": query, "Respuesta": response}

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([nueva_fila])], ignore_index=True)
    else:
        df = pd.DataFrame([nueva_fila])

    df.to_csv(log_path, index=False)

# ---------- BOTÓN DE DESCARGA DE FORMULARIO ----------
def mostrar_descarga_formulario():
    with open("formulario_vacaciones.docx", "rb") as f:
        st.download_button(
            label="📄 Descargar Formulario de Vacaciones",
            data=f,
            file_name="formulario_vacaciones.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

# ---------- EJECUCIÓN ----------
documentos = cargar_documentos()
vectores, docs = construir_cadena(documentos)

# Modelo LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
cadena = load_qa_chain(llm, chain_type="stuff")

# ---------- ENTRADA DE USUARIO ----------
query = st.text_input("📥 Escribe tu pregunta aquí:")

if query:
    if "formulario de vacaciones" in query.lower():
        st.success("Aquí tienes el formulario que solicitaste:")
        mostrar_descarga_formulario()
    else:
        documentos_similares = vectores.similarity_search(query)
        respuesta = cadena.run(input_documents=documentos_similares, question=query)

        st.write("✅", respuesta)
        st.write("📌 Fuente:", documentos_similares[0].metadata["source"])

        log_interaction(query, respuesta)

