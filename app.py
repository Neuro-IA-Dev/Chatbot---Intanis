import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import csv

# Configuración inicial
st.set_page_config(page_title="Asistente de Reglamentos Intanis")
st.image("logo_Intanis.png", width=180)
st.title("👨‍💼 Asistente de Reglamentos Intanis")
st.write("Haz preguntas sobre los reglamentos internos de la empresa (Conducta Empresarial, Seguridad de la Información y RIOHS).")

# Clave API
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Cargar y preparar documentos
pdfs = ["ConductaEmpresarial.PDF", "PoliticaSeguridadInformacion.PDF", "RIOHS.PDF"]
documents = []
for pdf in pdfs:
    loader = PyPDFLoader(pdf)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Inicializar cadena de QA
llm = ChatOpenAI(temperature=0)
qa = load_qa_chain(llm, chain_type="stuff")

# Función para registrar logs
def log_interaction(question, answer):
    with open("chat_logs.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([question, answer])

# Input del usuario
query = st.text_input("💬 Escribe tu pregunta aquí")

# Procesar pregunta
if query:
    keywords = ["formulario", "vacaciones", "permiso", "licencia", "descanso"]
    if any(kw in query.lower() for kw in keywords):
        st.success("Puedes descargar el formulario de vacaciones o permisos laborales aquí:")
        with open("formulario_vacaciones.docx", "rb") as f:
            st.download_button(
                label="📥 Descargar Formulario de Permiso (Word)",
                data=f,
                file_name="formulario_vacaciones.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        log_interaction(query, "Formulario entregado.")
        st.stop()

    # Realizar búsqueda y generar respuesta
    result = qa({"query": query})
    response = result["result"]
    source_docs = result.get("source_documents", [])

    # Mostrar respuesta
    st.markdown(f"**🔍 Respuesta:** {response}")

    # Mostrar fuentes
    if source_docs:
        st.markdown("📄 **Fuente(s):**")
        for i, doc in enumerate(source_docs):
            fuente = doc.metadata.get("source", "desconocida")
            st.write(f"{i+1}. {fuente}")

    # Registrar logs
    log_interaction(query, response)

# Opción para descargar los logs
with open("chat_logs.csv", "r", encoding="utf-8") as f:
    st.download_button("📥 Descargar logs", f, file_name="chat_logs.csv", mime="text/csv")
