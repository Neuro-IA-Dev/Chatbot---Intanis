import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
import csv
from datetime import datetime

# Configuración inicial
st.set_page_config(page_title="Asistente de Reglamentos Intanis", page_icon="🧑‍💼")

# Logo y título
st.image("logo_Intanis.png", width=180)
st.title("🧑‍💼 Asistente de Reglamentos Intanis")
st.markdown("Haz preguntas sobre los reglamentos internos de la empresa (Conducta Empresarial, Seguridad de la Información y RIOHS).")

# Función para registrar logs
def log_interaction(user_question, response):
    with open("chat_logs.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), user_question, response])

# Cargar y procesar los 3 PDFs
def load_and_process_pdfs():
    pdf_paths = [
        "ConductaEmpresarial.PDF",
        "PoliticaSeguridadInformacion.PDF",
        "RIOHS.PDF"
    ]
    pages = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db

# Cargar base de datos (vector store)
db = load_and_process_pdfs()
# Input de usuario
query = st.text_input("✍️ Escribe tu pregunta aquí")

if "formulario de vacaciones" in query.lower():
    st.write("Aquí tienes el formulario de vacaciones:")
    with open("formulario_vacaciones.docx", "rb") as f:
        st.download_button(
            label="📄 Descargar Formulario de Vacaciones",
            data=f,
            file_name="formulario_vacaciones.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    # Y evitar seguir procesando
    log_interaction(query, "Formulario entregado.")
    st.stop()


# Procesamiento
if query:
    docs = db.similarity_search(query)
    llm = ChatOpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    st.write(response)
    if "formulario de vacaciones" in query.lower():
        st.info("Puedes descargar el formulario de vacaciones aquí:")
        with open("formulario_vacaciones.docx", "rb") as f:
            st.download_button(
                label="📄 Descargar Formulario",
                data=f,
                file_name="formulario_vacaciones.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    else:
        result = qa({"query": query})
        response = result["result"]
        source_docs = result.get("source_documents", [])

    # Guardar log
        log_interaction(query, response)
        st.markdown(f"**🧠 Respuesta:** {response}")

        if source_docs:
            st.markdown("📄 **Fuente(s):**")
            for i, doc in enumerate(source_docs):
                fuente = doc.metadata.get("source", "desconocida")
                st.write(f"{i+1}. {fuente}")

   

    # Botón para descargar logs
    with open("chat_logs.csv", "r", encoding="utf-8") as f:
        st.download_button("⬇️ Descargar logs", f, file_name="chat_logs.csv", mime="text/csv")
keywords = ["formulario", "vacaciones", "permiso", "licencia", "descanso"]

if any(word in query.lower() for word in keywords):
    st.success("Puedes descargar el formulario de vacaciones o permisos laborales aquí:")
    with open("formulario_vacaciones.docx", "rb") as f:
        st.download_button(
            label="⬇️ Descargar Formulario de Permiso (Word)",
            data=f,
            file_name="formulario_vacaciones.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
