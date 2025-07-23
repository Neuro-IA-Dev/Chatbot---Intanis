import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv

# Configurar clave API
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Título e interfaz
st.image("logo_Intanis.png", width=180)
st.title("👨‍💼 Asistente de Reglamentos Intanis")
st.markdown("Haz preguntas sobre los reglamentos internos de la empresa (Conducta Empresarial, Seguridad de la Información y RIOHS).")

# Cargar y procesar documentos
pdf_files = ["ConductaEmpresarial.PDF", "PoliticaSeguridadInformacion.PDF", "RIOHS.PDF"]
all_text = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    for page in pages:
        page.metadata["source"] = pdf_file
    all_text.extend(pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_chunks = text_splitter.split_documents(all_text)

# Crear vectorstore
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(all_chunks, embeddings)

# Configurar cadena QA
llm = ChatOpenAI(temperature=0)
qa = load_qa_chain(llm, chain_type="stuff")

# Función para guardar logs
def log_interaction(query, response):
    with open("chat_logs.csv", "a", newline="", encoding="utf-8") as logfile:
        writer = csv.writer(logfile)
        writer.writerow([query, response])

# Entrada del usuario
query = st.chat_input("🌟 Escribe tu pregunta aquí")

if query:
    # Mostrar la pregunta del usuario
    st.markdown(f"📝 **Pregunta:** {query}")

    # Si es una solicitud de formulario
    if any(word in query.lower() for word in ["formulario", "vacaciones", "permiso", "licencia", "descanso"]):
        st.success("Puedes descargar el formulario de vacaciones o permisos laborales aquí:")
        with open("formulario_vacaciones.docx", "rb") as f:
            st.download_button(
                label="📄 Descargar Formulario de Permiso (Word)",
                data=f,
                file_name="formulario_vacaciones.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        log_interaction(query, "Formulario entregado.")
    
    else:
        # Procesamiento normal de consulta
        docs = db.similarity_search(query)
        result = qa.invoke({"input_documents": docs, "question": query})
        response = result["output_text"]
        source_docs = result.get("source_documents", [])

        st.markdown(f"**💭 Respuesta:** {response}")

        if source_docs:
            st.markdown("\n📄 **Fuente(s):**")
            for i, doc in enumerate(source_docs):
                fuente = doc.metadata.get("source", "desconocida")
                st.write(f"{i+1}. {fuente}")

        log_interaction(query, response)

# Mostrar botón para descargar logs siempre al final
with open("chat_logs.csv", "r", encoding="utf-8") as f:
    st.download_button(
        label="⬇️ Descargar logs",
        data=f,
        file_name="chat_logs.csv",
        mime="text/csv"
    )
