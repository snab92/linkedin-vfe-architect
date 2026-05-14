"""
LinkedIn VFE Architect — Agente Conversacional para Generación de Value Forward Emails.

Arquitectura:
    - RAG Engine: FAISS sobre documentos locales en data/knowledge/
    - Web Search: DuckDuckGoSearchRun para noticias en tiempo real
    - LLM: Gemini 1.5 Flash vía langchain-google-genai
    - Agente: CONVERSATIONAL_REACT_DESCRIPTION con ConversationBufferMemory
    - UI: Streamlit con barra lateral de configuración y transparencia de razonamiento

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página (debe ser lo primero de Streamlit)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn VFE Architect",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Imports que requieren la API key (cargados tras set_page_config)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.agents import initialize_agent, AgentType, Tool
    from langchain.chains import RetrievalQA
except ImportError as exc:
    st.error(
        f"❌ Dependencia faltante: {exc}. "
        "Asegúrate de instalar todas las librerías con: `pip install -r requirements.txt`"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
KNOWLEDGE_DIR = Path("data/knowledge")
CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 150
GEMINI_MODEL = "gemini-1.5-flash"

TONE_PROMPTS: dict[str, str] = {
    "Ejecutivo": (
        "Utiliza un lenguaje directo, orientado a resultados y métricas de negocio. "
        "Sin rodeos. Cada frase debe aportar valor concreto."
    ),
    "Empático": (
        "Muestra genuino interés por los desafíos del prospecto. "
        "Usa lenguaje cálido, personal y comprensivo antes de presentar cualquier propuesta."
    ),
    "Persuasivo": (
        "Aplica principios de persuasión (escasez, prueba social, autoridad). "
        "Crea urgencia sin ser agresivo. Termina siempre con un CTA irresistible."
    ),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de inicialización
# ──────────────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    """Obtiene la GOOGLE_API_KEY desde st.secrets o variables de entorno.

    Returns:
        La API key como string.

    Raises:
        ValueError: Si la clave no se encuentra en ninguna fuente.
    """
    key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError(
            "GOOGLE_API_KEY no encontrada. "
            "Agrégala en .streamlit/secrets.toml (local) o en Settings > Secrets (Streamlit Cloud)."
        )
    return key


def load_documents(knowledge_dir: Path) -> list:
    """Carga todos los documentos PDF y TXT desde el directorio de conocimiento.

    Args:
        knowledge_dir: Ruta al directorio que contiene los archivos de conocimiento.

    Returns:
        Lista de documentos LangChain listos para indexar.
    """
    docs = []
    pdf_files = glob.glob(str(knowledge_dir / "**/*.pdf"), recursive=True)
    txt_files = glob.glob(str(knowledge_dir / "**/*.txt"), recursive=True)

    for path in pdf_files:
        try:
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        except Exception as exc:
            st.warning(f"⚠️ No se pudo cargar '{path}': {exc}")

    for path in txt_files:
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        except Exception as exc:
            st.warning(f"⚠️ No se pudo cargar '{path}': {exc}")

    return docs


@st.cache_resource(show_spinner="🧠 Indexando base de conocimientos con Chroma…")
def build_vectorstore(knowledge_dir: Path, api_key: str) -> Optional[Chroma]:
    """Construye y cachea el vectorstore Chroma a partir de los documentos locales.

    Args:
        knowledge_dir: Directorio con archivos de conocimiento.
        api_key: Clave de la API de Google para generar embeddings.

    Returns:
        Instancia de FAISS o None si no hay documentos.
    """
    if not knowledge_dir.exists():
        st.info(
            f"ℹ️ Carpeta `{knowledge_dir}` no encontrada. "
            "El agente operará sin RAG. Crea la carpeta y agrega documentos para activarlo."
        )
        return None

    try:
        docs = load_documents(knowledge_dir)
    except Exception as exc:
        st.error(f"❌ Error al leer documentos: {exc}")
        return None

    if not docs:
        st.info("ℹ️ No se encontraron documentos en la carpeta de conocimiento.")
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key,
        )
        vectorstore = Chroma.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as exc:
        st.error(f"❌ Error al construir el índice FAISS: {exc}")
        return None


def build_agent(
    api_key: str,
    vectorstore: Optional[FAISS],
    tone: str,
    memory: ConversationBufferMemory,
) -> object:
    """Construye el agente conversacional LangChain con las herramientas disponibles.

    Args:
        api_key: Clave de la API de Google.
        vectorstore: Índice FAISS con el conocimiento interno (puede ser None).
        tone: Nombre del tono de voz seleccionado por el usuario.
        memory: Objeto de memoria conversacional compartido.

    Returns:
        Agente LangChain listo para invocar.
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )
    except Exception as exc:
        st.error(f"❌ Error al inicializar Gemini: {exc}")
        st.stop()

    tools: list[Tool] = []

    # Tool 1: RAG sobre base de conocimientos interna
    if vectorstore:
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
            )
            tools.append(
                Tool(
                    name="KnowledgeBase_LinkedIn",
                    func=rag_chain.run,
                    description=(
                        "Consulta la base de conocimientos interna sobre estrategias de LinkedIn, "
                        "estructuras de VFE, mejores prácticas de outreach y segmentación de audiencia. "
                        "Úsala cuando necesites marcos estratégicos o estructuras de email probadas."
                    ),
                )
            )
        except Exception as exc:
            st.warning(f"⚠️ RAG no disponible: {exc}")

    # Tool 2: DuckDuckGo para noticias en tiempo real
    try:
        search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="WebSearch_Noticias",
                func=search.run,
                description=(
                    "Busca noticias recientes, comunicados de prensa, cambios de liderazgo "
                    "y eventos relevantes sobre la empresa objetivo. "
                    "Úsala para encontrar ganchos de conversación actuales y relevantes."
                ),
            )
        )
    except Exception as exc:
        st.warning(f"⚠️ Búsqueda web no disponible: {exc}")

    if not tools:
        st.error("❌ No hay herramientas disponibles para el agente.")
        st.stop()

    tone_instruction = TONE_PROMPTS.get(tone, "")
    system_prefix = f"""Eres el LinkedIn VFE Architect, un experto en redacción de Value Forward Emails (VFE) para LinkedIn.

TONO DE VOZ ACTIVO: {tone}
INSTRUCCIÓN DE TONO: {tone_instruction}

Tu misión es crear correos de valor extraordinarios combinando:
1. Investigación de mercado en tiempo real de la empresa objetivo
2. Mejores prácticas internas de outreach en LinkedIn

PROCESO OBLIGATORIO para cada VFE:
1. Usa WebSearch_Noticias para investigar noticias recientes de la empresa objetivo
2. Usa KnowledgeBase_LinkedIn para obtener la estructura VFE adecuada según el tamaño de audiencia
3. Sintetiza ambas fuentes para crear un email personalizado, relevante y accionable

Responde siempre en español. Muestra tu razonamiento paso a paso."""

    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            agent_kwargs={"prefix": system_prefix},
        )
        return agent
    except Exception as exc:
        st.error(f"❌ Error al inicializar el agente: {exc}")
        st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Estado de sesión
# ──────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Inicializa las variables de estado de la sesión de Streamlit."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "current_tone" not in st.session_state:
        st.session_state.current_tone = "Ejecutivo"


def reset_memory() -> None:
    """Reinicia la memoria conversacional y el historial del chat."""
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    st.session_state.agent = None


# ──────────────────────────────────────────────────────────────────────────────
# UI Principal
# ──────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str]:
    """Renderiza la barra lateral con configuración del agente.

    Returns:
        Tupla (tono_seleccionado, api_key).
    """
    with st.sidebar:
        st.title("⚙️ Configuración")
        st.divider()

        st.subheader("🎭 Tono de Voz")
        tone = st.selectbox(
            "Selecciona el estilo del email:",
            options=list(TONE_PROMPTS.keys()),
            index=0,
            help="El tono afecta la estructura y el lenguaje del VFE generado.",
        )

        if tone != st.session_state.get("current_tone"):
            st.session_state.current_tone = tone
            st.session_state.agent = None  # Forzar reconstrucción del agente

        st.divider()

        st.subheader("🔑 API Key")
        manual_key = st.text_input(
            "GOOGLE_API_KEY (opcional):",
            type="password",
            help="Solo si no usas st.secrets o variables de entorno.",
        )

        st.divider()

        if st.button("🔄 Reiniciar Memoria", use_container_width=True):
            reset_memory()
            st.success("✅ Memoria reiniciada.")
            st.rerun()

        st.divider()
        st.caption(
            "📁 Coloca tus documentos en `data/knowledge/` "
            "para activar el motor RAG."
        )

        api_key = manual_key or ""
        return tone, api_key


def render_vfe_form() -> Optional[str]:
    """Renderiza el formulario de parámetros del VFE.

    Returns:
        Prompt construido para el agente, o None si el formulario está incompleto.
    """
    st.subheader("📋 Parámetros del VFE")

    col1, col2 = st.columns(2)
    with col1:
        empresa = st.text_input(
            "🏢 Empresa Objetivo",
            placeholder="Ej: Mercado Libre",
        )
        cargo = st.text_input(
            "👤 Cargo del Prospecto",
            placeholder="Ej: VP de Operaciones",
        )
    with col2:
        objetivo = st.text_area(
            "🎯 Objetivo del VFE",
            placeholder="Ej: Agendar una demo de nuestra solución de automatización logística",
            height=100,
        )
        audiencia = st.selectbox(
            "👥 Tamaño de Audiencia",
            options=[
                "Individual (1 persona)",
                "Pequeña (< 50 personas)",
                "Mediana (50–500 personas)",
                "Grande (> 500 personas)",
            ],
        )

    if st.button("✉️ Generar VFE", type="primary", use_container_width=True):
        if not empresa or not cargo or not objetivo:
            st.warning("⚠️ Por favor completa todos los campos antes de generar.")
            return None

        prompt = (
            f"Genera un Value Forward Email (VFE) para LinkedIn con los siguientes parámetros:\n\n"
            f"- **Empresa objetivo**: {empresa}\n"
            f"- **Cargo del prospecto**: {cargo}\n"
            f"- **Objetivo del VFE**: {objetivo}\n"
            f"- **Tamaño de audiencia**: {audiencia}\n\n"
            f"Sigue tu proceso: investiga noticias recientes de {empresa}, "
            f"consulta la base de conocimientos para la estructura adecuada "
            f"según el tamaño de audiencia '{audiencia}', y redacta el VFE completo."
        )
        return prompt

    return None


def main() -> None:
    """Función principal que orquesta toda la aplicación Streamlit."""
    init_session_state()

    # Header
    st.title("✉️ LinkedIn VFE Architect")
    st.caption(
        "Genera Value Forward Emails con IA: RAG sobre mejores prácticas + "
        "investigación en tiempo real de tu empresa objetivo."
    )
    st.divider()

    # Sidebar
    tone, manual_key = render_sidebar()

    # Obtener API key
    try:
        api_key = manual_key or get_api_key()
    except ValueError as exc:
        st.error(f"🔑 {exc}")
        st.stop()

    os.environ["GOOGLE_API_KEY"] = api_key

    # Construir vectorstore (cacheado)
    vectorstore = build_vectorstore(KNOWLEDGE_DIR, api_key)

    # Construir agente si no existe o si el tono cambió
    if st.session_state.agent is None:
        with st.spinner("🤖 Inicializando agente…"):
            st.session_state.agent = build_agent(
                api_key=api_key,
                vectorstore=vectorstore,
                tone=tone,
                memory=st.session_state.memory,
            )

    # Mostrar estado del sistema
    col_rag, col_search, col_tone = st.columns(3)
    with col_rag:
        if vectorstore:
            st.success("✅ RAG activo")
        else:
            st.warning("⚠️ RAG inactivo (sin documentos)")
    with col_search:
        st.success("✅ Web Search activo")
    with col_tone:
        st.info(f"🎭 Tono: **{tone}**")

    st.divider()

    # Formulario VFE
    prompt = render_vfe_form()

    st.divider()

    # Chat interface
    st.subheader("💬 Conversación con el Agente")

    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("thoughts"):
                with st.expander("🔍 Ver razonamiento del agente"):
                    st.text(message["thoughts"])

    # Procesar prompt del formulario
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Investigando y redactando tu VFE…"):
                try:
                    # Capturar el output verbose del agente para mostrar el razonamiento
                    import io
                    from contextlib import redirect_stdout

                    thought_buffer = io.StringIO()
                    with redirect_stdout(thought_buffer):
                        response = st.session_state.agent.run(prompt)

                    thoughts = thought_buffer.getvalue()

                    st.markdown(response)

                    if thoughts.strip():
                        with st.expander("🔍 Ver razonamiento del agente"):
                            st.text(thoughts)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "thoughts": thoughts,
                        }
                    )

                except Exception as exc:
                    error_msg = f"❌ Error al generar respuesta: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )

    # Chat libre
    if user_input := st.chat_input("Escribe un seguimiento o ajuste para el VFE…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("💭 Pensando…"):
                try:
                    import io
                    from contextlib import redirect_stdout

                    thought_buffer = io.StringIO()
                    with redirect_stdout(thought_buffer):
                        response = st.session_state.agent.run(user_input)

                    thoughts = thought_buffer.getvalue()
                    st.markdown(response)

                    if thoughts.strip():
                        with st.expander("🔍 Ver razonamiento del agente"):
                            st.text(thoughts)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "thoughts": thoughts,
                        }
                    )

                except Exception as exc:
                    error_msg = f"❌ Error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
