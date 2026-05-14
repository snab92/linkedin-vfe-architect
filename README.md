# ✉️ LinkedIn VFE Architect

> Agente Conversacional con RAG + Web Search para generar Value Forward Emails de alta conversión en LinkedIn.
> **Proyecto Final — Maestría en IA**

---

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit UI                      │
│  Sidebar: Tono de Voz | Main: Formulario VFE        │
└────────────────────┬────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   LangChain Agent   │
          │  CONVERSATIONAL_    │
          │  REACT_DESCRIPTION  │
          └──────┬──────┬───────┘
                 │      │
    ┌────────────▼──┐  ┌▼──────────────────┐
    │  Tool 1: RAG  │  │  Tool 2: Search   │
    │  FAISS +      │  │  DuckDuckGo       │
    │  Embeddings   │  │  (tiempo real)    │
    └───────┬───────┘  └───────────────────┘
            │
    ┌───────▼────────┐
    │ data/knowledge │
    │  *.pdf / *.txt │
    └────────────────┘
```

---

## 🚀 Instalación Local

### 1. Clonar y preparar entorno

```bash
git clone <tu-repo>
cd linkedin-vfe-architect
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configurar API Key

Crea el archivo `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "tu_google_api_key_aqui"
```

> Obtén tu API Key en: https://aistudio.google.com/app/apikey

### 3. Agregar documentos al RAG (opcional pero recomendado)

```
data/
└── knowledge/
    ├── estructuras_vfe.txt      ← ya incluido
    ├── outreach_linkedin.txt    ← ya incluido
    ├── casos_de_exito.pdf       ← agrega los tuyos
    └── guia_sectores.txt        ← agrega los tuyos
```

### 4. Ejecutar

```bash
streamlit run app.py
```

---

## ☁️ Despliegue en Streamlit Cloud

1. Sube tu repositorio a GitHub (**sin** el archivo `secrets.toml`)
2. Ve a [share.streamlit.io](https://share.streamlit.io) → New app
3. Selecciona tu repo y `app.py` como archivo principal
4. Ve a **Settings → Secrets** y pega:

```toml
GOOGLE_API_KEY = "tu_google_api_key_aqui"
```

5. Deploy ✅

---

## 📁 Estructura de la Carpeta de Conocimiento

Para que el RAG sea realmente poderoso, organiza tus documentos así:

```
data/knowledge/
│
├── estructuras_vfe.txt
│   → Reglas de estructura por tamaño de audiencia
│   → Componentes del VFE (Gancho, Cuerpo, CTA)
│   → Errores fatales a evitar
│
├── outreach_linkedin.txt
│   → Psicología del prospecto
│   → Estrategia de secuencia (4 pasos)
│   → Fórmulas probadas (PAS, AIDA, Insight Provocador)
│
├── casos_de_exito.txt (o .pdf)
│   → Ejemplos reales de VFEs que generaron respuesta
│   → Incluye el contexto, el email y el resultado
│
├── sectores_latam.txt
│   → Desafíos específicos por industria (fintech, retail, logística…)
│   → Lenguaje y terminología de cada sector
│
└── mi_empresa.txt
    → Propuesta de valor de tu empresa
    → Casos de uso principales
    → Diferenciadores vs competencia
```

### 💡 El "Toque Maestro"

Cuanto más específicos sean tus documentos, mejor será el agente. Ejemplo de contenido de alto impacto en `sectores_latam.txt`:

```
SECTOR: E-COMMERCE / RETAIL LATAM 2024
Desafío #1: El 68% de los retailers LATAM reportan que su mayor problema es...
Terminología clave que usan los VP de Operaciones: "fulfillment", "última milla", "NPS post-entrega"
Gancho que funciona: "Con el crecimiento del D2C en LATAM, el...
```

---

## 🎭 Tonos de Voz Disponibles

| Tono | Cuándo usarlo |
|------|---------------|
| **Ejecutivo** | C-Level, decisores de alto nivel, contextos formales |
| **Empático** | Prospectos con desafíos visibles, post-crisis empresarial |
| **Persuasivo** | Ciclos de venta activos, seguimientos, deals calientes |

---

## 🔧 Variables de Entorno Soportadas

| Variable | Fuente | Descripción |
|----------|--------|-------------|
| `GOOGLE_API_KEY` | `st.secrets` o `.env` | API Key de Google AI Studio |

---

## 📦 Stack Tecnológico

| Componente | Tecnología |
|------------|------------|
| LLM | Gemini 1.5 Flash |
| Embeddings | Google Embedding-001 |
| Vector Store | FAISS (CPU) |
| Agente | LangChain CONVERSATIONAL_REACT |
| Memoria | ConversationBufferMemory |
| Web Search | DuckDuckGo |
| UI | Streamlit |

---

## 🎓 Notas para la Defensa de Maestría

**Decisiones de arquitectura a destacar:**

1. **FAISS sobre Chroma/Pinecone**: Elección deliberada para despliegue sin dependencias externas. Todo corre localmente o en Streamlit Cloud sin bases de datos adicionales.

2. **CONVERSATIONAL_REACT vs ReAct simple**: El tipo `CONVERSATIONAL_REACT_DESCRIPTION` mantiene memoria entre turnos, permitiendo refinamientos iterativos del VFE en la misma sesión.

3. **Type hints + Docstrings Google**: Estándar de ingeniería de software para mantenibilidad y documentación automática.

4. **`@st.cache_resource` en vectorstore**: El índice FAISS se construye una sola vez por sesión, evitando re-indexar en cada interacción del usuario.

5. **Separación de herramientas**: RAG y Search son herramientas independientes que el agente combina según el contexto, demostrando razonamiento multi-paso real.
