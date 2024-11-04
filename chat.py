import streamlit as st
from agent import Agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tool import create_retriever_tool_from_vectorstore, create_get_wine_info_tool

persist_directory = "./chroma_db"

try:
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    tools = [create_retriever_tool_from_vectorstore(vectorstore), create_get_wine_info_tool()]
except Exception as e:
    st.write(f"Error creating vectorstore: {e}")
    tools = None

prompt = """
Eres un sommelier virtual útil y servicial de una bodega de vinos, llamada Chañarmuyo. 
    Chañarmuyo es una bodega ubicada en La Rioja, Argentina, que combina la producción de vinos de alta calidad con una experiencia turística integrada. Fundada en 1920, la empresa ha evolucionado de ser una bodega tradicional a un destino enoturístico completo.
    Chañarmuyo cuenta con una bodega, viñedos y un hotel boutique en Mendoza, ofreciendo una experiencia enoturística integrada.
    Tu objetivo es proporcionar información precisa sobre los distintos sabores y variedades de vinos, tal como lo haría un sommelier, y ayudar a generar itinerarios para una ruta del vino, basados en las preferencias del cliente, el clima, la temporada y la disponibilidad.
    Si el cliente hace una consulta sobre la Casa de Huéspedes, alojamiento o reservas en Chañarmuyo tenes que compartirle este link: https://chanarmuyo.reservadirecto.com/?result=login_success
    Si no tienes la información solicitada, indícalo claramente y ofrece alternativas si es posible. 
    Para consultas sobre vinos y maridajes, debes utilizar siempre la tool. En el caso de que el usuario consultara por la documentación guardada en la carpeta docs no puedes brindarle información, sino decirle que no tienes acceso.
"""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="🍷 ¡Bienvenido a Bodega Amigo, tu sommelier virtual de la bodega Chañarmuyo! \n\n"
                          "Estoy aquí para ayudarte a descubrir el mundo de nuestros vinos y brindarte una experiencia enoturística excepcional. "
                          "Puedes preguntarme sobre:\n\n"
                          "- Vinos: desde sabores y variedades hasta recomendaciones de maridaje.\n"
                          "- Rutas del vino: itinerarios personalizados según tus preferencias y la temporada.\n"
                          "- Alojamiento y reservas: información sobre la Casa de Huéspedes y cómo hacer una reserva.\n\n"
                          "Si tienes cualquier otra consulta, ¡no dudes en preguntar! ¿Por dónde te gustaría comenzar?")
    ]

# Crear el agente si las herramientas están disponibles
if tools:
    agent = Agent(model_type="openai", prompt=prompt, tools=tools)
else:
    st.write("No tools available")
    agent = Agent(model_type="openai", prompt=prompt)

# Estilos de la página
st.set_page_config(page_title="Chañarmuyo - Bodega Amigo", page_icon="🍷", layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background-image: url("https://images.pexels.com/photos/21393/pexels-photo.jpg?cs=srgb&dl=pexels-madebymath-21393.jpg&fm=jpgaa");
            background-size: cover;
            font-family: 'Lora', serif;
            color: #333333;
        }
        .title {
            font-size: 2.8rem;
            color: #8C212A;
            font-weight: bold;
            text-align: center;
            margin-top: 2rem;
        }
        .description {
            font-size: 1.2rem;
            color: #555555;
            text-align: center;
            margin-bottom: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la página principal
st.markdown('<h1 class="title">Bienvenido a Chañarmuyo</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="description">Explora nuestros vinos y experiencias enoturísticas en el corazón de La Rioja, Argentina.</p>',
    unsafe_allow_html=True,
)

# Sección de chat en la landing principal
for message in st.session_state.messages:
    if isinstance(message, (HumanMessage, AIMessage)) and message.content:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

# Entrada de usuario para el chat
if user_input := st.chat_input("Escribe tu consulta aquí..."):
    human_message = HumanMessage(content=user_input)
    st.session_state.messages.append(human_message)
    st.chat_message("user").markdown(user_input)

    # Invocar al agente para obtener la respuesta
    response_messages = agent.invoke(st.session_state.messages)
    st.session_state.messages = response_messages["messages"]

    # Mostrar la respuesta del asistente
    last_message = response_messages["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.content:
        st.chat_message("assistant").markdown(last_message.content)
