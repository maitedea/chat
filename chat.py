import streamlit as st
from agent import Agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tool import create_retriever_tool_from_vectorstore


persist_directory = "./chroma_db" 

try:
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    tools = [create_retriever_tool_from_vectorstore(vectorstore)]
except Exception as e:
    st.write(f"Error creating vectorstore: {e}")
    tools = None

prompt = """
Eres un sommelier virtual útil y servicial de una bodega de vinos, llamada Chañarmuyo. 
    Chañarmuyo es una bodega ubicada en La Rioja, Argentina, que combina la producción de vinos de alta calidad con una experiencia turística integrada. Fundada en 1920, la empresa ha evolucionado de ser una bodega tradicional a un destino enoturístico completo.
    Chañarmuyo cuenta con una bodega, viñedos y un hotel boutique en Mendoza, ofreciendo una experiencia enoturística integrada.
    Tu objetivo es proporcionar información precisa sobre los distintos sabores y variedades de vinos, tal como lo haría un sommelier, y ayudar a generar itinerarios para una ruta del vino, basados en las preferencias del cliente, el clima, la temporada y la disponibilidad.
    Si no tienes la información solicitada, indícalo claramente y ofrece alternativas si es posible.
"""

if tools:
    agent = Agent(model_type="openai", prompt=prompt, tools=tools)
else:
    st.write("No tools available")
    agent = Agent(model_type="openai", prompt=prompt)


st.title("Agent Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display all previous chat messages
for message in st.session_state.messages:
    if isinstance(message, (HumanMessage, AIMessage)) and message.content:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)


# React to user input
if prompt := st.chat_input("User input"):
    # Create a HumanMessage and add it to chat history
    human_message = HumanMessage(content=prompt)
    st.session_state.messages.append(human_message)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Invoke the agent to get a list of AI messages, including potentially retrieved documents
    response_messages = agent.invoke(st.session_state.messages)

    # Update the session state with the new response
    st.session_state.messages = response_messages["messages"]

    # Display only the last AI message with content
    last_message = response_messages["messages"][-1]


    if isinstance(last_message, AIMessage) and last_message.content:
        st.chat_message("assistant").markdown(last_message.content)

