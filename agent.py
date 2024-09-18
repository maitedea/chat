from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from LangGraph import build_graph


class Agent:
    def __init__(self, model_type="openai", prompt="Be a helpful assistant", tools=None):
        if model_type == "openai":
            self.model = ChatOpenAI(temperature=0, model_name="gpt-4o")
        elif model_type == "groq":
            self.model = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
        else:
            raise ValueError("Unsupported model type. Please choose 'openai' or 'groq'.")
        
        self.tools = tools if tools else []
        self.system_prompt = prompt
        
        # Build the graph using the function build_graph
        self.graph = build_graph(llm=self.model, prompt=self.system_prompt, tools=self.tools)

    def invoke(self, messages):
        # Initialize the state with the provided messages
        initial_state = {"messages": messages}
        
        # Run the graph synchronously and obtain the output
        graph_output = self.graph.invoke(initial_state)
        
        return graph_output
