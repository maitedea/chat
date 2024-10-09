from langchain.tools.retriever import create_retriever_tool

def create_retriever_tool_from_vectorstore(vectorstore):
    retriever = vectorstore.as_retriever()
    return create_retriever_tool(
        retriever,
        "retrieve_company_docs",
        "Search and return information about Chanarmuyo",
    )