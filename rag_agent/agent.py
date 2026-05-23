from google.adk.agents import Agent

from .rag_tool import retrieve_context

# ------------------------------------
# Gemini Model
# ------------------------------------
MODEL = "gemini-2.5-flash"

# ------------------------------------
# RAG Tool
# ------------------------------------
def pdf_rag_tool(query: str):

    context = retrieve_context(query)

    return f"""
    Context:
    {context}

    User Question:
    {query}
    """

# ------------------------------------
# ADK Agent
# ------------------------------------
root_agent = Agent(

    name="vertex_pdf_rag_agent",

    model=MODEL,

    description="Enterprise PDF RAG Agent",

    instruction="""
    You are an intelligent PDF assistant.

    Use retrieved PDF context
    to answer accurately.
    """,

    tools=[pdf_rag_tool]
)