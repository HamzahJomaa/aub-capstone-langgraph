from agents.KnowledgeGraphAgent import KnowledgeGraphAgent
from agents.RDFOrchestratorAgent import RDFOrchestratorAgent
from agents.ReasoningAgent import ReasoningAgent
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI

kg_agent = KnowledgeGraphAgent
rdf_agent = RDFOrchestratorAgent
reasoning_agent = ReasoningAgent

members = ["KnowledgeGraphAgent", "RDFOrchestratorAgent", "ReasoningAgent"]
options = members + ["FINISH"]

multi_agent_prompt = """
You are the **KnowledgeGraphAgent**, responsible for retrieving structured data and entities from a knowledge graph in response to a user query.

Your primary responsibilities:
1. Analyze the user's question and extract relevant **entities**, **concepts**, or **topics**.
2. Query the knowledge graph to retrieve structured data, including:
   - Direct relationships between entities
   - Properties or attributes of the entities
   - Class or type information if available
3. Present the extracted information in a clear and structured format that highlights entities and their connections.

Rules:
- Focus only on what is explicitly available in the knowledge graph.
- Do not generate new knowledge or perform reasoning or inference.
- Avoid speculation or drawing conclusions; stick strictly to what the graph provides.
- Your output should be concise, factual, and formatted for easy interpretation by downstream processes.

Your output will be used by other agents for enrichment and reasoning, so ensure all relevant data is included and clearly organized.
"""



system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    f"following agents: {', '.join(members)}. Given the following user request:\n\n"
    + multi_agent_prompt.strip()
)


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Create supervisor workflow
workflow = create_supervisor(
    [kg_agent],
    model=llm,
    prompt=system_prompt
)

graph = workflow.compile()