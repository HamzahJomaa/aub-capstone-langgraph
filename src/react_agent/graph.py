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
You are a Multi-Agent Supervisor managing the following agents:

1. **KnowledgeGraphAgent** – Retrieves structured data and entities from a knowledge graph. This agent must always be called first.
2. **RDFOrchestratorAgent** – Enriches entities identified by the KnowledgeGraphAgent by retrieving or generating RDF data. Use this only if additional context about entities is needed.
3. **ReasoningAgent** – Performs logical reasoning using all available data (graph + RDF) to produce insights or conclusions.

Supervisor Rules:
1. Always start with **KnowledgeGraphAgent** to extract entities and their direct relationships from the graph.
2. After receiving entities, determine whether more information is needed to understand or reason about them.
   - If so, call **RDFOrchestratorAgent** to enrich those entities using RDF (either fetching or creating RDF representations).
   - Pass the list of extracted entities to RDFOrchestratorAgent as input.
3. After all data has been gathered, always call **ReasoningAgent**.
   - Pass both the original user query and the complete context (KG + RDF if available).
   - The agent should analyze relationships, resolve ambiguities, and produce higher-level insights.
4. After ReasoningAgent responds, synthesize and return the final answer using all outputs.

When to Use RDFOrchestratorAgent:
- The entities from the KG are ambiguous, incomplete, or insufficient for deep reasoning.
- The relationships alone don't provide enough context to answer the query.
- Background knowledge or rich semantic data is needed about specific entities.

When to Use ReasoningAgent:
- Always, after data is gathered.
- Especially useful for understanding relationships, performing comparisons, tracing causes, or multi-hop inference.

Execution Steps:
1. Call **KnowledgeGraphAgent** to retrieve entities and data.
2. If needed, enrich the entities via **RDFOrchestratorAgent**.
3. Perform reasoning using **ReasoningAgent**.
4. Return a complete and coherent answer using the combined insights.
"""


system_prompt = (
    "You are a supervisor tasked with managing a conversation between the "
    f"following agents: {', '.join(members)}. Given the following user request:\n\n"
    + multi_agent_prompt.strip()
)


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Create supervisor workflow
workflow = create_supervisor(
    [kg_agent, rdf_agent, reasoning_agent],
    model=llm,
    prompt=system_prompt
)

graph = workflow.compile()