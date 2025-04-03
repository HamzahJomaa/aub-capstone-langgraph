from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph_supervisor import create_supervisor

# Define evaluation schema
class Evaluation(BaseModel):
    result: bool = Field(description="The setup of the joke")

class State(MessagesState):
    next: str

# Initialize LLM and Graph
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
graph = Neo4jGraph(
    url="bolt://207.154.238.179:7687",
    username="neo4j",
    password="securePass123",
    database="englishdata"
)

# Prompt Template for Cypher generation
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
Task: Generate a Cypher query to retrieve data from a graph database.

### Instructions:
1. Understand the Schema and identify node types/relationships.
2. Use UPPERCASE for relationship names and backticks for unknown ones.
3. Use `CONTAINS` for property filters.
4. Format:
    MATCH (a)-[relation]-(b)
    WHERE a.id CONTAINS 'Trump'
    RETURN a, relation, b
5. Use schema-defined terms only. Avoid using unrecognized labels or "and" in name searches.

### Schema:
{schema}

### User Query:
{question}
"""
)

# Build Graph Chain
graph_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
    validate_cypher=True,
    verbose=False,
    return_direct=True,
    return_intermediate_steps=True
)

# Knowledge Graph Retrieval Tool
@tool
def knowledge_graph_retreivel(query: str) -> str:
    """Gets the KG results"""
    result = graph_chain.invoke(query)
    return result

# Relevance Check Tool
@tool
def check_prompt(query: str, results: list) -> bool:
    """Checks if KG results are relevant to the query."""
    prompt = ChatPromptTemplate([
        ("system", """
        You are a helpful assistant. Determine if results are relevant.
        Query: {query}
        Results: {results}
        Answer with True or False.
        """)
    ]) | llm.with_structured_output(Evaluation)
    return prompt.invoke({"query": query, "results": results})

# Define Agents
knowledge_graph_agent = create_react_agent(
    model=llm,
    tools=[knowledge_graph_retreivel, check_prompt],
    name="knowledge_graph_agent",
    prompt="""
You are an agent whose job is to retrieve data from the knowledge graph.
After retrieving the data, you must check whether the results are relevant to the query.
"""
)

# Logical Reasoning Prompt
logical_reasoning_prompt = """
Use first-order logic to analyze the knowledge graph results.
Steps:
1. Interpret entities and relationships.
2. Use predicates and logical reasoning.
3. Infer additional relationships.
4. Identify inconsistencies or gaps.
"""

@tool
def logical_reasoning_tool(query: str) -> str:
    """Check if there is need for logical reasoning"""
    return llm.invoke(query)

logical_reasoning = create_react_agent(
    model=llm,
    tools=[logical_reasoning_tool],
    name="reasoning_agent",
    prompt=logical_reasoning_prompt
)

# Supervisor Prompt
multi_agent_prompt = """
You are a Multi-Agent Supervisor managing two agents:
1. **knowledge_graph_agent** – Retrieves structured data from a knowledge graph.
2. **reasoning_agent** – Performs logical inference on retrieved data.

Workflow:
- Always invoke knowledge_graph_agent first.
- If reasoning (comparisons, causal, multi-step) is needed, invoke reasoning_agent.
- Return the final response integrating both agents' outputs.
"""

# Create Supervisor Workflow
workflow = create_supervisor(
    [knowledge_graph_agent, logical_reasoning],
    model=llm,
    prompt=multi_agent_prompt
)

# Compile workflow
graph = workflow.compile()
