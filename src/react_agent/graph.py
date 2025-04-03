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

logical_reasoning_prompt = """
You are a logical reasoning assistant. Your job is to analyze a knowledge graph using first-order logic.

Entities and relationships are given below:

{query}

Instructions:
1. Convert relationships into predicates using first-order logic.
2. Infer new facts logically from existing data.
3. Highlight inferred relationships.
4. Provide structured logic + human-readable insights.

"""

@tool
def logical_reasoning_tool(query: str) -> str:
    """Applies first-order logic to a knowledge graph to infer facts and provide analysis."""    
    prompt = PromptTemplate.from_template("""
    You have been provided with results from a knowledge graph, which contains structured data about entities and relationships. Using first-order logic, you are tasked with analyzing the graph and building an understanding of the underlying knowledge.

Steps:

1. Interpret the Knowledge Graph:
   - Examine the entities (nodes) and the relationships (edges) between them.
   - Represent the graph using predicates, constants, and variables.

2. Apply First-Order Logic:
   - Identify relevant facts, axioms, and rules based on the graph's structure.
   - Use first-order logic to infer new facts or relationships from the given data.

3. Build Knowledge:
   - Based on the knowledge graph's information and first-order logic reasoning, generate a coherent set of statements that describe the relationships and properties of the entities.
   - These statements should reflect the reasoning and should also include inferred relationships that were not explicitly stated in the graph.

4. Provide Insights:
   - Identify any potential inconsistencies or gaps in the knowledge represented by the graph.
   - Propose new connections or facts that can logically extend the graph.

Example Input Data (Knowledge Graph):

- Entities: `Person`, `Country`, `City`
- Relationships: 
  - `Person(x) -> LivesIn(x, y)` (A person lives in a city)
  - `City(y) -> LocatedIn(y, z)` (A city is located in a country)
  - `Country(z) -> HasCapital(z, c)` (A country has a capital city)

Query:
{query}
    """)
    chain = prompt | llm
    result = chain.invoke({"query": query})
    print(result)
    return result

reasoning_agent = create_react_agent(
    model=llm,
    tools=[logical_reasoning_tool],
    name="reasoning_agent",
    prompt=logical_reasoning_prompt
)




# Supervisor Prompt
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import MessagesState, END
from langgraph.types import Command


members = ["knowledge_graph_agent", "reasoning_agent"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

multi_agent_prompt = """

You are a Multi-Agent Supervisor managing two agents:
1. **knowledge_graph_
agent** – Retrieves data from a knowledge graph.
2. **reasoning_agent** – Performs logical reasoning using the retrieved data.

Supervisor Rules:
1. Always call the **knowledge_graph_agent** first to get structured data.
2. After that, always call the **reasoning_agent**, even if the user query seems simple.
   - Pass both the original user query and the retrieved data to the reasoning_agent.
3. The reasoning_agent should analyze the data, infer relationships, and generate deeper insights.
4. Once both agents have responded, construct a final answer using the outputs from both.
5. Each agent must return control to the supervisor immediately after responding.

When to Use Reasoning:
- Entity Relationships (e.g., "What is the relation between A and B?")
- Comparisons (e.g., "Which is larger, A or B?")
- Causal Inference (e.g., "What happens if A is removed from B?")
- Multi-Step Logic (e.g., "If A leads to B and B to C, what happens when A occurs?")

Execution Steps:
1. Get the user query.
2. Call **knowledge_graph_agent** to fetch relevant data.
3. Pass the query + data to **reasoning_agent** for inference.
4. Combine both outputs and respond to the user.

Always make sure reasoning is applied when needed.
"""



system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request," + multi_agent_prompt
    
)



# Create Supervisor Workflow
workflow = create_supervisor(
    [knowledge_graph_agent, reasoning_agent],
    model=llm,
    prompt=system_prompt
)

# Compile workflow
graph = workflow.compile()
