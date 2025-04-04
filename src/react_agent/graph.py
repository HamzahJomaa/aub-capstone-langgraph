# === Standard Library ===
import os
import sys
from typing import Literal
from typing_extensions import TypedDict

# === Third-Party Libraries ===
import pandas as pd

# === LangChain Core ===
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field

# === LangChain Components ===
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.tools import BaseTool, StructuredTool, Tool

# === LangGraph ===
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph_supervisor import create_supervisor

# === LangChain Neo4j ===
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain, Neo4jChatMessageHistory


# === Language Model Setup ===
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)


# === Graph Configuration ===
graph = Neo4jGraph(
    url="bolt://207.154.238.179:7687",
    username="neo4j",
    password="securePass123",
    database="englishdata",
    refresh_schema=True
)


# === Prompt Template ===
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query to retrieve data from a graph database.

### **Instructions:**
1. **Understand the Schema:**
   - Identify node types and relationships relevant to the user’s query.
   - Use predefined relationship mappings when available.

2. **Relationship Formatting Rules:**
   - Relationship names must be written in UPPERCASE with underscores (`_`) instead of spaces.
   - If a relationship is not in the mapping, **enclose it in backticks (`) to avoid syntax errors**.

3. **Query Construction:**
   - When filtering node properties, use `CONTAINS` instead of exact equality (`=`).
   - Use **comparison operators** for numeric values.
   - Ensure relationships are enclosed correctly.
   - Don't use relations keep it [relation]

4. **Strict Output Format:**
   - **Return only the Cypher query**—no explanations or additional text.
   - The query must pass `EXPLAIN` before execution to ensure syntax validity.
   - ALL NODES AND RELATIONS ARE in ENGLISH.

- Use schema-defined relationships to correctly structure the query.
- Try to use the word in more than one form to broaden the search.
- Avoid labels/relations not present in the schema.
- Don’t use "and" for name-based searches.

---

### **Schema:**
{schema}

### **User Query:**
{question}
"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# === Chain ===

graph_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
    validate_cypher=True,
    verbose=True,
    return_direct=True,
    return_intermediate_steps=True,
)


# === Tools ===

@tool
def knowledge_graph_retreivel(query: str) -> str:
    """Gets the KG results"""
    result = graph_chain.invoke(query)
    return result

class evaluation(BaseModel):
    result: bool = Field(description="The setup of the joke")

@tool
def check_prompt(query: str, results: list) -> bool:
    """
    Checks if the results are relevant to the query.
    """
    chain = ChatPromptTemplate([
        ("system", """
        You are a helpful assistant. You will be given a user query and a set of results.
        Determine if the results are relevant to the query. If they are relevant, respond with "True", otherwise respond "False".

        Query: {query}
        Results: {results}

        Answer with True or False.
        """),
    ]) | llm.with_structured_output(evaluation)

    result = chain.invoke({"query": query, "results": results})
    return result


# === Agents ===

knowledge_graph_agent = create_react_agent(
    model=llm,
    tools=[knowledge_graph_retreivel],
    name="knowledge_graph_agent",
    prompt=(
        "You are an agent whose job is to retrieve data from the knowledge graph. "
        "After retrieving the data, you must check whether the results are relevant "
        "to the query."
    )
)

@tool
def get_more_info_tool(entities: list) -> str:
    """Tool responsible for getting more entities"""
    results = graph_chain.invoke({"query": entities})
    return results


@tool
def start_reasoning_tool(data: str) -> str:
    """ Tool required for analyze data based on first order logic, This should always run"""
    print("Specialized Reasoning Started\n {}".format(data))
    prompt = PromptTemplate.from_template("""

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


    Input:
    {input}

    """)
    chain = prompt | llm
    results = chain.invoke({"input": data})
    return results


reasoning_agent = create_react_agent(
    model=llm,
    tools=[start_reasoning_tool, get_more_info_tool],
    name="reasoning_agent",
    prompt="You are an expert agent responsible only to navigate between tools. Always start by start_reasoning_tool. Always retrieve the reasoning as is to the supervisor"
)



# === Supervisor Setup ===

members = ["knowledge_graph_agent", "reasoning_agent"]
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
3. Pass the data to **reasoning_agent** for inference.
4. Combine both outputs and respond to the user.

Always make sure reasoning is applied when needed.
"""

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request," + multi_agent_prompt
)

# === Workflow Compilation ===

workflow = create_supervisor(
    [knowledge_graph_agent, reasoning_agent],
    model=llm,
    prompt=system_prompt
)

graph = workflow.compile()
