from langchain_neo4j import Neo4jGraph
import os
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.pydantic_v1 import BaseModel, Field
import os
import sys
import pandas as pd
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Literal

class evaluation(BaseModel):
    result: bool = Field(description="The setup of the joke")

class State(MessagesState):
    next: str



graph = Neo4jGraph(username=os.environ["NEO4J_USER"], database="english")

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
    ```cypher
        MATCH (a)-[relation]-(b)
        WHERE a.id CONTAINS 'Trump'
        RETURN a, relation, b
     ```
   - Use **comparison operators** for numeric values.
   - Ensure relationships are enclosed correctly:
     ```cypher
     MATCH (a:Person)-[relation]->(b)
     RETURN a, relation, b
     ```
   - Don't use relations keep it [relation]

4. **Strict Output Format:**
   - **Return only the Cypher query**—no explanations or additional text.
   - The query must pass `EXPLAIN` before execution to ensure syntax validity.
   - ALL NODES AND RELATIONS ARE in ENGLISH.

- Identify if the input mentions a **Person, Organization, or Location** based on keywords.
- If an entity is a company, business, university, or institution → classify it as `Organization`.
- If an entity is a city, country, or region → classify it as `Location`.
- If an entity is a person’s name or has attributes like age or nationality → classify it as `Person`.
- Use schema-defined relationships to correctly structure the query.
- Try to use the word more in more than one shape in order to search for it.
- Don't use labels for nodes and relationships that are not available.
- Dont use ands for name searching


---

### **Schema:**
{schema}

### **User Query:**
{question}

"""

CYPHER_GENERATION_PROMPT = PromptTemplate(input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE)
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

graph_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True,
    validate_cypher=True,
    verbose=False,
    return_direct=True,
    return_intermediate_steps=True,
)


@tool
def knowledge_graph_retreivel(query: str) -> str:
    """Gets the KG results"""
    result = graph_chain.invoke(query)
    print(query)
    # print(result["intermediate_steps"])
    return result


@tool
def check_prompt(query: str, results: list) -> bool:
    """
    Checks if the results are relevant to the query by invoking an LLM model.
    Returns True if relevant, False otherwise.
    """
    # Here we create a prompt that explains to the model what we want it to do.
    # We simply want the model to state if the given results are relevant or not.

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


knowledge_graph_agent = create_react_agent(
    model=llm,
    tools=[knowledge_graph_retreivel, check_prompt],
    name="knowledge_graph_agent",
    prompt=(
        "You are an agent whose job is to retrieve data from the knowledge graph. "
        "After retrieving the data, you must check whether the results are relevant "
        "to the query."
    )
)