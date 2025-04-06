from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI  # if using OpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
graph = Neo4jGraph(username="neo4j", database="englishdata", refresh_schema=True)


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


@tool
def KGRetriever(query: str) -> str:
    """Gets the KG results"""
    result = graph_chain.invoke(query)
    return result

KnowledgeGraphAgent = create_react_agent(
    model=llm,
    tools=[KGRetriever],
    name="KnowledgeGraphAgent",
    prompt=(
        "You are an agent whose job is to retrieve data from the knowledge graph. "
        "After retrieving the data, you must check whether the results are relevant "
        "to the query."
    )
)