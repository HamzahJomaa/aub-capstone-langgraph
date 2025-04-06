from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.tools import tool
from rdflib import Graph
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
graph = Neo4jGraph(username="neo4j", database="ontologies", refresh_schema=True)

rdf_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    allow_dangerous_requests=True,
    validate_cypher=True,
    verbose=True,
    return_direct=True,
    return_intermediate_steps=True,
)


@tool
def RDFInspector(query: str) -> str:
    """Gets the rdf results"""
    result = rdf_chain.invoke(query)
    return result


def _sanitize_output(text: str):
    _, after = text.split("```turtle")
    return after.split("```")[0]


def get_rdf_file(entity_name: str, file_path: str = None) -> str:
    """
    Generates RDF Turtle data for the given entity and saves it to a local file.
    Returns the full path to the saved TTL file.
    """
    # Prompt to generate RDF
    prompt = ChatPromptTemplate(["""
    You are a semantic web assistant. Generate a valid RDF file in Turtle (.ttl) format that provides rich semantic information about the following entity:

    Entity: {entity_name}

    The RDF must include:

    1. A unique, dereferenceable URI for the entity.
    2. The following standard prefixes:

       @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
       @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
       @prefix owl: <http://www.w3.org/2002/07/owl#> .
       @prefix foaf: <http://xmlns.com/foaf/0.1/> .
       @prefix schema: <http://schema.org/> .
       @prefix dcterms: <http://purl.org/dc/terms/> .
       @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

    3. Include the following properties (as applicable to the entity type):

       - `rdf:type` (e.g., `foaf:Person`, `schema:Organization`)
       - `rdfs:label` — human-readable name or label
       - `rdfs:comment` or `schema:description` — a clear, informative description
       - `foaf:name` — full name
       - `schema:birthDate` / `schema:foundingDate` — typed as `xsd:date`
       - `schema:location` — string or URI
       - `foaf:homepage` — URL
       - `foaf:isPrimaryTopicOf` — a related article or webpage
       - `owl:sameAs` — links to relevant external resources such as Wikidata or DBpedia (at least 2)

    4. Ensure:
       - All literals with datatypes (e.g., dates) use correct `^^xsd:` types.
       - URIs are enclosed in `< >`.
       - Strings are in double quotes and optional language tags or datatypes are added as needed.
       - Output only valid Turtle code — no explanations, just the code.

    Use realistic, coherent, and well-linked data. Follow semantic web best practices and produce output suitable for integration into linked data ecosystems.

    """])

    chain = prompt | llm | StrOutputParser() | _sanitize_output

    rdf_content = chain.invoke({"entity_name": entity_name})

    print(rdf_content)

    # File path setup
    if not file_path:
        file_path = f"{entity_name.replace(' ', '_').lower()}.ttl"

    # Save to file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(rdf_content)

    return file_path


def upload_to_neo4j_and_cleanup(file_path: str):
    """
    Parses the given TTL file into a Neo4j store, then deletes the file.
    """
    auth_data = {
        'uri': os.environ["NEO4J_URI"],
        'database': "ontologies",
        'user': "neo4j",
        'pwd': os.environ["NEO4J_PASSWORD"],
    }

    config = Neo4jStoreConfig(
        auth_data=auth_data,
        handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
        batching=True
    )

    graph = Graph(store=Neo4jStore(config=config))
    graph.parse(file_path, format="ttl")
    graph.close(True)  # Commit if batching is used

    # Remove file after successful parse
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"✅ Deleted file: {file_path}")
    else:
        print(f"⚠️ File not found for deletion: {file_path}")


@tool
def RDFGenerator(entity: str) -> str:
    """
    Generates RDF data for the given entity, uploads it to the Neo4j database, and cleans up the temporary TTL file.

    Args:
        entity (str): The name or identifier of the entity for which RDF data should be generated.

    Returns:
        str: The path to the uploaded TTL file or a confirmation message upon successful upload.
    """
    ttl_file = get_rdf_file(entity)
    upload_to_neo4j_and_cleanup(ttl_file)


RDFOrchestratorAgent = create_react_agent(
    model=llm,
    tools=[RDFInspector, RDFGenerator],
    name="RDFOrchestratorAgent",
    prompt="""
You are RDFOrchestratorAgent, an intelligent agent responsible for enriching entity information using RDF.

You operate **after entities have been extracted** by the KnowledgeGraphAgent.

You have access to two tools:

1. **RDFInspector** – Use this if RDF already exists for an entity. It retrieves existing RDF data.
2. **RDFGenerator** – Use this if RDF does not exist or more RDF is needed. It creates new RDF data for the entity.

### Instructions:

- You receive a list of entities.
- For each entity:
  - First, attempt to retrieve its RDF using **RDFInspector**.
  - If no RDF is found or the data is incomplete, use **RDFGenerator** to generate RDF.
- Your goal is to **enrich each entity** with contextual or semantic RDF data to support downstream reasoning.
- After processing all entities, return the enriched RDF data as your output.

### Output:

Return the collected or generated RDF for the entities.
Do not perform reasoning or inference—your job is to gather contextual data only.
"""
)