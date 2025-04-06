from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import tool
from langchain_openai import ChatOpenAI  # Or whichever LLM you're using
from langgraph.prebuilt import create_react_agent

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

@tool
def GetMoreInfoTool(entities: list) -> str:
    """Tool responsible for getting more entities"""
    results = graph_chain.invoke({"query": entities})
    return results



@tool
def StartReasoningTool(data: str) -> str:
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
    print(results.content)
    return results

ReasoningAgent = create_react_agent(
    model=llm,
    tools=[StartReasoningTool, GetMoreInfoTool],
    name="ReasoningAgent",
    prompt="You are an expert agent responsible only to navigate between tools. Always start by start_reasoning_tool. Always retrieve the reasoning as is to the supervisor"
)

