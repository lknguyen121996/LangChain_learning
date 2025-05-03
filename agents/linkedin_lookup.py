import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langgraph.prebuilt import (
    create_react_agent
)
from langsmith import Client
from langchain_community.tools.tavily_search import TavilySearchResults


load_dotenv()

def lookup(name: str) -> str:
    model = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    template = """
    Given the name {name}
    I want you to find a LinkedIn profile url that belongs to them.
    You will get a list of urls, and you need to return the one that belongs to them.
    """
    prompt_template = PromptTemplate(template=template, input_variables=["name"])
    tools_for_agent = [
        Tool(
            name="Crawl_4_profile",
            func=TavilySearchResults(k=4).run,
            description="useful for when you need the linkedin url of a person"
        )
    ]
    agent = create_react_agent(
        model=model,
        tools=tools_for_agent
    )
    formatted_input = prompt_template.format(name=name)
    resp = agent.invoke({"messages": [formatted_input]})
    return resp["messages"]
