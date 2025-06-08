from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import render_text_description, Tool
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from typing import Union, List, Tuple
from langchain.schema import AgentAction, AgentFinish
from dotenv import load_dotenv
from third_parties.linkedin import get_linkedin_profile
from agents.linkedin_lookup import lookup
from agent_custom.agents import get_text_length, replace_text
from agent_custom.callbacks import AgentCallbackHandler

information = """
    William Henry Gates III (born October 28, 1955) is an American businessman and philanthropist. A pioneer of the microcomputer revolution of the 1970s and 1980s, he co-founded the software company Microsoft in 1975 with his childhood friend Paul Allen. Following the company's 1986 initial public offering (IPO), Gates became then the youngest ever billionaire in 1987, at age 31. Forbes magazine ranked him as the world's wealthiest person for 18 out of 24 years between 1995 and 2017, including 13 years consecutively from 1995 to 2007. He became the first centibillionaire in 1999, when his net worth briefly surpassed $100 billion. On the 2024 Forbes list, he was ranked the world's seventh wealthiest person, with an estimated net worth of $128 billion.

Born and raised in Seattle, Washington, Gates was privately educated at Lakeside School, where he befriended Allen and developed his computing interests. In 1973, he enrolled at Harvard College, where he took classes including Math 55 and graduate level computer science courses, but he dropped out in 1975 to co-found and lead Microsoft. He served as its CEO for the next 25 years and also became president and chairman of the board when the company incorporated in 1981. Succeeded as CEO by Steve Ballmer in 2000, he transitioned to chief software architect, a position he held until 2008. He stepped down as chairman of the board in 2014 and became technology adviser to CEO Satya Nadella and other Microsoft leaders, a position he still holds. He resigned from the board in 2020.

Over time, Gates has reduced his role at Microsoft to focus on his philanthropic work with the Gates Foundation, the world's largest private charitable organization. Focusing on areas including health, education, and poverty alleviation, Gates became known for his efforts to eradicate transmissible diseases such as tuberculosis, malaria, and polio. Gates and his then-wife Melinda French Gates co-chaired the Bill & Melinda Gates Foundation until 2024, when the latter resigned following the couple's divorce; the foundation was subsequently renamed, with Gates as its sole chair.

Gates is founder and chairman of several other companies, including BEN, Cascade Investment, TerraPower, Gates Ventures, and Breakthrough Energy. In 2010, he and Warren Buffett founded the Giving Pledge, whereby they and other billionaires pledge to give at least half their wealth to philanthropy. Named as one of the 100 most influential people of the 20th century by Time magazine in 1999, he has received numerous other honors and accolades, including a Presidential Medal of Freedom, awarded jointly to him and French Gates in 2016 for their philanthropic work. The subject of several documentary films, he published the first of three planned memoirs, Source Code: My Beginnings, in 2025.
    """


def third_party():
    load_dotenv()
    summary_template = """
    Given the linkedin information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """
    sumary_prompt = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = sumary_prompt | llm
    linkedin_profile = get_linkedin_profile()
    res = chain.invoke({"information": linkedin_profile})
    print(res)


def linkedin_lookup():
    lookup_agent = lookup("Nguyen Kim Long ANZ hcltech")
    print(lookup_agent)


def custom_agent(question: str):
    # Step 1: Define the prompt template
    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action convert to dictionary format
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
    """

    # Step 2: Setup tools and prompt
    tools = [get_text_length, replace_text]
    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=",".join([t.name for t in tools]),
    )

    # Step 3: Initialize components
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        stop="\nObservation",
        callbacks=[AgentCallbackHandler()],
    )
    intermediate_steps = []

    # Step 4: Create the agent chain
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # Step 5: Run agent in a loop until we get AgentFinish
    max_iterations = 5  # Prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": question, "agent_scratchpad": intermediate_steps}
        )

        if isinstance(agent_step, AgentFinish):
            print(f"Final Answer: {agent_step.return_values}")
            break

        # Execute tool if agent requests it
        tool_name = agent_step.tool
        tool_to_use = find_tool(tools, tool_name)
        tool_input = agent_step.tool_input
        if isinstance(tool_input, str):
            tool_input = eval(tool_input)
        observation = tool_to_use.func(tool_input)
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))


def find_tool(tools: List[Tool], tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def format_log_to_str(
    intermediate_steps: List[Tuple[AgentAction, str]],
    observation_prefix: str = "Observation:",
    llm_prefix: str = "Thought:",
) -> str:
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += action.log
        thoughts += f"\n{observation_prefix}{observation}\n{llm_prefix}"
    return thoughts


if __name__ == "__main__":
    custom_agent(input("Enter a question: "))
