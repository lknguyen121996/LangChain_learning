from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from third_parties.linkedin import get_linkedin_profile



information = """
    William Henry Gates III (born October 28, 1955) is an American businessman and philanthropist. A pioneer of the microcomputer revolution of the 1970s and 1980s, he co-founded the software company Microsoft in 1975 with his childhood friend Paul Allen. Following the company's 1986 initial public offering (IPO), Gates became then the youngest ever billionaire in 1987, at age 31. Forbes magazine ranked him as the world's wealthiest person for 18 out of 24 years between 1995 and 2017, including 13 years consecutively from 1995 to 2007. He became the first centibillionaire in 1999, when his net worth briefly surpassed $100 billion. On the 2024 Forbes list, he was ranked the world's seventh wealthiest person, with an estimated net worth of $128 billion.

Born and raised in Seattle, Washington, Gates was privately educated at Lakeside School, where he befriended Allen and developed his computing interests. In 1973, he enrolled at Harvard College, where he took classes including Math 55 and graduate level computer science courses, but he dropped out in 1975 to co-found and lead Microsoft. He served as its CEO for the next 25 years and also became president and chairman of the board when the company incorporated in 1981. Succeeded as CEO by Steve Ballmer in 2000, he transitioned to chief software architect, a position he held until 2008. He stepped down as chairman of the board in 2014 and became technology adviser to CEO Satya Nadella and other Microsoft leaders, a position he still holds. He resigned from the board in 2020.

Over time, Gates has reduced his role at Microsoft to focus on his philanthropic work with the Gates Foundation, the world's largest private charitable organization. Focusing on areas including health, education, and poverty alleviation, Gates became known for his efforts to eradicate transmissible diseases such as tuberculosis, malaria, and polio. Gates and his then-wife Melinda French Gates co-chaired the Bill & Melinda Gates Foundation until 2024, when the latter resigned following the couple's divorce; the foundation was subsequently renamed, with Gates as its sole chair.

Gates is founder and chairman of several other companies, including BEN, Cascade Investment, TerraPower, Gates Ventures, and Breakthrough Energy. In 2010, he and Warren Buffett founded the Giving Pledge, whereby they and other billionaires pledge to give at least half their wealth to philanthropy. Named as one of the 100 most influential people of the 20th century by Time magazine in 1999, he has received numerous other honors and accolades, including a Presidential Medal of Freedom, awarded jointly to him and French Gates in 2016 for their philanthropic work. The subject of several documentary films, he published the first of three planned memoirs, Source Code: My Beginnings, in 2025.
    """
def main():
    load_dotenv()
    summary_template = """
    Given the linkedin information {information} about a person I want you to create:
    1. a short summary
    2. two interesting facts about them
    """
    sumary_prompt = PromptTemplate(template=summary_template, input_variables=["information"])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    chain = sumary_prompt | llm
    linkedin_profile = get_linkedin_profile()
    res = chain.invoke({"information": linkedin_profile})
    print(res)
    
if __name__ == "__main__":
    main()