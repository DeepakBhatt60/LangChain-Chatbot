from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatMistralAI(
    model="ministral-3b-2512"
)


class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


parser = PydanticOutputParser(
    pydantic_object=Movie
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the paragraph.

        {format_instructions}
        """
    ),
    (
        "human",
        "{paragraph}"
    )
])


para = input("Give me a paragraph: ")

messages = prompt.invoke({
    "paragraph": para,
    "format_instructions": parser.get_format_instructions()
})

response = model.invoke(messages)

print(response.content)