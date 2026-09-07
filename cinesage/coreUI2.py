import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬")
st.title("🎬 Movie Info Extractor")
st.write("Paste a paragraph describing a movie, and I'll extract structured details from it.")


class Movie(BaseModel):
    title: str
    release_year: Optional[int] = None
    genre: List[str]
    director: Optional[str] = None
    cast: List[str]
    rating: Optional[float] = None
    summary: str


# Initialize model once
if "model" not in st.session_state:
    st.session_state.model = ChatMistralAI(
        model="ministral-3b-2512"
    )

parser = PydanticOutputParser(pydantic_object=Movie)

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

para = st.text_area("Give me a paragraph:", height=180, placeholder="e.g. Inception is a 2010 sci-fi thriller directed by Christopher Nolan...")

if st.button("Extract Movie Info"):
    if not para.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting..."):
            try:
                messages = prompt.invoke({
                    "paragraph": para,
                    "format_instructions": parser.get_format_instructions()
                })

                response = st.session_state.model.invoke(messages)
                movie = parser.parse(response.content)

                st.success("Extraction complete!")

                st.subheader(movie.title)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Release Year", movie.release_year or "N/A")
                with col2:
                    st.metric("Rating", movie.rating if movie.rating is not None else "N/A")

                st.write("**Genre:**", ", ".join(movie.genre) if movie.genre else "N/A")
                st.write("**Director:**", movie.director or "N/A")
                st.write("**Cast:**", ", ".join(movie.cast) if movie.cast else "N/A")
                st.write("**Summary:**")
                st.write(movie.summary)

                with st.expander("Raw JSON"):
                    st.json(movie.model_dump())

            except Exception as e:
                st.error(f"Failed to parse movie info: {e}")
                st.text(response.content if 'response' in locals() else "No response received.")