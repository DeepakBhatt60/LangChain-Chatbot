import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

load_dotenv()

# ---------------- Model ----------------

model = ChatMistralAI(
    model="ministral-3b-2512",
    temperature=0.7
)

# ---------------- Prompt ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are CineSage, an AI assistant specialized in extracting
        useful information from movie-related content.

        Extract the following information from the given paragraph:

        Movie Name:
        Release Year:
        Genre:
        Director:
        Writers:
        Producers:
        Cast:
        Main Characters:
        Language:
        Runtime:
        Country:
        Rating:
        Awards:
        Themes:
        Plot:
        Key Highlights:

        Also provide a quick summary of the paragraph in 2-4 sentences.

        Rules:
        - Extract information only from the given paragraph.
        - Do not make up or assume information.
        - If information is not mentioned, write "Not mentioned".
        - Keep the answer clear, concise and well organized.
        """
    ),
    (
        "human",
        """
        Extract useful information from the following movie paragraph:

        {paragraph}
        """
    )
])

# ---------------- UI ----------------

st.set_page_config(
    page_title="CineSage",
    page_icon="🎬"
)

st.title("🎬 CineSage")
st.write("Extract useful information from a movie paragraph.")

paragraph = st.text_area(
    "Enter Movie Paragraph",
    height=200,
    placeholder="Paste your movie paragraph here..."
)

if st.button("🔍 Extract Information"):

    if paragraph.strip():

        with st.spinner("Analyzing movie information..."):

            messages = prompt.invoke({
                "paragraph": paragraph
            })

            response = model.invoke(messages)

        st.subheader("📋 Movie Information")
        st.write(response.content)

    else:
        st.warning("Please enter a movie paragraph.")