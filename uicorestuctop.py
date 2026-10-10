import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional

# -------------------- Page Config --------------------

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Information Extractor")
st.write("Enter a movie description and extract structured movie information.")

# -------------------- LLM --------------------

llm = ChatOllama(
    model="gemma2:2b",
    temperature=0
)

# -------------------- Pydantic Model --------------------

class movie(BaseModel):
    title: str
    genre: List[str]
    rating: float
    director: Optional[str] = None
    producers: Optional[str] = None
    cast: List[str]
    year: int
    runtime: Optional[str] = None

# -------------------- Parser --------------------

parser = PydanticOutputParser(pydantic_object=movie)

# -------------------- Prompt --------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at extracting movie information from text.

{format_instructions}
"""
    ),
    (
        "human",
        "{Discription}"
    )
])

# -------------------- Input --------------------

para = st.text_area(
    "Movie Description",
    height=220,
    placeholder="Paste a movie description here..."
)

# -------------------- Button --------------------

if st.button("Extract Information", use_container_width=True):

    if para.strip():

        with st.spinner("Extracting..."):

            response1 = prompt.invoke(
                {
                    "Discription": para,
                    "format_instructions": parser.get_format_instructions()
                }
            )

            response = llm.invoke(response1)

            try:
                movie_data = parser.parse(response.content)

                st.success("Extraction Successful!")

                st.subheader("Extracted Information")

                st.write(f"**Title:** {movie_data.title}")
                st.write(f"**Genre:** {', '.join(movie_data.genre)}")
                st.write(f"**Rating:** {movie_data.rating}")
                st.write(f"**Director:** {movie_data.director}")
                st.write(f"**Producer:** {movie_data.producers}")
                st.write(f"**Cast:** {', '.join(movie_data.cast)}")
                st.write(f"**Year:** {movie_data.year}")
                st.write(f"**Runtime:** {movie_data.runtime}")

                with st.expander("Raw LLM Output"):
                    st.code(response.content, language="json")

            except Exception as e:
                st.error("Failed to parse the model output.")
                st.exception(e)

                with st.expander("Raw LLM Output"):
                    st.code(response.content, language="json")

    else:
        st.warning("Please enter a movie description.")