import streamlit as st
import json
from utils import call_openrouter_api, format_json, convert_to_csv

# Page configuration
st.set_page_config(
    page_title="Text to Quiz JSON Converter",
    page_icon="📝",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextArea textarea {
        font-family: monospace;
    }
    .json-output {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📝 Text to Quiz JSON Converter")
st.markdown("""
This app converts your text into a structured quiz format using AI. Simply paste your text below,
and the app will generate quiz questions with multiple choice answers in a organized JSON format.
""")

# API Key input with default value
api_key = st.text_input(
    "Enter your OpenRouter API Key",
    value="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",
    type="password",
    help="Your API key will not be stored"
)

# Example text
example_text = """
The Python programming language was created by Guido van Rossum and was first released in 1991.
Python is known for its simple syntax and readability. It supports multiple programming paradigms,
including procedural, object-oriented, and functional programming. Python's name was inspired by
the British comedy group Monty Python.
"""

# Text input
st.subheader("Input Text")
text_input = st.text_area(
    "Enter your text here",
    height=200,
    help="Paste the text you want to convert into a quiz",
    placeholder=example_text
)

# Process button
if st.button("Generate Quiz JSON", disabled=not text_input):
    if not text_input:
        st.error("Please enter some text")
    else:
        with st.spinner("Generating quiz..."):
            result, error = call_openrouter_api(text_input, api_key)

            if error:
                st.error(error)
            else:
                # Create two columns for JSON and CSV output
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Generated Quiz JSON")
                    formatted_json = format_json(result)
                    st.code(formatted_json, language="json")
                    st.download_button(
                        label="Download JSON",
                        data=formatted_json,
                        file_name="quiz.json",
                        mime="application/json"
                    )

                with col2:
                    st.subheader("Generated Quiz CSV")
                    csv_data = convert_to_csv(result)
                    st.code(csv_data, language="csv")
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name="quiz.csv",
                        mime="text/csv"
                    )

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. Enter your OpenRouter API key in the secure input field
    2. Paste your text in the input area
    3. Click 'Generate Quiz JSON' to convert your text
    4. The generated JSON will appear below
    5. Download options:
       - Use 'Download JSON' for structured format with chapters
       - Use 'Download CSV' for simple question-answer format

    The generated JSON will include:
    - Subject name based on the content
    - Chapters containing related questions
    - Multiple choice questions with:
        - Question text
        - Four answer options
        - Correct answer
        - Detailed explanation
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and OpenRouter API")