import streamlit as st
import nest_asyncio
import os
import re
from pathlib import Path
from typing import List, Union
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_parse import LlamaParse
from pydantic import BaseModel, Field
from IPython.display import display, Markdown, Image

nest_asyncio.apply()

openapi = st.secrets["api_keys"]["openapi"]
llama_api= st.secrets["api_keys"]["llama_api"]
os.makedirs('data', exist_ok=True)
os.makedirs('data_images', exist_ok=True)

# Initialize parser
parser = LlamaParse(
    api_key=llama_api,
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="anthropic-sonnet-3.5",
)

st.title('Multimodel Report Generation from Slide Deck')

# Initialize session state
if 'parsed_data' not in st.session_state:
    st.session_state['parsed_data'] = None

if 'text_nodes' not in st.session_state:
    st.session_state['text_nodes'] = None

if 'index' not in st.session_state:
    st.session_state['index'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Display chat messages from history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# File uploader for PDF files
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

def get_page_number(file_name):
    """Gets page number of images using regex on file names"""
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

def _get_sorted_image_files(image_dir):
    """Get image files sorted by page."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files

def get_text_nodes(json_dicts, image_dir=None):
    """Split docs into nodes, by separator."""
    nodes = []
    image_files = _get_sorted_image_files(image_dir) if image_dir is not None else None
    md_texts = [d["md"] for d in json_dicts]
    for idx, md_text in enumerate(md_texts):
        chunk_metadata = {"page_num": idx + 1}
        if image_files is not None:
            image_file = image_files[idx]
            chunk_metadata["image_path"] = str(image_file)
        chunk_metadata["parsed_text_markdown"] = md_text
        node = TextNode(
            text="",
            metadata=chunk_metadata,
        )
        nodes.append(node)
    return nodes

def update_index(text_nodes):
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)
    index = VectorStoreIndex(text_nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir="./storage_nodes_summary")
    return index

def load_or_create_index():
    if os.path.exists("storage_nodes_summary"):
        # Remove old index data
        for file in Path("storage_nodes_summary").iterdir():
            os.remove(file)
    index = update_index(st.session_state['text_nodes'])
    return index

# Handle file parsing
if st.sidebar.button("Parse Documents"):
    if uploaded_file:
        with st.spinner("Parsing documents, please wait..."):
            # Clear previous data
            st.session_state['parsed_data'] = None
            st.session_state['text_nodes'] = None
            st.session_state['index'] = None
            # Remove old files from 'data' and 'data_images'
            for file in Path("data").iterdir():
                os.remove(file)
            for file in Path("data_images").iterdir():
                os.remove(file)
            # Remove old index if exists
            if os.path.exists("storage_nodes_summary"):
                for file in Path("storage_nodes_summary").iterdir():
                    os.remove(file)

            for file in uploaded_file:
                pdf_path = os.path.join("data", file.name)
                with open(pdf_path, "wb") as f:
                    f.write(file.getbuffer())

                md_json_objs = parser.get_json_result([pdf_path])
                md_json_list = md_json_objs[0]["pages"]
                image_dicts = parser.get_images(md_json_objs, download_path="data_images")
                st.session_state['parsed_data'] = md_json_list
                st.session_state['text_nodes'] = get_text_nodes(md_json_list, "data_images")
                st.session_state['index'] = update_index(st.session_state['text_nodes'])
                st.sidebar.success(f"Document {file.name} parsed successfully!")
    else:
        st.warning("Please upload a PDF file first.")

# Only proceed if the document has been parsed
text_nodes = st.session_state['text_nodes'] if 'text_nodes' in st.session_state else None
index = st.session_state['index'] if 'index' in st.session_state else None

if text_nodes and index:
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)
    llm = OpenAI(model="gpt-4o", api_key=openapi)
    Settings.llm = llm
    Settings.embed_model = embed_model

    retriever = index.as_retriever()

    system_prompt = """\
    You are a report generation assistant tasked with producing a well-formatted context given parsed context.

    You will be given context from one or more reports that take the form of parsed text.

    You are responsible for producing a report with interleaving text and images - in the format of interleaving text and "image" blocks.
    Since you cannot directly produce an image, the image block takes in a file path - you should write in the file path of the image instead.

    How do you know which image to generate? Each context chunk will contain metadata including an image render of the source chunk, given as a file path.
    Include ONLY the images from the chunks that have heavy visual elements (you can get a hint of this if the parsed text contains a lot of tables).
    You MUST include at least one image block in the output.

    You MUST output your response as a tool call in order to adhere to the required output format. Do NOT give back normal text.
    """

    # Ensure proper class definitions
    class TextBlock(BaseModel):
        text: str = Field(..., description="The text for this block.")

    class ImageBlock(BaseModel):
        file_path: str = Field(..., description="File path to the image.")

    class ReportOutput(BaseModel):
        blocks: List[Union[TextBlock, ImageBlock]] = Field(
            ..., description="A list of text and image blocks."
        )

        def render(self) -> None:
            for b in self.blocks:
                if isinstance(b, TextBlock):
                    st.markdown(b.text)
                else:
                    st.image(b.file_path)

    # Initialize LLM and Query Engine
    llm = OpenAI(model="gpt-4o", system_prompt=system_prompt,api_key=openapi)
    sllm = llm.as_structured_llm(output_cls=ReportOutput)
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        llm=sllm,
        response_mode="compact",
    )

    if prompt := st.chat_input("Enter your query here:"):
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if query_engine:
            try:
                response = query_engine.query(prompt)
                # Render the response if it's an instance of ReportOutput
                if isinstance(response.response, ReportOutput):
                    response.response.render()
                else:
                    st.markdown("Unexpected response format.")
            except Exception as e:
                response_text = f"Error during query execution: {e}"
                
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state['messages'].append({"role": "assistant", "content": response_text})
        else:
            st.warning("Query engine not initialized.")

