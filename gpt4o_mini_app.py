import streamlit as st
from llama_index.core import Document
import nest_asyncio
nest_asyncio.apply()
import os
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_parse import LlamaParse
from typing import List
import json
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
import tempfile

# Utility functions
def get_text_nodes(json_list: List[dict]) -> List[TextNode]:
    text_nodes = []
    for idx, page in enumerate(json_list):
        text_node = TextNode(text=page["md"], metadata={"page": page["page"]})
        text_nodes.append(text_node)
    return text_nodes

def save_jsonl(data_list: List[dict], filename: str):
    """Save a list of dictionaries as JSON Lines."""
    with open(filename, "w") as file:
        for item in data_list:
            json.dump(item, file)
            file.write("\n")

def load_jsonl(filename: str) -> List[dict]:
    """Load a list of dictionaries from JSON Lines."""
    data_list = []
    with open(filename, "r") as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

@st.cache_data
def parse_with_model(model_name: str, tmp_file_path: str) -> List[Document]:
    parser = LlamaParse(
        result_type="markdown",
        api_key=st.secrets["general"]["llama_api"],
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name=model_name,
    )
    json_objs = parser.get_json_result(tmp_file_path)
    json_list = json_objs[0]["pages"]
    docs = get_text_nodes(json_list)

    save_jsonl([d.dict() for d in docs], f"docs_{model_name}.jsonl")
    docs_dicts = load_jsonl(f"docs_{model_name}.jsonl")
    
    return [Document.model_validate(d) for d in docs_dicts]

# Function to generate file hash
def get_file_hash(file) -> str:
    """Generate a hash for the uploaded file."""
    file.seek(0)
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)
    return file_hash

# Tabs setup
tab1, tab2, tab3 = st.tabs(['GPT-4o Mini Parser', 'GPT-4o Parser', 'RAG Pipeline'])

# File upload and session state management
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf", key="upload_pdf")

# Initialize session state variables if not already present
if 'file_hash' not in st.session_state:
    st.session_state.file_hash = None
if 'docs_mini' not in st.session_state:
    st.session_state.docs_mini = None
if 'docs_gpt4o' not in st.session_state:
    st.session_state.docs_gpt4o = None
if 'response' not in st.session_state:
    st.session_state.response = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'model_option' not in st.session_state:
    st.session_state.model_option = None

if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file)
    if st.session_state.file_hash != file_hash:
        st.session_state.file_hash = file_hash
        st.session_state.docs_mini = None
        st.session_state.docs_gpt4o = None
        st.session_state.response = None
        st.session_state.metadata = None
        # Add a flag to force refresh
        st.session_state.force_refresh = True

# GPT-4o Mini Parser Tab
with tab1:
    st.header("GPT-4o Mini Parser")

    if uploaded_file is not None:
        if st.button('Start Parsing with GPT-4o Mini', key='gptmini'):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Parse the document
            st.session_state.docs_mini = parse_with_model("openai-gpt-4o-mini", tmp_file_path)
            st.success('Parsing Completed')

            # Remove the temporary file
            os.remove(tmp_file_path)

        # Display parsed content
        if st.session_state.docs_mini:
            num_pages = len(st.session_state.docs_mini)
            if num_pages > 1:
                page = st.slider('Select page', min_value=0, max_value=num_pages - 1, value=0)
                st.write('GPT-4o Mini Parser Output', st.session_state.docs_mini[page].get_content(metadata_mode="all"))
            elif num_pages == 1:
                st.write('GPT-4o Mini Parser Output', st.session_state.docs_mini[0].get_content(metadata_mode="all"))
        else:
            st.warning("No content was parsed from the document.")

# GPT-4o Parser Tab
with tab2:
    st.header("GPT-4o Parser")
    
    if uploaded_file is not None:
        if st.button('Start Parsing with GPT-4o', key='GPT'):
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Parse the document
                st.session_state.docs_gpt4o = parse_with_model("openai-gpt4o", tmp_file_path)
                st.success('Parsing Completed')
            except Exception as e:
                st.error(f"An error occurred during parsing: {e}")
            finally:
                # Remove the temporary file
                os.remove(tmp_file_path)
    
    # Display parsed content
    if st.session_state.docs_gpt4o:
        num_pages = len(st.session_state.docs_gpt4o)
        if num_pages > 1:
            page = st.slider('Select page', min_value=0, max_value=num_pages - 1, value=0, key='slider_gpt4o')
            st.write('GPT-4o Parser Output', st.session_state.docs_gpt4o[page].get_content(metadata_mode="all"))
        elif num_pages == 1:
            st.write('GPT-4o Parser Output', st.session_state.docs_gpt4o[0].get_content(metadata_mode="all"))
    else:
        st.warning("No content was parsed from the document.")

# RAG Pipeline Tab
with tab3:
    st.header("RAG Pipeline")

    # Model selection dropdown
    model_option = st.selectbox(
        "Choose the model for the RAG Pipeline:",
        ("GPT-4o Mini", "GPT-4o")
    )

    # Input for the query
    user_query = st.text_input("Enter your query:")

    # Slider for top_k
    top_k = st.slider('Select top_k value:', min_value=1, max_value=10, value=5)

    if uploaded_file and user_query:
        if st.button("Run RAG Pipeline") or st.session_state.get('model_option') != model_option:
            st.session_state.model_option = model_option

            if model_option == "GPT-4o Mini":
                if st.session_state.docs_mini is None:
                    st.warning("Please parse the document using GPT-4o Mini first.")
                else:
                    Settings.llm = OpenAI(model="gpt-4o-mini", api_key=st.secrets["general"]["openapi"])
                    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=st.secrets["general"]["openapi"])

                    # Load documents parsed by GPT-4o Mini from session state
                    docs_mini = st.session_state.docs_mini
                    index_mini = VectorStoreIndex(docs_mini)
                    query_engine_mini = index_mini.as_query_engine(similarity_top_k=top_k)

                    # Query the GPT-4o Mini engine
                    response_mini = query_engine_mini.query(user_query)
                    
                    # Save the response and metadata to session state
                    st.session_state.response = response_mini
                    st.session_state.metadata = response_mini.metadata

            elif model_option == "GPT-4o":
                if st.session_state.docs_gpt4o is None:
                    st.warning("Please parse the document using GPT-4o first.")
                else:
                    Settings.llm = OpenAI(model="gpt-4o", api_key=st.secrets["general"]["openapi"])
                    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=st.secrets["general"]["openapi"])

                    # Load documents parsed by GPT-4o from session state
                    docs_gpt4o = st.session_state.docs_gpt4o
                    index_gpt4o = VectorStoreIndex(docs_gpt4o)
                    query_engine_gpt4o = index_gpt4o.as_query_engine(similarity_top_k=top_k)

                    # Query the GPT-4o engine
                    response_gpt4o = query_engine_gpt4o.query(user_query)

                    # Save the response and metadata to session state
                    st.session_state.response = response_gpt4o
                    st.session_state.metadata = response_gpt4o.metadata

        if st.session_state.response:
            st.write("Response:", st.session_state.response)
            st.write("Metadata:")
            for node_id, metadata in st.session_state.metadata.items():
                st.write(f"Node ID: {node_id}")
                st.json(metadata)
        else:
            st.warning("No response available. Please run the RAG pipeline.")
