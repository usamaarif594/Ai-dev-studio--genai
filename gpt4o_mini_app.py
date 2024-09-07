import os
import hashlib
from llama_index.core.schema import TextNode
from typing import List
import json
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_parse import LlamaParse
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
import nest_asyncio
nest_asyncio.apply()
import streamlit as st

openapi = st.secrets["general"]["openapi"]
llama_api = st.secrets["general"]["llama_api"]

# Sidebar for file upload
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

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
        api_key=llama_api,
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

# Check if a new file is uploaded
if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    
    if st.session_state.get('file_hash') != file_hash:
        # Clear session state for documents if the file is new
        st.session_state.clear()
        st.session_state.file_hash = file_hash

# GPT-4o Mini Parser Tab
with tab1:
    st.header("GPT-4o Mini Parser")
    if uploaded_file:
        os.makedirs("./temp", exist_ok=True)
        file_path = f"./temp/{uploaded_file.name}"
        
        if 'docs_mini' not in st.session_state:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            if st.button("Start Parsing with GPT-4o Mini"):
                st.session_state.docs_mini = parse_with_model("openai-gpt-4o-mini", file_path)

        if 'docs_mini' in st.session_state:
            page = st.slider('Select page', min_value=0, max_value=len(st.session_state.docs_mini)-1, value=0)
            st.write('GPT-4o Mini Parser Output', st.session_state.docs_mini[page].get_content(metadata_mode="all"))

# GPT-4o Parser Tab
with tab2:
    st.header("GPT-4o Parser")
    if uploaded_file:
        os.makedirs("./temp", exist_ok=True)
        file_path = f"./temp/{uploaded_file.name}"

        if 'docs_gpt4o' not in st.session_state:
            if st.button("Start Parsing with GPT-4o"):
                st.session_state.docs_gpt4o = parse_with_model("openai-gpt4o", file_path)

        if 'docs_gpt4o' in st.session_state:
            page = st.slider('Select page', min_value=0, max_value=len(st.session_state.docs_gpt4o)-1, value=0, key='slider_gpt4o')
            st.write('GPT-4o Parser Output', st.session_state.docs_gpt4o[page].get_content(metadata_mode="all"))

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
                Settings.llm = OpenAI(model="gpt-4o-mini", api_key=openapi)
                Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)

                # Load documents parsed by GPT-4o Mini from session state
                if 'docs_mini' in st.session_state:
                    docs_mini = st.session_state.docs_mini
                else:
                    docs_mini_dicts = load_jsonl("docs_openai-gpt-4o-mini.jsonl")
                    docs_mini = [Document.model_validate(d) for d in docs_mini_dicts]

                index_mini = VectorStoreIndex(docs_mini)
                query_engine_mini = index_mini.as_query_engine(similarity_top_k=top_k)

                # Query the GPT-4o Mini engine
                response_mini = query_engine_mini.query(user_query)
                
                # Save the response and metadata to session state
                st.session_state.response = response_mini
                st.session_state.metadata = response_mini.metadata

            elif model_option == "GPT-4o":
                Settings.llm = OpenAI(model="gpt-4o", api_key=openapi)
                Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)

                # Load documents parsed by GPT-4o from session state
                if 'docs_gpt4o' in st.session_state:
                    docs_gpt4o = st.session_state.docs_gpt4o
                else:
                    docs_gpt4o_dicts = load_jsonl("docs_openai-gpt4o.jsonl")
                    docs_gpt4o = [Document.model_validate(d) for d in docs_gpt4o_dicts]

                index_gpt4o = VectorStoreIndex(docs_gpt4o)
                query_engine_gpt4o = index_gpt4o.as_query_engine(similarity_top_k=top_k)

                # Query the GPT-4o engine
                response_gpt4o = query_engine_gpt4o.query(user_query)

                # Save the response and metadata to session state
                st.session_state.response = response_gpt4o
                st.session_state.metadata = response_gpt4o.metadata

        # Display the results from session state
        if st.session_state.response:
            st.subheader('Response')
            st.write(st.session_state.response.response)

            st.subheader('Metadata')
            for node_id, meta in st.session_state.metadata.items():
                st.write(f"Node ID: {node_id} - Page: {meta['page']}") 
