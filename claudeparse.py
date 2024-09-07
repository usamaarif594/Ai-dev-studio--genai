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

openapi = st.secrets["general"]["openapi"]
llama_api = st.secrets["general"]["llama_api"]

st.sidebar.header('Upload File')
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf", key="pdf_file")

tab1, tab2, tab3 = st.tabs(['Parse with Sonnet', 'Parse with GPT-4o', 'RAG Pipeline'])

def get_text_nodes(json_list: List[dict]):
    text_nodes = []
    for idx, page in enumerate(json_list):
        text_node = TextNode(text=page["md"], metadata={"page": page["page"]})
        text_nodes.append(text_node)
    return text_nodes

def save_jsonl(data_list, filename):
    """Save a list of dictionaries as JSON Lines."""
    with open(filename, "w") as file:
        for item in data_list:
            json.dump(item, file)
            file.write("\n")

def load_jsonl(filename):
    """Load a list of dictionaries from JSON Lines."""
    data_list = []
    with open(filename, "r") as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list
@st.cache_data
def parse_with_model(model_name, tmp_file_path):
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
    
    return [Document.parse_obj(d) for d in docs_dicts]

# Tab 1: Upload and Parse with Sonnet
with tab1:
    st.header('Parse with Sonnet')

    if uploaded_file is not None:
        if st.button('Start Parsing with Sonnet', key='Sonnet'):
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.session_state.docs_sonnet = parse_with_model("anthropic-sonnet-3.5", tmp_file_path)
            st.success('Parsing Completed')
            
            os.remove(tmp_file_path)
        if 'docs_sonnet' in st.session_state:
            st.sidebar.subheader('Read Parsed Content')
            max_pages = len(st.session_state.docs_sonnet) - 1
            page_number = st.sidebar.slider('Select Page', min_value=0, max_value=max_pages, value=0)
            
            if st.sidebar.button('Show Page Content'):
                st.write(st.session_state.docs_sonnet[page_number].get_content(metadata_mode="all"))
    else:
        st.warning('Please upload a PDF file in the sidebar.')

# Tab 2: Parse with GPT-4o
with tab2:
    st.header('Parse with GPT-4o')

    if uploaded_file is not None:
        if st.button('Start Parsing with GPT-4o', key='GPT'):
            
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            st.session_state.docs_gpt4o = parse_with_model("openai-gpt4o", tmp_file_path)
            st.success('Parsing Completed')
            
            os.remove(tmp_file_path)
        if 'docs_gpt4o' in st.session_state:
            st.subheader('Read GPT-4o Parsed Content')
            max_pages_gpt4o = len(st.session_state.docs_gpt4o) - 1
            page_number_gpt4o = st.sidebar.slider('Select Page for GPT-4o', min_value=0, max_value=max_pages_gpt4o, value=0, key='gpt4o_page_slider')
            
            if st.sidebar.button('Show Page Content', key='ShowGPTContent'):
                st.write(st.session_state.docs_gpt4o[page_number_gpt4o].get_content(metadata_mode="all"))
    else:
        st.warning('Please upload a PDF file in the sidebar.')

with tab3:
    st.header('RAG Pipeline')

    # Check if any documents are available
    if ('docs_sonnet' in st.session_state or 'docs_gpt4o' in st.session_state):

        # Allow user to select a model
        model_option = st.selectbox(
            'Select Model',
            ['Sonnet', 'GPT-4o']
        )

        # Allow user to input their own query
        user_query = st.text_input('Enter your query').strip()

        # Add a button to submit the query
        submit_button = st.button('Submit Query')

        # Ensure that user_query is not empty and the button is clicked
        if submit_button:
        # Ensure that user_query is not empty
            if not user_query:
                st.warning('Query cannot be empty.')
            else:
                # Initialize models and indexes based on selected option
                if model_option == 'Sonnet' and 'docs_sonnet' in st.session_state:
                    Settings.llm = OpenAI(model="gpt-4o", api_key=openapi)
                    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)
                    
                    index = VectorStoreIndex(st.session_state.docs_sonnet)
                    query_engine = index.as_query_engine(similarity_top_k=5)
                    
                    try:
                        response = query_engine.query(user_query)
                        
                        # Display only response content
                        st.subheader('Sonnet Response')
                        if hasattr(response, 'response') and response.response:
                            st.write(response.response)
                        else:
                            st.warning('No response content available.')

                        # Display only the content of the first source node, if available
                        # if hasattr(response, 'source_nodes') and len(response.source_nodes) > 0:
                            # st.subheader('Sonnet Response Source')
                            # st.write(response.source_nodes[].get_content())
                        # else:
                            # st.warning('No source nodes available.')
                    except Exception as e:
                        st.error(f"An error occurred with Sonnet: {str(e)}")

                elif model_option == 'GPT-4o' and 'docs_gpt4o' in st.session_state:
                    Settings.llm = OpenAI(model="gpt-4o", api_key=openapi)
                    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)
                    
                    index_gpt4o = VectorStoreIndex(st.session_state.docs_gpt4o)
                    query_engine_gpt4o = index_gpt4o.as_query_engine(similarity_top_k=5)
                    
                    try:
                        response_gpt4o = query_engine_gpt4o.query(user_query)
                        
                        # Display only response content
                        st.subheader('GPT-4o Response')
                        if hasattr(response_gpt4o, 'response') and response_gpt4o.response:
                            st.write(response_gpt4o.response)
                        else:
                            st.warning('No response content available.')

                        # Display only the content of the first source node, if available
                        # if hasattr(response_gpt4o, 'source_nodes') and len(response_gpt4o.source_nodes) > 0:
                            # st.subheader('GPT-4o Response Source')
                            # st.write(response_gpt4o.source_nodes[10].get_content())
                        # else:
                            # st.warning('No source nodes available.')
                    except Exception as e:
                        st.error(f"An error occurred with GPT-4o: {str(e)}")

    else:
        st.warning('Please parse the file in either Tab 1 or Tab 2 first.')
