import streamlit as st
import nest_asyncio
import os
import re
from pathlib import Path
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.schema import TextNode, ImageNode
import openai
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

nest_asyncio.apply()





openapi = st.secrets["openai"]["api_key"]
llama_api = st.secrets["llama"]["api_key"]

nest_asyncio.apply()

# Ensure the 'claims' directory exists
CLAIMS_DIR = "claims"
os.makedirs(CLAIMS_DIR, exist_ok=True)
os.makedirs("data_images", exist_ok=True)
# Initialize OpenAI API client
openai.api_key = openapi

# Define the function here
def get_page_number(file_name: str) -> int:
    match = re.search(r"-page-(\d+)\.jpg$", file_name)
    if match:
        return int(match.group(1))
    return 0

# Initialize Streamlit app
st.title("Insurance Claim Document Analysis & Chat")

# Initialize session state variables
if 'parsed_data' not in st.session_state:
    st.session_state['parsed_data'] = None

if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'image_files' not in st.session_state:
    st.session_state['image_files'] = []

if 'show_image' not in st.session_state:
    st.session_state['show_image'] = False

# Display chat messages from history
for message in st.session_state['messages']:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
uploaded_files = st.sidebar.file_uploader("Upload your claim documents", accept_multiple_files=True)

# Handle file uploads and parsing
if st.sidebar.button("Parse Documents") and uploaded_files:
    # Ensure the 'claims' directory exists
    os.makedirs(CLAIMS_DIR, exist_ok=True)
    os.makedirs("data_images", exist_ok=True)

    files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(CLAIMS_DIR, uploaded_file.name)
        st.write(f"Saving file: {file_path}")  # Debugging statement
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        files.append(file_path)

    # Debug: Check if files are saved
    for file_path in files:
        if not os.path.exists(file_path):
            st.error(f"File not found after saving: {file_path}")
            st.stop()

    try:
        parser = LlamaParse(
            api_key=llama_api,
            result_type="markdown",
            parsing_instruction="This is an auto insurance claim document.",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            show_progress=True,
            gpt4o_api_key=openapi
        )

        md_json_objs = parser.get_json_result(files)
        parser.get_images(md_json_objs, download_path="data_images")

        md_json_list = []
        for obj in md_json_objs:
            md_json_list.extend(obj["pages"])

        def _get_sorted_image_files(image_dir: str) -> list:
            raw_files = [f for f in Path(image_dir).iterdir() if f.is_file()]
            sorted_files = sorted(raw_files, key=lambda p: get_page_number(p.name))
            return sorted_files

        def get_text_nodes(json_dicts: list, image_dir: str) -> list:
            nodes = []
            docs = [doc["md"] for doc in json_dicts]
            image_files = _get_sorted_image_files(image_dir)
            st.session_state['image_files'] = image_files
            for idx, doc in enumerate(docs):
                page_number = idx + 1
                node = TextNode(
                    text=doc,
                    metadata={"image_path": str(image_files[idx]), "page_num": page_number},
                )
                image_node = ImageNode(
                    image_path=str(image_files[idx]),
                    metadata={"page_num": page_number},
                )
                nodes.extend([node, image_node])
            return nodes

        text_nodes = get_text_nodes(md_json_list, "data_images")

        embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi)
        llm = OpenAI("gpt-4o", api_key=openapi)

        Settings.llm = llm
        Settings.embed_model = embed_model

        if not Path("storage_insurance").exists():
            index = VectorStoreIndex(text_nodes, embed_model=embed_model)
            index.storage_context.persist(persist_dir="./storage_insurance")
        else:
            ctx = StorageContext.from_defaults(persist_dir="./storage_insurance")
            index = load_index_from_storage(ctx)

        st.session_state['parsed_data'] = text_nodes
        st.session_state['query_engine'] = index.as_query_engine()
        st.success("Documents parsed successfully!")

    except FileNotFoundError as e:
        st.error(f"File not found error during parsing: {e}")
    except Exception as e:
        st.error(f"Error during parsing: {e}")


# if st.session_state['image_files']:
#     screenshot_files = st.session_state['image_files']
#     options = [f"Page {get_page_number(f.name)}" for f in screenshot_files]
#     selected_image = st.sidebar.selectbox("Select a screenshot to view:", options=options)

#     if st.sidebar.button('Show Images'):
#         st.session_state['show_image'] = True

#     if st.session_state['show_image']:
#         # Convert Path object to string
#         selected_file_path = str(screenshot_files[options.index(selected_image)])
#         st.image(selected_file_path, caption=f"Screenshot: {selected_image}", width=400)


# Chat input handling
if prompt := st.chat_input("Enter your query here:"):
    # Add user message to chat history
    st.session_state['messages'].append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response using the query engine
    if st.session_state['query_engine']:
        response = st.session_state['query_engine'].query(prompt)
        response_text = str(response)
    else:
        response_text = "No query engine available. Please upload and parse documents first."
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    # Add assistant response to chat history
    st.session_state['messages'].append({"role": "assistant", "content": response_text})
