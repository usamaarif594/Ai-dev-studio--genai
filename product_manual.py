import streamlit as st
import nest_asyncio
import os
import re
from pathlib import Path
import typing as t
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.schema import NodeWithScore, MetadataMode
from llama_index.core.base.response.schema import Response
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import ImageNode
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings

nest_asyncio.apply()





openapi_key = st.secrets["api_keys"]["openai_api_key"]
llama_api_key = st.secrets["api_keys"]["llama_api_key"]
os.makedirs('data', exist_ok=True)
os.makedirs('data_images', exist_ok=True)

parser = LlamaParse(
    api_key=llama_api_key,
    result_type="markdown",
    parsing_instruction="You are given IKEA assembly instruction manuals",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="openai-gpt4o",
    show_progress=True,
)

st.title('Chat with Product Instruction Manuals')

# Initialize session state for parsed data
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

def get_text_nodes(json_dicts, image_dir) -> t.List[TextNode]:
    """Creates nodes from json + images"""
    nodes = []
    docs = [doc["md"] for doc in json_dicts]  # extract text
    image_files = _get_sorted_image_files(image_dir)  # extract images

    for idx, doc in enumerate(docs):
        node = TextNode(
            text=doc,
            metadata={"image_path": str(image_files[idx]), "page_num": idx + 1},
        )
        nodes.append(node)

    return nodes

def update_index(text_nodes):
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi_key)
    index = VectorStoreIndex(text_nodes, embed_model=embed_model)
    index.storage_context.persist(persist_dir="./storage_ikea")
    return index

def load_or_create_index():
    if os.path.exists("storage_ikea"):
        # Remove old index data
        for file in Path("storage_ikea").iterdir():
            os.remove(file)
    index = update_index(st.session_state['text_nodes'])
    return index

# Handle file parsing
if st.sidebar.button("Parse Documents"):
    if uploaded_file:
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
        if os.path.exists("storage_ikea"):
            for file in Path("storage_ikea").iterdir():
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
    embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=openapi_key)
    llm = OpenAI("gpt-4o", api_key=openapi_key)

    Settings.llm = llm
    Settings.embed_model = embed_model

    retriever = index.as_retriever()

    QA_PROMPT_TMPL = """\
    Below we give parsed text from slides in two different formats, as well as the image.

    We parse the text in both 'markdown' mode as well as 'raw text' mode. Markdown mode attempts \
    to convert relevant diagrams into tables, whereas raw text tries to maintain the rough spatial \
    layout of the text.

    Use the image information first and foremost. ONLY use the text/markdown information 
    if you can't understand the image.

    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query. Explain whether you got the answer
    from the parsed markdown or raw text or image, and if there's discrepancies, and your reasoning for the final answer.

    Query: {query_str}
    Answer: """

    QA_PROMPT = PromptTemplate(QA_PROMPT_TMPL)

    gpt_4o_mm = OpenAIMultiModal(model="gpt-4o", max_new_tokens=4096, api_key=openapi_key)

    class MultimodalQueryEngine(CustomQueryEngine):
        qa_prompt: PromptTemplate
        retriever: BaseRetriever
        multi_modal_llm: OpenAIMultiModal

        def __init__(
            self,
            qa_prompt: PromptTemplate,
            retriever: BaseRetriever,
            multi_modal_llm: OpenAIMultiModal,
        ):
            super().__init__(
                qa_prompt=qa_prompt, retriever=retriever, multi_modal_llm=multi_modal_llm
            )

        def custom_query(self, query_str: str):
            nodes = self.retriever.retrieve(query_str)
            image_nodes = [
                NodeWithScore(node=ImageNode(image_path=n.node.metadata["image_path"]))
                for n in nodes
            ]
            ctx_str = "\n\n".join(
                [r.node.get_content(metadata_mode=MetadataMode.LLM) for r in nodes]
            )
            fmt_prompt = self.qa_prompt.format(context_str=ctx_str, query_str=query_str)
            llm_response = self.multi_modal_llm.complete(
                prompt=fmt_prompt,
                image_documents=[image_node.node for image_node in image_nodes],
            )
            return Response(
                response=str(llm_response),
                source_nodes=nodes,
                metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
            )

    query_engine = MultimodalQueryEngine(
        qa_prompt=QA_PROMPT,
        retriever=index.as_retriever(similarity_top_k=9),
        multi_modal_llm=gpt_4o_mm,
    )

    # Chat input handling
    if prompt := st.chat_input("Enter your query here:"):
        # Add user message to chat history
        st.session_state['messages'].append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response using the query engine
        if query_engine:
            try:
                response = query_engine.custom_query(prompt)
                response_text = str(response.response)
            except Exception as e:
                response_text = f"Error during query execution: {e}"
        else:
            response_text = "No query engine available. Please upload and parse documents first."
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response_text)
        
        # Add assistant response to chat history
        st.session_state['messages'].append({"role": "assistant", "content": response_text})
else:
    st.sidebar.warning("No documents parsed yet. Please upload a PDF file and parse it.")
