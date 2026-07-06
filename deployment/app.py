%%writefile AirlineChatBot_project/deployment/app.py
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
PDF_PATH = "FlykiteAirlinesHRP.pdf"
ASTRA_COLLECTION_NAME = "flykite_hr_policies"

# Astra DB credentials
ASTRA_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

# LLM Configuration
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

SYSTEM_PROMPT = """You are an expert HR assistant for Flykite Airlines. \
Answer employee questions clearly and accurately using only the HR policy \
document context provided below.

Guidelines:
- Be concise but detailed
- Use bullet points for lists or multi-part answers
- Always cite the relevant policy area (e.g. "Per the Leave Policy...")
- If the answer is not in the context, say: \
"I couldn't find that in the Flykite HR handbook. Please contact HR directly."

Context from HR Handbook:
{context}"""

# --- Caching Functions ---
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="thenlper/gte-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Initialize Astra DB Vector Store
    vector_store = AstraDBVectorStore(
        collection_name=ASTRA_COLLECTION_NAME,
        embedding=embeddings,
        api_endpoint=ASTRA_API_ENDPOINT,
        token=ASTRA_APPLICATION_TOKEN,
        namespace=ASTRA_KEYSPACE,
    )

    # Auto-ingest if collection is empty
    try:
        dummy_check = vector_store.similarity_search("test_query_placeholder", k=1)
    except Exception:
        dummy_check = []

    if not dummy_check:
        print("🔄 Astra DB collection is empty. Ingesting PDF...")
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, chunk_overlap=20, encoding_name='cl100k_base'
        )
        chunks = splitter.split_documents(docs)
        vector_store.add_documents(chunks)
        print("✅ PDF ingested successfully into Astra DB!")

    return vector_store.as_retriever(search_kwargs={"k": 4})


def load_llm():
    model_path = hf_hub_download(
        repo_id=model_name_or_path,
        filename=model_basename,
        resume_download=True,
        cache_dir="./huggingface_cache",
    )
    return Llama(
        model_path=model_path,
        n_threads=4,
        n_batch=512,
        n_gpu_layers=0,
        n_ctx=4096,
    )


# --- Global singletons (loaded once at startup) ---
print(" Loading retriever and LLM at startup...")
retriever = load_retriever()
llm = load_llm()
print("✅ Models ready!")


# --- Chat Function ---
def respond(message, history):
    """Core chat function called by Gradio on every user turn."""
    # Build context from retrieved docs
    docs = retriever.invoke(message)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build chat messages list for Llama.cpp
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    ]
    # Append prior conversation so the LLM has full context
    for user_msg, bot_msg in history:
        chat_messages.append({"role": "user", "content": user_msg})
        chat_messages.append({"role": "assistant", "content": bot_msg})
    chat_messages.append({"role": "user", "content": message})

    # Generate response
    response = llm.create_chat_completion(messages=chat_messages)
    answer = response["choices"][0]["message"]["content"]

    # Append page references
    pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
    if pages:
        answer += f"\n\n* Referenced pages: {', '.join(str(p) for p in pages)}*"

    return answer


# --- Example Questions ---
example_questions = [
    "How do I apply for annual leave, and how much notice is required?",
    "What is the process for requesting emergency bereavement leave?",
    "Can unused sick days be carried over to the next year?",
    "What happens to my leave balance if I resign during the year?",
    "Are flight crew entitled to additional rest days after long-haul flights?",
]

# --- Gradio UI ---
with gr.Blocks(
    title="Flykite HR Bot",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.Markdown(
        """
        # ✈️ Flykite Airlines HR Policy Assistant
        Instant answers from the official Flykite Airlines HR handbook.

        **Ask about:** Leave policies · Benefits · Code of conduct ·
        Disciplinary procedures · Compliance
        """
    )

    # Credential check
    if not all([ASTRA_API_ENDPOINT, ASTRA_APPLICATION_TOKEN]):
        gr.Error(
            " Missing Astra DB credentials. "
            "Please set `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN` "
            "in your `.env` file."
        )
    else:
        chatbot = gr.ChatInterface(
            fn=respond,
            examples=example_questions,
            title="Chat with the HR Handbook",
            description="Type a question or click an example below.",
            retry_btn="🔄 Retry",
            undo_btn="↩️ Undo",
            clear_btn="️ Clear",
            type="messages",   # modern message-style history
            fill_height=True,
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
