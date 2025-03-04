import os
from langchain_community.llms.ollama import Ollama
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
from vector import load_vector_db

# Choose a lightweight model that works well on CPU
MODEL_NAME = "granite3-moe:1b" # Alternatives: "tinyllama"

def setup_ollama_model():
    """Initialize the Ollama LLM with CPU configuration."""
    return Ollama(
        model=MODEL_NAME,
        temperature=0.1,
        num_ctx=2048,  # Context window size
        num_gpu=0,     # Use CPU only
    )

def setup_openai_model():
    """Initialize the OpenAI gpt-4o-mini model."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    return ChatOpenAI(model="gpt-4o-mini")

def create_prompt_from_docs(docs):
    """Format retrieved documents into context for the prompt."""
    context_text = ""
    for i, (doc, score) in enumerate(docs):
        context_text += f"Document {i+1} [Relevance: {1-score:.4f}]:\n"
        context_text += f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        context_text += f"Content: {doc.page_content}\n\n"
    
    return context_text

def generate_answer(query, retrieved_docs):
    """Generate an answer based on the query and retrieved documents."""
    # Setup the model
    llm = setup_ollama_model()
    
    # Create a prompt template
    prompt_template = """
    Sei un assistente esperto che risponde a domande basandosi sui documenti forniti.
    
    Contesto:
    {context}
    
    Domanda: {question}
    
    Fornisci una risposta dettagliata basata solo sulle informazioni contenute nei documenti sopra.
    Se le informazioni non sono sufficienti per rispondere, indica chiaramente che non puoi rispondere
    basandoti solo sui documenti forniti.
    
    Risposta:
    """
    
    # Format the prompt with the retrieved documents
    context = create_prompt_from_docs(retrieved_docs)
    
    # Create and run the chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Generate the answer
    result = chain.run(context=context, question=query)
    
    return result

def rag_pipeline(query_text, n_results=5):
    """Full RAG pipeline: retrieve documents and generate an answer."""
    # Load the vector database
    db = load_vector_db()
    
    # Perform similarity search
    retrieved_docs = db.similarity_search_with_score(query_text, k=n_results)
    
    print(f"\nQuery: {query_text}")
    print(f"Retrieved {len(retrieved_docs)} relevant chunks")
    
    # Generate answer
    print("\nGenerating answer...")
    answer = generate_answer(query_text, retrieved_docs)
    
    print("\n--- Generated Answer ---")
    print(answer)
    
    return answer
