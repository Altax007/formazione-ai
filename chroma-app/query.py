# Add this code after your existing code
from vector import load_vector_db
from generate import rag_pipeline, generate_answer


def query_vector_db(query_text, n_results=5):
    """Query the vector database with a question and get relevant document chunks."""
    # Connect to the existing database
    db = load_vector_db()
    
    # Perform similarity search
    results = db.similarity_search_with_score(query_text, k=n_results)
    
    print(f"\nQuery: {query_text}")
    print(f"Found {len(results)} relevant chunks:\n")
    
    for i, (doc, score) in enumerate(results):
        print(f"Result {i+1} [Relevance: {1-score:.4f}]:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
        print(f"Content: {doc.page_content}\n")
    
    return results

def query_and_generate(query_text, n_results=5):
    """Query vector DB and generate an answer."""
    results = query_vector_db(query_text, n_results)
    answer = generate_answer(query_text, results)
    return answer, results

if __name__ == "__main__":
    # Example query with retrieval
    query_vector_db("Come gestire le minacce di sicurezza?")

    # Example query with generation
    rag_pipeline("Quali sono le best practices per la sicurezza informatica?")

