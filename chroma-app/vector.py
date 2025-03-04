from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import requests
import os
import tempfile

# Configurazione Chroma DB
CHROMA_DB_PATH = "./chroma_db"

# Caricamento del modello di embeddings locale
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Lista dei PDF da caricare
pdf_list = [
    "https://www.agid.gov.it/sites/default/files/repository_files/allegato_1-_linee_guida_per_ladozione_di_un_ciclo_di_sviluppo_di_software_sicuro.pdf",
    "https://www.agid.gov.it/sites/default/files/repository_files/allegato_2_-_linee_guida_per_lo_sviluppo_sicuro_di_codice.pdf",
    "https://www.agid.gov.it/sites/default/files/repository_files/allegato_3_-_linee_guida_per_la_configurazione_per_adeguare_la_sicurezza_del_software_di_base.pdf",
    "https://www.agid.gov.it/sites/default/files/repository_files/allegato_4_-_linee_guida_per_la_modellazione_delle_minacce-dlt.pdf"
]

def load_vector_db():
    """Load existing Chroma DB."""
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name="collection"
    )

def download_pdf(url, output_path):
    """Download a PDF from a URL and save it locally."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def document_ingestion(pdf_list):
    """Carica i PDF, genera embeddings e salva in Chroma DB."""
    all_docs = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, pdf_url in enumerate(pdf_list):
            print(f"Processing PDF {i+1}/{len(pdf_list)}: {pdf_url}")
            # Download PDF to temporary file
            local_path = os.path.join(temp_dir, f"document_{i}.pdf")
            downloaded_path = download_pdf(pdf_url, local_path)
            
            if not downloaded_path:
                continue
                
            # Load the local PDF file
            loader = PyPDFLoader(downloaded_path)
            data = loader.load()

            # Dividere il testo in chunk
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(data)

            print(f"  - Extracted {len(docs)} chunks from document")
            all_docs.extend(docs)

        if not all_docs:
            print("No documents were processed successfully!")
            return

        print(f"Creating vector database with {len(all_docs)} document chunks...")
        
        # Creazione della base dati vettoriale con LangChain
        vector_db = Chroma.from_documents(
            documents=all_docs,
            embedding=embedding_model,
            collection_name="collection",
            persist_directory=CHROMA_DB_PATH
        )
        
        print("Documenti indicizzati con successo in Chroma DB!")
        return vector_db
        
    except Exception as e:
        print(f"Error during document ingestion: {e}")
    finally:
        # Clean up temporary files
        for file in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

if __name__ == "__main__":
    # Esegui il caricamento dei documenti
    vector_db = document_ingestion(pdf_list)