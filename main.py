import os
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_community.llms.gpt4all import GPT4All

#################################################
# PDF LOADING, CHUNKING, & EMBEDDING
#################################################
def load_pdfs_from_folder(folder_path):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            documents.append({"text": text, "file_name": file})
    return documents

def chunk_text(text, chunk_size=500):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(chunks, model):
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def query_faiss_index(index, query, model, chunks, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    results = [{"chunk": chunks[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results

#################################################
# TOKEN LIMIT HANDLING
#################################################
def approximate_token_count(text: str) -> int:
    """
    A rough approximation: 
    - Many LLMs average ~4 characters per token (English).
    - Adjust the divisor as needed for your model/language.
    """
    return len(text) // 4

def summarize_text(llm, text: str, max_length: int = 500) -> str:
    """
    A simple summarization approach if a chunk is too long. 
    For demonstration, we prompt the model to summarize the text.
    """
    prompt = (
        f"Please summarize the following text in about {max_length} characters:\n\n"
        f"{text}\n\nSummary:"
    )
    summary = llm.invoke(prompt, max_tokens=2048) #original is 512
    return summary

#################################################
# LLM RESPONSE GENERATION (with token-limit safety)
#################################################
def generate_response(llm, query, retrieved_chunks, max_context_tokens=1500):
    """
    1. Build a context from retrieved_chunks.
    2. If context exceeds the max_context_tokens limit, 
       summarize or shorten chunks until it's within limit.
    3. Generate the final answer using GPT4All.
    """
    # Combine retrieved chunks
    context_text = "\n\n".join([chunk["chunk"] for chunk in retrieved_chunks])
    prompt = f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"

    # Approximate how many tokens we have
    current_token_count = approximate_token_count(prompt)

    # If over limit, do a simple approach: Summarize entire context
    if current_token_count > max_context_tokens:
        print("Context is too long; summarizing before final answer...")
        context_text = summarize_text(llm, context_text, max_length=1000)
        # Rebuild the prompt with the summarized context
        prompt = f"Context: {context_text}\n\nQuestion: {query}\nAnswer:"
    
    # Now call GPT4All with your final prompt
    response = llm.invoke(prompt, max_tokens=1024)
    return response

#################################################
# MAIN
#################################################
def main():
    folder_path = input("Enter the folder path containing PDFs: ").strip()
    if not os.path.exists(folder_path):
        print("Invalid folder path. Please try again.")
        return

    print("Loading SentenceTransformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Loading PDFs...")
    documents = load_pdfs_from_folder(folder_path)

    print("Chunking text...")
    chunks = []
    for doc in documents:
        chunks.extend(chunk_text(doc["text"]))

    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks, embedding_model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Loading LLM (GPT4All)...")
    # Adjust model path, backend, etc., as needed
    llm = GPT4All(
        model="PATH_TO_YOUR_MODEL",
        backend="TYPE_OF_BACKEND", #ie llama
        verbose=True,
        max_tokens=1024  # default for each generation step
    )

    print("Ready to chat with your documents!")
    print("Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        print("Retrieving relevant chunks...")
        retrieved_chunks = query_faiss_index(index, query, embedding_model, chunks)

        print("Generating response...")
        response = generate_response(llm, query, retrieved_chunks)
        print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
