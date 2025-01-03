# METIL_Open
open source scalable RAG using LLMS
README: Document Query System Using FAISS and GPT4All
Overview

This project is a Python-based system designed to query and summarize content from PDF documents. It integrates FAISS for efficient similarity searches, SentenceTransformer for text embeddings, and GPT4All for generating responses based on document content. The system is tailored to handle long documents, providing chunking, summarization, and response generation while respecting token limits.
Features

    PDF Loading: Extracts and reads text from PDFs in a specified folder.
    Text Chunking: Splits large texts into smaller, manageable chunks.
    Embedding Generation: Uses a pre-trained SentenceTransformer model to generate embeddings for text chunks.
    FAISS Indexing: Creates a searchable index for efficient retrieval of relevant chunks.
    Token Limit Management: Handles token limitations for LLMs by summarizing or shortening text.
    LLM Integration: Uses GPT4All to generate responses to user queries based on the retrieved text context.

Requirements
Python Libraries

    os
    faiss
    numpy
    PyPDF2
    sentence-transformers
    langchain_community.llms.gpt4all

Additional Resources

    SentenceTransformer Model: Pre-trained model (all-MiniLM-L6-v2 recommended).
    GPT4All Model: Specify the path to your GPT4All model.

Installation

    Clone the repository and navigate to the project directory.
    Install required Python libraries:

    pip install faiss-cpu numpy PyPDF2 sentence-transformers langchain_community

    Download the GPT4All model and configure the main() function with its path.

Usage

    Run the script:

    python script_name.py

    Enter the folder path containing the PDFs when prompted.
    Follow the instructions to interact with your document database:
        Enter a query to retrieve and interact with document content.
        Type exit to quit the application.

Core Components
PDF Processing

    load_pdfs_from_folder(folder_path): Reads and extracts text from PDF files.
    chunk_text(text, chunk_size=500): Splits text into smaller chunks for processing.

Embedding and Indexing

    generate_embeddings(chunks, model): Converts text chunks into embeddings.
    build_faiss_index(embeddings): Builds a FAISS index for efficient similarity searches.

Query and Response

    query_faiss_index(index, query, model, chunks, top_k=3): Retrieves relevant chunks based on query similarity.
    generate_response(llm, query, retrieved_chunks, max_context_tokens=1500): Generates an LLM response using retrieved chunks.

Configuration

    Embedding Model: Update the SentenceTransformer model in the main() function.
    LLM Configuration: Specify the path to the GPT4All model and its backend in GPT4All() initialization.

Example Workflow

    Place your PDF files in a folder.
    Run the script and specify the folder path.
    Enter a query such as:

    What are the key points from the document about AI advancements?

    The system retrieves relevant content and generates a summarized response.

Limitations and Future Enhancements

    Model Compatibility: Ensure the GPT4All model matches your backend and computational capacity.
    Large Contexts: Summarization can reduce granularity for very long documents.
    Scalability: Performance may decrease with very large datasets; consider optimizing the FAISS index or embedding generation.

License

This project is open-source under the Apache 2.0 License. Contributions and enhancements are welcome!
