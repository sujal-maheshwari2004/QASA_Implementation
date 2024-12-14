from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA

# Step 1: Load Documents with PyMuPDFLoader
def load_documents(file_path):
    loader = PyMuPDFLoader(file_path)  # Load the PDF file
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents found in the file: {file_path}")
    print(f"Loaded {len(documents)} pages from {file_path}.")
    return documents

# Step 2: Split Documents into Chunks
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Step 3: Create Vector Store with OllamaEmbeddings
def create_vectorstore(documents, embedding_model="llama3.2"):
    embeddings = OllamaEmbeddings(model=embedding_model)  # Use Ollama for embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Step 4: Create RAG Pipeline with Few-Shot Prompts
def create_rag_pipeline(vectorstore, model_name="llama3.2"):
    # Few-shot examples
    few_shot_prompt = """
    Example 1:
    Question: What is the significance of AI in healthcare?
    Context:
    - AI enables early disease diagnosis through predictive analytics.
    - It enhances patient care via personalized treatment recommendations.
    Rationale:
    AI in healthcare revolutionizes patient outcomes by enabling early detection and tailoring treatment strategies to individual needs.
    Final Answer:
    AI's significance in healthcare lies in improving early diagnosis and providing personalized treatment, leading to better patient outcomes.

    Example 2:
    Question: What are the challenges of renewable energy adoption?
    Context:
    - High initial infrastructure costs.
    - Variability in energy production due to weather dependence.
    Rationale:
    Renewable energy adoption faces barriers like high setup costs and unpredictable output, requiring advancements in storage and grid technologies.
    Final Answer:
    The main challenges include high costs and variability, necessitating better storage and grid solutions to stabilize energy supply.

    Now answer the following question using a similar approach but provide only the answer and dont give the context or the rationale:
    """

    # ChatOllama initialization with few-shot prompts
    llm = ChatOllama(
        model=model_name,
        preamble=few_shot_prompt
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top-3 chunks

    def generate_answer(query):
        # Use the new `invoke` method for document retrieval
        retrieved_docs = retriever.invoke(query)
        combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Full prompt for the LLM
        full_prompt = (
            f"{few_shot_prompt}\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{combined_context}\n\n"
            f"Rationale:\nGenerate rationale from the above evidence.\n\n"
            f"Final Answer:"
        )
        
        # Use `invoke` for ChatOllama to process the prompt
        response = llm.invoke(full_prompt)
        return response, retrieved_docs

    return generate_answer

# Step 5: Query the Pipeline
def ask_question(rag_pipeline, question):
    try:
        # Query the pipeline
        answer, sources = rag_pipeline(question)

        # Display answer
        print(f"\nAnswer:\n{answer}\n")
        
        # Display sources
        print("Sources:")
        for idx, source in enumerate(sources, 1):
            source_name = source.metadata.get("source", "Unknown")
            page_number = source.metadata.get("page_number", "N/A")
            print(f"{idx}: {source_name} - Page: {page_number}")
        
        return answer, sources
    except Exception as e:
        print(f"Error during retrieval or generation: {e}")
        return None, None

# Main Execution
if __name__ == "__main__":
    # Path to the PDF document
    file_path = "test.pdf"

    try:
        # Load and preprocess documents
        documents = load_documents(file_path)
        split_docs = split_documents(documents)

        # Create vector store with OllamaEmbeddings and RAG pipeline
        vectorstore = create_vectorstore(split_docs)
        rag_pipeline = create_rag_pipeline(vectorstore)

        # Ask a question
        question = input("Ask a question: ")
        ask_question(rag_pipeline, question)
    except Exception as e:
        print(f"Pipeline error: {e}")
