import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain.llms import GooglePalm
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GooglePalmEmbeddings
from pypdf import PdfReader
from dotenv import load_dotenv

# Load API Key from environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Initialize Gemini and ChromaDB
llm = GooglePalm(api_key=GOOGLE_API_KEY, temperature=0.3)
embeddings = GooglePalmEmbeddings(api_key=GOOGLE_API_KEY)
chroma_db_dir = "chroma_storage"
vector_db = Chroma(persist_directory=chroma_db_dir, embedding_function=embeddings)

### Utility Functions ###

# Load PDF file and extract text
def pdf_to_text(file: UploadFile) -> str:
    """Extract text from PDF."""
    reader = PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Text splitter
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Create prompt templates
def create_prompt_template(field: str) -> PromptTemplate:
    return PromptTemplate(
        input_variables=["transcript"],
        template=f"Based on the company concall transcript, list the {field}:\n\n{{transcript}}"
    )

# Define LLM chains
def create_chain(llm: GooglePalm, field: str) -> LLMChain:
    prompt_template = create_prompt_template(field)
    return LLMChain(llm=llm, prompt=prompt_template)

# Store transcript chunks into ChromaDB
def store_in_vector_db(text_chunks):
    documents = [{"page_content": chunk} for chunk in text_chunks]
    vector_db.add_documents(documents)

# Analyze the transcript text
def analyze_transcript(transcript: str) -> dict:
    fields = ["pros and cons", "MOATs", "business model", "expansion plans", "debt reasons", 
              "reasons to invest", "reasons to avoid investing"]
    
    # Create chains for analysis
    chains = {field: create_chain(llm, field) for field in fields}
    
    # Split the transcript into chunks
    chunks = split_text(transcript)

    # Store chunks into vector DB for future reference
    store_in_vector_db(chunks)
    
    # Analyze the transcript chunk by chunk
    results = {field: [] for field in fields}
    for chunk in chunks:
        for field, chain in chains.items():
            result = chain.run(chunk)
            results[field].append(result)
    
    # Concatenate the results
    final_results = {field: "\n".join(results[field]) for field in results}
    return final_results

### FastAPI Endpoints ###

@app.post("/upload-transcript/")
async def upload_transcript(file: UploadFile = File(...)):
    """API to upload a PDF file and analyze the company concall transcript."""
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Please upload a PDF."})

    try:
        # Define the save path in the 'transcripts/' directory
        save_path = f"transcripts/{file.filename}"

        # Save the uploaded file
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())

        # Convert PDF to text
        with open(save_path, "rb") as saved_file:
            transcript_text = pdf_to_text(saved_file)

        # Analyze the text
        analysis_results = analyze_transcript(transcript_text)

        # Return analysis results
        return analysis_results

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.get("/retrieve-chunk/")
async def retrieve_chunk(query: str):
    """API to retrieve a document chunk from ChromaDB using query."""
    try:
        # Retrieve documents matching query from vector DB
        docs = vector_db.similarity_search(query)
        return {"documents": docs}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
