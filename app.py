from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from neo4j import GraphDatabase
from openai import OpenAI

# Load environment variables
load_dotenv()

# FastAPI application
app = FastAPI()

# Neo4j and OpenAI configurations
NEO4J_URL = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4-0125-preview"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Database session setup
def get_neo4j_session():
    return GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)

# Pydantic model for receiving PDF file path
class ProcessRequest(BaseModel):
    pdf_file_path: str

@app.post("/process-pdf/")
def process_pdf(request: ProcessRequest):
    try:
        # Import processing modules
        from llama_parse import LlamaParse
        from llama_index.core.node_parser import MarkdownElementNodeParser

        # Load and parse PDF
        documents = LlamaParse(result_type="markdown").load_data(request.pdf_file_path)
        llm = OpenAI(model=GENERATION_MODEL)
        node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        return {"status": "success", "message": "PDF processed and data stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally for development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
