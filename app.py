from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import json
from neo4j import GraphDatabase
from openai import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import  VectorStoreIndex
from llama_index.core import Settings
import requests
from tempfile import NamedTemporaryFile
from llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownElementNodeParser
from openai import OpenAI
import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# FastAPI application
app = FastAPI()

# Add a CORS middleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Neo4j and OpenAI configurations
NEO4J_URL = os.environ["NEO4J_URI"]
NEO4J_USER = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = "neo4j"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4-0125-preview"
TABLE_REF_SUFFIX = '_table_ref'
TABLE_ID_SUFFIX  = '_table'

llm = OpenAI(model=GENERATION_MODEL)
Settings.llm = llm

def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses

    # Create a temporary file to hold the downloaded PDF
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name  # Return the file path of the downloaded PDF

def get_embedding(client, text, model):
    response = client.embeddings.create(
                    input=text,
                    model=model,
                )
    return response.data[0].embedding

def LoadEmbedding(label, property):
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)
    openai_client = OpenAI (api_key = os.environ["OPENAI_API_KEY"])

    with driver.session() as session:
        # get chunks in document, together with their section titles
        result = session.run(f"MATCH (ch:{label}) RETURN id(ch) AS id, ch.{property} AS text")
        # call OpenAI embedding API to generate embeddings for each proporty of node
        # for each node, update the embedding property
        count = 0
        for record in result:
            id = record["id"]
            text = record["text"]
            
            # For better performance, text can be batched
            embedding = get_embedding(openai_client, text, EMBEDDING_MODEL)
            
            # key property of Embedding node differentiates different embeddings
            cypher = "CREATE (e:Embedding) SET e.key=$key, e.value=$embedding, e.model=$model"
            cypher = cypher + " WITH e MATCH (n) WHERE id(n) = $id CREATE (n) -[:HAS_EMBEDDING]-> (e)"
            session.run(cypher,key=property, embedding=embedding, id=id, model=EMBEDDING_MODEL) 
            count = count + 1

        session.close()
        
        print("Processed " + str(count) + " " + label + " nodes for property @" + property + ".")
        return count

# Database session setup
def get_neo4j_session():
    return GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD), database=NEO4J_DATABASE)

def initialiseNeo4jSchema():
    cypher_schema = [
        "CREATE CONSTRAINT personNif IF NOT EXISTS FOR (p:Person) REQUIRE p.nif IS UNIQUE;",  # Ensuring NIF is unique for each person
        "CREATE CONSTRAINT sectionKey IF NOT EXISTS FOR (c:Section) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT chunkKey IF NOT EXISTS FOR (c:Chunk) REQUIRE (c.key) IS UNIQUE;",
        "CREATE CONSTRAINT documentKey IF NOT EXISTS FOR (c:Document) REQUIRE (c.url_hash) IS UNIQUE;",
        "CREATE VECTOR INDEX `chunkVectorIndex` IF NOT EXISTS FOR (e:Embedding) ON (e.value) OPTIONS { indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};"
    ]

    driver = GraphDatabase.driver(NEO4J_URL, database=NEO4J_DATABASE, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        for cypher in cypher_schema:
            session.run(cypher)
    driver.close()

# Pydantic model for receiving PDF file path and customer_id
class ProcessRequest(BaseModel):
    pdf_file_path: str
    customer_id: str
    customer_name: str

@app.post("/process-pdf/")
async def process_pdf(request: ProcessRequest):
    try:
        # Load and parse PDF
        pdf_file_path = download_pdf(request.pdf_file_path)
        
        documents = LlamaParse(result_type="markdown").load_data(pdf_file_path)
        node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
        nodes = node_parser.get_nodes_from_documents(documents)
        base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

        initialiseNeo4jSchema()

        driver = GraphDatabase.driver(NEO4J_URL, database=NEO4J_DATABASE, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        print("Start saving person to Neo4j...")
        with driver.session() as session:
            # First, handle the person node
            person_cypher = """
            MERGE (p:Person {nif: $nif})
            ON CREATE SET p.name=$name
            ON MATCH SET p.name=$name
            """
            session.run(person_cypher, nif=request.customer_id, name=request.customer_name)
            
        print("Start saving documents to Neo4j...")
        i = 0
        with driver.session() as session:
            for doc in documents:
                cypher = "MERGE (d:Document {url_hash: $doc_id}) ON CREATE SET d.url=$url;"
                session.run(cypher, doc_id=doc.doc_id, url=doc.doc_id)

                result = session.run("""
                    MATCH (p:Person {nif: $nif}), (d:Document {url_hash: $doc_id})
                    MERGE (d)-[r:ASSOCIATED_WITH]->(p)
                    RETURN r
                """, {'nif': request.customer_id, 'doc_id': doc.doc_id})
                i = i + 1
            session.close()
        
        print(f"{i} documents saved.")
        
        print("Start saving nodes to Neo4j...")
        i = 0
        with driver.session() as session:
            for node in base_nodes: 
        
                # >>1 Create Section node
                cypher  = "MERGE (c:Section {key: $node_id})\n"
                cypher += " FOREACH (ignoreMe IN CASE WHEN c.type IS NULL THEN [1] ELSE [] END |\n"
                cypher += "     SET c.hash = $hash, c.text=$content, c.type=$type, c.class=$class_name, c.start_idx=$start_idx, c.end_idx=$end_idx )\n"
                cypher += " WITH c\n"
                cypher += " MATCH (d:Document {url_hash: $doc_id})\n"
                cypher += " MERGE (d)<-[:HAS_DOCUMENT]-(c);"
        
                node_json = json.loads(node.json())
        
                session.run(cypher, node_id=node.node_id, hash=node.hash, content=node.get_content(), type='TEXT', class_name=node.class_name()
                                  , start_idx=node_json['start_char_idx'], end_idx=node_json['end_char_idx'], doc_id=node.ref_doc_id)
        
                # >>2 Link node using NEXT relationship
        
                if node.next_node is not None: # and node.next_node.node_id[-1*len(TABLE_REF_SUFFIX):] != TABLE_REF_SUFFIX:
                    cypher  = "MATCH (c:Section {key: $node_id})\n"    # current node should exist
                    cypher += "MERGE (p:Section {key: $next_id})\n"    # previous node may not exist
                    cypher += "MERGE (p)<-[:NEXT]-(c);"
        
                    session.run(cypher, node_id=node.node_id, next_id=node.next_node.node_id)
        
                if node.prev_node is not None:  # Because tables are in objects list, so we need to link from the opposite direction
                    cypher  = "MATCH (c:Section {key: $node_id})\n"    # current node should exist
                    cypher += "MERGE (p:Section {key: $prev_id})\n"    # previous node may not exist
                    cypher += "MERGE (p)-[:NEXT]->(c);"
        
                    if node.prev_node.node_id[-1 * len(TABLE_ID_SUFFIX):] == TABLE_ID_SUFFIX:
                        prev_id = node.prev_node.node_id + '_ref'
                    else:
                        prev_id = node.prev_node.node_id
        
                    session.run(cypher, node_id=node.node_id, prev_id=prev_id)
                i = i + 1
            session.close()
        
        print(f"{i} nodes saved.")
        
        print("Start saving objects to Neo4j...")
        i = 0
        with driver.session() as session:
            for node in objects:               
                node_json = json.loads(node.json())
        
                # Object is a Table, then the ????_ref_table object is created as a Section, and the table object is Chunk
                if node.node_id[-1 * len(TABLE_REF_SUFFIX):] == TABLE_REF_SUFFIX:
                    if node.next_node is not None:  # here is where actual table object is loaded
                        next_node = node.next_node
        
                        obj_metadata = json.loads(str(next_node.json()))
        
                        cypher  = "MERGE (s:Section {key: $node_id})\n"
                        cypher += "WITH s MERGE (c:Chunk {key: $table_id})\n"
                        cypher += " FOREACH (ignoreMe IN CASE WHEN c.type IS NULL THEN [1] ELSE [] END |\n"
                        cypher += "     SET c.hash = $hash, c.definition=$content, c.text=$table_summary, c.type=$type, c.start_idx=$start_idx, c.end_idx=$end_idx )\n"
                        cypher += " WITH s, c\n"
                        cypher += " MERGE (s) <-[:UNDER_SECTION]- (c)\n"
                        cypher += " WITH s MATCH (d:Document {url_hash: $doc_id})\n"
                        cypher += " MERGE (d)<-[:HAS_DOCUMENT]-(s);"
        
                        session.run(cypher, node_id=node.node_id, hash=next_node.hash, content=obj_metadata['metadata']['table_df'], type='TABLE'
                                          , start_idx=node_json['start_char_idx'], end_idx=node_json['end_char_idx']
                                          , doc_id=node.ref_doc_id, table_summary=obj_metadata['metadata']['table_summary'], table_id=next_node.node_id)
                        
                    if node.prev_node is not None:
                        cypher  = "MATCH (c:Section {key: $node_id})\n"    # current node should exist
                        cypher += "MERGE (p:Section {key: $prev_id})\n"    # previous node may not exist
                        cypher += "MERGE (p)-[:NEXT]->(c);"
        
                        if node.prev_node.node_id[-1 * len(TABLE_ID_SUFFIX):] == TABLE_ID_SUFFIX:
                            prev_id = node.prev_node.node_id + '_ref'
                        else:
                            prev_id = node.prev_node.node_id
                        
                        session.run(cypher, node_id=node.node_id, prev_id=prev_id)
                i = i + 1
            session.close()
        
        print("Start creating chunks for each TEXT Section...")
        with driver.session() as session:
            cypher  = "MATCH (s:Section) WHERE s.type='TEXT' \n"
            cypher += "WITH s CALL {\n"
            cypher += "WITH s WITH s, split(s.text, '\n') AS para\n"
            cypher += "WITH s, para, range(0, size(para)-1) AS iterator\n"
            cypher += "UNWIND iterator AS i WITH s, trim(para[i]) AS chunk, i WHERE size(chunk) > 0\n"
            cypher += "CREATE (c:Chunk {key: s.key + '_' + i}) SET c.type='TEXT', c.text = chunk, c.seq = i \n"
            cypher += "CREATE (s) <-[:UNDER_SECTION]-(c) } IN TRANSACTIONS OF 500 ROWS ;"
            
            session.run(cypher)
            session.close()

        print(f"{i} objects saved.")
        
        print("=================DONE====================")
        
        driver.close()

        # For smaller amount (<2000) of text data to embed
        LoadEmbedding("Chunk", "text")
        
        return {"status": "success", "message": "PDF processed and data stored successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run locally for development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, loop='asyncio')
