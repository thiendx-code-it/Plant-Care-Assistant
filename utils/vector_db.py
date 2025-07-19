from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional
from config.settings import settings

class VectorDBManager:
    """Manager for Pinecone vector database operations"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.index = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Pinecone and embeddings"""
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            
            # Initialize Azure OpenAI embeddings
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT_EMBED,
                api_key=settings.AZURE_OPENAI_API_KEY_EMBED,
                azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_EMBED,
                api_version=settings.AZURE_OPENAI_API_VERSION,
                model="text-embedding-3-small"  # Use the correct model
            )
            
            # Check if index exists, create if not
            existing_indexes = [index.name for index in pc.list_indexes()]
            if settings.PINECONE_INDEX_NAME not in existing_indexes:
                pc.create_index(
                    name=settings.PINECONE_INDEX_NAME,
                    dimension=1536,  # text-embedding-3-small dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
            
            # Get the index
            self.index = pc.Index(settings.PINECONE_INDEX_NAME)
            
            # Initialize vectorstore
            self.vectorstore = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                text_key="text"
            )
            
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None) -> bool:
        """Add documents to the vector database"""
        try:
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            
            # Create Document objects
            docs = []
            for i, doc in enumerate(documents):
                chunks = text_splitter.split_text(doc)
                for chunk in chunks:
                    metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                    docs.append(Document(page_content=chunk, metadata=metadata))
            
            # Add to vectorstore
            self.vectorstore.add_documents(docs)
            return True
            
        except Exception as e:
            print(f"Error adding documents to vector database: {e}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, score_threshold: float = None) -> List[Document]:
        """Search for similar documents"""
        try:
            if score_threshold:
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                # Filter by score threshold
                filtered_results = [doc for doc, score in results if score >= score_threshold]
                return filtered_results
            else:
                return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return []
    
    def update_knowledge_base(self, plant_name: str, care_info: str, source: str = "web_search") -> bool:
        """Update knowledge base with new plant care information"""
        try:
            metadata = {
                "plant_name": plant_name,
                "source": source,
                "type": "care_instructions"
            }
            
            return self.add_documents([care_info], [metadata])
        except Exception as e:
            print(f"Error updating knowledge base: {e}")
            return False
    
    def get_plant_care_info(self, plant_name: str, query: str = None) -> List[Document]:
        """Get plant care information from vector database"""
        try:
            # Create search query
            search_query = f"{plant_name} care instructions"
            if query:
                search_query += f" {query}"
            
            # Search with score threshold
            results = self.similarity_search(
                search_query, 
                k=5, 
                score_threshold=settings.MIN_RAG_SIMILARITY_SCORE
            )
            
            return results
        except Exception as e:
            print(f"Error retrieving plant care info: {e}")
            return []
    
    def initialize_with_sample_data(self):
        """Initialize vector database with sample plant care data"""
        sample_data = [
            {
                "text": "Monstera deliciosa care: Water when top inch of soil is dry. Prefers bright, indirect light. Humidity 40-60%. Temperature 65-80°F. Fertilize monthly during growing season.",
                "metadata": {"plant_name": "Monstera deliciosa", "source": "manual", "type": "care_instructions"}
            },
            {
                "text": "Snake plant (Sansevieria) care: Very drought tolerant, water every 2-3 weeks. Tolerates low light but prefers bright, indirect light. Low humidity requirements. Temperature 60-80°F.",
                "metadata": {"plant_name": "Sansevieria", "source": "manual", "type": "care_instructions"}
            },
            {
                "text": "Pothos care: Water when soil feels dry 1-2 inches down. Thrives in medium to bright, indirect light. Average home humidity is fine. Temperature 65-75°F. Very easy to care for.",
                "metadata": {"plant_name": "Pothos", "source": "manual", "type": "care_instructions"}
            },
            {
                "text": "Fiddle leaf fig care: Water when top inch of soil is dry, usually weekly. Needs bright, indirect light. Prefers humidity 30-65%. Temperature 60-75°F. Sensitive to overwatering.",
                "metadata": {"plant_name": "Ficus lyrata", "source": "manual", "type": "care_instructions"}
            }
        ]
        
        documents = [item["text"] for item in sample_data]
        metadatas = [item["metadata"] for item in sample_data]
        
        return self.add_documents(documents, metadatas)