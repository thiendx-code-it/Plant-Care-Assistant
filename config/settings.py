import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables
load_dotenv()

class Settings:
    """Configuration settings for the Plant Care Assistant"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY_CHAT: str = os.getenv("AZURE_OPENAI_API_KEY_CHAT", "")
    AZURE_OPENAI_ENDPOINT_CHAT: str = os.getenv("AZURE_OPENAI_ENDPOINT_CHAT", "")
    AZURE_OPENAI_DEPLOYMENT_CHAT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", "")
    AZURE_OPENAI_API_VERSION: str = "2023-12-01-preview"
    
    AZURE_OPENAI_API_KEY_EMBED: str = os.getenv("AZURE_OPENAI_API_KEY_EMBED", "")
    AZURE_OPENAI_ENDPOINT_EMBED: str = os.getenv("AZURE_OPENAI_ENDPOINT_EMBED", "")
    AZURE_OPENAI_DEPLOYMENT_EMBED: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_EMBED", "")
    
    # External API Keys
    PLANT_ID_API_KEY: str = os.getenv("PLANT_ID_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "plant-care-knowledge")
    
    # Business Rules
    MIN_PLANT_ID_CONFIDENCE: float = 0.7
    MIN_RAG_SIMILARITY_SCORE: float = 0.8
    GROWTH_TRACKING_MIN_DAYS: int = 7
    
    # API Endpoints
    PLANT_ID_API_URL: str = "https://api.plant.id/v3/identification"
    PLANT_ID_HEALTH_URL: str = "https://api.plant.id/v3/health_assessment"
    OPENWEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5/weather"
    TAVILY_API_URL: str = "https://api.tavily.com/search"
    
    def validate_required_keys(self) -> list[str]:
        """Validate that all required API keys are present"""
        missing_keys = []
        required_keys = [
            ("AZURE_OPENAI_API_KEY_CHAT", self.AZURE_OPENAI_API_KEY_CHAT),
            ("AZURE_OPENAI_ENDPOINT_CHAT", self.AZURE_OPENAI_ENDPOINT_CHAT),
            ("AZURE_OPENAI_DEPLOYMENT_CHAT", self.AZURE_OPENAI_DEPLOYMENT_CHAT),
            ("AZURE_OPENAI_API_KEY_EMBED", self.AZURE_OPENAI_API_KEY_EMBED),
            ("AZURE_OPENAI_ENDPOINT_EMBED", self.AZURE_OPENAI_ENDPOINT_EMBED),
            ("AZURE_OPENAI_DEPLOYMENT_EMBED", self.AZURE_OPENAI_DEPLOYMENT_EMBED),
            ("PLANT_ID_API_KEY", self.PLANT_ID_API_KEY),
            ("OPENWEATHER_API_KEY", self.OPENWEATHER_API_KEY),
            ("PINECONE_API_KEY", self.PINECONE_API_KEY),
            ("TAVILY_API_KEY", self.TAVILY_API_KEY),
        ]
        
        for key_name, key_value in required_keys:
            if not key_value:
                missing_keys.append(key_name)
        
        return missing_keys

# Global settings instance
settings = Settings()