# 🌱 AI Plant Care Assistant

A comprehensive AI-powered plant care assistant that helps you identify plants, monitor their health, provide care advice, and track growth over time.

## Features

- **🔍 Plant Identification**: Identify plants from images using AI
- **🩺 Health Assessment**: Detect diseases and assess plant health
- **💡 Care Advice**: Get personalized care recommendations
- **🌤️ Weather Integration**: Weather-based care adjustments
- **📅 Care Scheduling**: Automated care schedule generation
- **📈 Growth Tracking**: Monitor and predict plant growth
- **🧠 Knowledge Base**: RAG-powered plant care information
- **📊 Dashboard**: Comprehensive overview of all your plants

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Virtual environment (venv)

### 1. Clone and Setup Virtual Environment

```bash
# Navigate to your project directory
cd d:\Projects\AIPro\hackathon

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration

1. Copy the example environment file:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your API keys:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Plant.id API
PLANT_ID_API_KEY=your_plant_id_api_key

# OpenWeather API
OPENWEATHER_API_KEY=your_openweather_api_key

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=plant-care-knowledge

# Tavily Search API
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Required API Keys

You'll need to obtain API keys from:

- **Azure OpenAI**: [Azure Portal](https://portal.azure.com/)
- **Plant.id**: [Plant.id API](https://plant.id/)
- **OpenWeather**: [OpenWeatherMap](https://openweathermap.org/api)
- **Pinecone**: [Pinecone](https://www.pinecone.io/)
- **Tavily**: [Tavily Search](https://tavily.com/)

### 5. Run the Application

```bash
# Make sure virtual environment is activated
# Run the Streamlit app
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### Getting Started

1. **Add Your First Plant**: Use the sidebar to add a new plant with a name and optional image
2. **Identify Plant**: Upload an image to identify the plant species
3. **Health Check**: Assess plant health and detect potential issues
4. **Get Care Advice**: Receive personalized care recommendations
5. **Track Growth**: Record measurements and monitor progress

### Main Features

#### Plant Identification
- Upload clear images of your plant
- Get species identification with confidence scores
- View detailed plant information

#### Health Assessment
- Upload images showing plant condition
- Get health scores and disease detection
- Receive treatment recommendations

#### Care Advice
- Get personalized care instructions
- Weather-adjusted recommendations
- Schedule-based care planning

#### Growth Tracking
- Record height, leaf count, and other measurements
- Upload progress photos
- View growth trends and predictions

#### Knowledge Base
- Search plant care information
- Add custom care notes
- Browse by categories
- Web search integration for latest information

## Project Structure

```
hackathon/
├── agents/                 # AI agents for different tasks
│   ├── base_agent.py      # Base agent class
│   ├── plant_identifier.py # Plant identification
│   ├── disease_detector.py # Health assessment
│   ├── care_advisor.py    # Care recommendations
│   ├── weather_advisor.py # Weather integration
│   ├── schedule_manager.py # Care scheduling
│   ├── growth_tracker.py  # Growth monitoring
│   └── knowledge_augmenter.py # Knowledge base management
├── config/                # Configuration files
│   └── settings.py       # Environment and API settings
├── utils/                 # Utility functions
│   ├── image_utils.py    # Image processing
│   ├── api_helpers.py    # API integration helpers
│   └── vector_db.py      # Vector database operations
├── ui/                   # UI components (future expansion)
├── data/                 # Data storage
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure all API keys are correctly set in the `.env` file
2. **Import Errors**: Make sure virtual environment is activated and dependencies are installed
3. **Pinecone Connection**: Verify Pinecone environment and index name are correct
4. **Image Upload Issues**: Ensure images are in supported formats (JPG, PNG, WEBP)

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all API keys are valid and have sufficient credits
3. Ensure your virtual environment is properly activated
4. Check that all required dependencies are installed

## Contributing

This project is designed for the hackathon. Future enhancements could include:
- Mobile app integration
- IoT sensor integration
- Community features
- Advanced analytics
- Multi-language support

## License

This project is created for educational and hackathon purposes.