import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import agents
from agents.plant_identifier import PlantIdentifierAgent
from agents.disease_detector import DiseaseDetectorAgent
from agents.care_advisor import CareAdvisorAgent
from agents.weather_advisor import WeatherAdvisorAgent
from agents.schedule_manager import ScheduleManagerAgent
from agents.knowledge_augmenter import KnowledgeAugmenterAgent
from agents.growth_tracker import GrowthTrackerAgent

# Import utilities
from utils.vector_db import VectorDBManager
from utils.image_utils import validate_image_format
from config.settings import Settings
from chat_workflow import PlantCareWorkflow, ChatState

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Care Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 2rem;
}
.agent-card {
    background-color: #f0f8f0;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #2E8B57;
    margin: 1rem 0;
}
.success-message {
    background-color: #d4edda;
    color: #155724;
    padding: 0.75rem;
    border-radius: 5px;
    border: 1px solid #c3e6cb;
}
.error-message {
    background-color: #f8d7da;
    color: #721c24;
    padding: 0.75rem;
    border-radius: 5px;
    border: 1px solid #f5c6cb;
}
.info-box {
    background-color: #e7f3ff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'plant_data' not in st.session_state:
    st.session_state.plant_data = {}
if 'current_plant_id' not in st.session_state:
    st.session_state.current_plant_id = None
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# Initialize agents
@st.cache_resource
def initialize_agents():
    """Initialize all agents"""
    try:
        agents = {
            'plant_identifier': PlantIdentifierAgent(),
            'disease_detector': DiseaseDetectorAgent(),
            'care_advisor': CareAdvisorAgent(),
            'weather_advisor': WeatherAdvisorAgent(),
            'schedule_manager': ScheduleManagerAgent(),
            'knowledge_augmenter': KnowledgeAugmenterAgent(),
            'growth_tracker': GrowthTrackerAgent()
        }
        return agents
    except Exception as e:
        st.error(f"Error initializing agents: {e}")
        return None

@st.cache_resource
def initialize_vector_db():
    """Initialize vector database"""
    try:
        vector_db = VectorDBManager()
        return vector_db
    except Exception as e:
        st.error(f"Error initializing vector database: {e}")
        return None

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Care Assistant</h1>', unsafe_allow_html=True)
    st.markdown("*Your AI-powered companion for optimal plant care*")
    
    # Initialize components
    agents = initialize_agents()
    vector_db = initialize_vector_db()
    
    if not agents:
        st.error("Failed to initialize agents. Please check your configuration.")
        return
    
    if not vector_db:
        st.error("Failed to initialize vector database. Please check your Pinecone configuration.")
        return
    
    st.session_state.vector_db = vector_db
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🌿 Navigation")
        
        page = st.selectbox(
            "Choose a feature:",
            [
                "🏠 Home",
                "💬 AI Chat Assistant",
                "🔍 Plant Identification",
                "🩺 Health Assessment",
                "💡 Care Advice",
                "🌤️ Weather Impact",
                "📅 Care Schedule",
                "📈 Growth Tracking",
                "🧠 Knowledge Base",
                "📊 Dashboard"
            ]
        )
        
        st.markdown("---")
        
        # Plant selection
        st.subheader("🌱 Current Plant")
        plant_options = list(st.session_state.plant_data.keys()) + ["+ Add New Plant"]
        selected_plant = st.selectbox("Select plant:", plant_options)
        
        if selected_plant == "+ Add New Plant":
            new_plant_name = st.text_input("Enter plant name:")
            if st.button("Add Plant") and new_plant_name:
                plant_id = f"plant_{len(st.session_state.plant_data) + 1}"
                st.session_state.plant_data[plant_id] = {
                    "name": new_plant_name,
                    "created_at": datetime.now().isoformat(),
                    "identification": None,
                    "health_status": None,
                    "care_schedule": None,
                    "growth_data": []
                }
                st.session_state.current_plant_id = plant_id
                st.success(f"Added {new_plant_name}!")
                st.rerun()
        elif selected_plant and selected_plant != "+ Add New Plant":
            st.session_state.current_plant_id = selected_plant
        
        # Display current plant info
        if st.session_state.current_plant_id:
            plant_info = st.session_state.plant_data.get(st.session_state.current_plant_id, {})
            st.info(f"**Current:** {plant_info.get('name', 'Unknown')}")
    
    # Main content area
    if page == "🏠 Home":
        show_home_page()
    elif page == "💬 AI Chat Assistant":
        show_chat_page(vector_db)
    elif page == "🔍 Plant Identification":
        show_plant_identification_page(agents['plant_identifier'])
    elif page == "🩺 Health Assessment":
        show_health_assessment_page(agents['disease_detector'])
    elif page == "💡 Care Advice":
        show_care_advice_page(agents['care_advisor'], vector_db)
    elif page == "🌤️ Weather Impact":
        show_weather_impact_page(agents['weather_advisor'])
    elif page == "📅 Care Schedule":
        show_care_schedule_page(agents['schedule_manager'])
    elif page == "📈 Growth Tracking":
        show_growth_tracking_page(agents['growth_tracker'])
    elif page == "🧠 Knowledge Base":
        show_knowledge_base_page(agents['knowledge_augmenter'], vector_db)
    elif page == "📊 Dashboard":
        show_dashboard_page(agents)

def show_home_page():
    """Display home page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>🌟 Welcome to Your Plant Care Assistant!</h3>
        <p>This AI-powered application helps you take the best care of your plants with:</p>
        <ul>
            <li>🔍 <strong>Plant Identification</strong> - Identify your plants from photos</li>
            <li>🩺 <strong>Health Assessment</strong> - Detect diseases and health issues</li>
            <li>💡 <strong>Personalized Care Advice</strong> - Get tailored care recommendations</li>
            <li>🌤️ <strong>Weather-Aware Care</strong> - Adjust care based on weather conditions</li>
            <li>📅 <strong>Smart Scheduling</strong> - Never miss watering or fertilizing</li>
            <li>📈 <strong>Growth Tracking</strong> - Monitor your plant's development</li>
            <li>🧠 <strong>Knowledge Base</strong> - Access comprehensive plant care information</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        if st.session_state.plant_data:
            st.markdown("### 📊 Your Plant Collection")
            
            total_plants = len(st.session_state.plant_data)
            identified_plants = sum(1 for p in st.session_state.plant_data.values() if p.get('identification'))
            plants_with_schedules = sum(1 for p in st.session_state.plant_data.values() if p.get('care_schedule'))
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Plants", total_plants)
            with col2:
                st.metric("Identified", identified_plants)
            with col3:
                st.metric("With Schedules", plants_with_schedules)
        
        # Getting started
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Add a plant** using the sidebar
        2. **Identify your plant** by uploading a photo
        3. **Assess its health** to detect any issues
        4. **Get care advice** tailored to your plant
        5. **Set up a schedule** for optimal care
        6. **Track growth** over time
        """)

def show_plant_identification_page(agent):
    """Display plant identification page"""
    st.header("🔍 Plant Identification")
    st.markdown("Upload a photo of your plant to identify the species and get detailed information.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your plant for best results"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Plant Image", use_container_width=True)
            
            # Validate image
            if not validate_image_format(image):
                st.error("Invalid image format. Please upload a PNG or JPEG image.")
                return
    
    with col2:
        if uploaded_file:
            if st.button("🔍 Identify Plant", type="primary"):
                with st.spinner("Analyzing plant image..."):
                    # Prepare input data
                    from utils.image_utils import prepare_image_for_api
                    try:
                        image_base64 = prepare_image_for_api(uploaded_file)
                        input_data = {
                            "image_base64": image_base64,
                            "include_details": True
                        }
                    except ValueError as e:
                        st.error(f"Error processing image: {str(e)}")
                        return
                    
                    # Run identification
                    result = asyncio.run(agent.execute(input_data))
                    
                    if result.get("success"):
                        identification_data = result["data"]
                        
                        # Store identification data
                        st.session_state.plant_data[st.session_state.current_plant_id]["identification"] = identification_data
                        
                        # Display results
                        st.success("Plant identified successfully!")
                        
                        # Check if plant was successfully identified
                        if identification_data.get("identified"):
                            # Plant details
                            st.subheader(f"🌿 {identification_data.get('plant_name', 'Unknown Plant')}")
                            
                            if identification_data.get("scientific_name"):
                                st.markdown(f"**Scientific Name:** *{identification_data['scientific_name']}*")
                            
                            if identification_data.get("confidence"):
                                confidence = identification_data["confidence"] * 100
                                st.progress(confidence / 100)
                                st.caption(f"Confidence: {confidence:.1f}%")
                            
                            # Additional details
                            if identification_data.get("common_names"):
                                common_names = identification_data["common_names"]
                                if common_names:
                                    st.markdown(f"**Common Names:** {', '.join(common_names)}")
                            
                            if identification_data.get("family"):
                                st.markdown(f"**Family:** {identification_data['family']}")
                            
                            if identification_data.get("description"):
                                st.markdown(f"**Description:** {identification_data['description']}")
                            
                            # Show URL if available
                            if identification_data.get("url"):
                                st.markdown(f"**More Info:** [View Details]({identification_data['url']})")
                            
                            # Show similar images if available
                            if identification_data.get("similar_images"):
                                st.subheader("📸 Similar Images")
                                similar_images = identification_data["similar_images"]
                                if similar_images:
                                    cols = st.columns(min(len(similar_images), 3))
                                    for i, img in enumerate(similar_images[:3]):
                                        with cols[i]:
                                            if img.get("url"):
                                                st.image(img["url"], caption=f"Similar plant {i+1}", use_container_width=True)
                                                if img.get("citation"):
                                                    st.caption(img["citation"])
                            
                            # Show alternative suggestions if available
                            if identification_data.get("all_suggestions"):
                                suggestions = identification_data["all_suggestions"]
                                if len(suggestions) > 1:  # More than just the top result
                                    st.subheader("🔍 Alternative Suggestions")
                                    for i, suggestion in enumerate(suggestions[1:4], 1):  # Skip first (already shown) and limit to 3
                                        with st.expander(f"Option {i+1}: {suggestion.get('plant_name', 'Unknown')} ({suggestion.get('probability', 0)*100:.1f}%)"):
                                            if suggestion.get("scientific_name"):
                                                st.markdown(f"**Scientific Name:** *{suggestion['scientific_name']}*")
                                            if suggestion.get("common_names"):
                                                st.markdown(f"**Common Names:** {', '.join(suggestion['common_names'])}")
                        else:
                            # Plant not identified or low confidence
                            st.warning(identification_data.get("message", "Plant identification was not successful"))
                            
                            # Show suggestions if available even when not identified
                            if identification_data.get("suggestions"):
                                st.subheader("🤔 Possible Matches")
                                for i, suggestion in enumerate(identification_data["suggestions"], 1):
                                    with st.expander(f"Possibility {i}: {suggestion.get('plant_name', 'Unknown')} ({suggestion.get('probability', 0)*100:.1f}%)"):
                                        if suggestion.get("scientific_name"):
                                            st.markdown(f"**Scientific Name:** *{suggestion['scientific_name']}*")
                                        if suggestion.get("common_names"):
                                            st.markdown(f"**Common Names:** {', '.join(suggestion['common_names'])}")
                    else:
                        st.error(f"Identification failed: {result.get('error', 'Unknown error')}")
    
    # Text-based identification
    st.markdown("---")
    st.subheader("📝 Describe Your Plant")
    st.markdown("Can't upload an image? Describe your plant instead!")
    
    plant_description = st.text_area(
        "Describe your plant",
        placeholder="Describe the leaves, flowers, size, growing conditions, etc.",
        height=100
    )
    
    if st.button("🔍 Identify from Description") and plant_description:
        with st.spinner("Analyzing plant description..."):
            input_data = {
                "description": plant_description,
                "action": "identify_by_description"
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                identification_data = result["data"]
                st.session_state.plant_data[st.session_state.current_plant_id]["identification"] = identification_data
                
                st.success("Plant identified from description!")
                st.json(identification_data)
            else:
                st.error(f"Identification failed: {result.get('error', 'Unknown error')}")

def show_health_assessment_page(agent):
    """Display health assessment page"""
    st.header("🩺 Plant Health Assessment")
    st.markdown("Analyze your plant's health using both visual analysis and symptom description for comprehensive assessment.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    # Combined assessment form
    with st.form("health_assessment_form"):
        st.subheader("📸 Plant Image (Required)")
        uploaded_file = st.file_uploader(
            "Upload a photo showing your plant's current condition",
            type=['png', 'jpg', 'jpeg'],
            key="health_image",
            help="Clear photos help identify diseases, pests, and health issues more accurately"
        )
        
        st.subheader("📝 Symptom Description (Required)")
        symptoms = st.text_area(
            "Describe any symptoms or issues you've noticed",
            placeholder="e.g., yellowing leaves, brown spots, wilting, pest damage, unusual growth patterns...",
            height=120,
            help="Detailed descriptions complement visual analysis for better diagnosis"
        )
        
        # Additional context
        st.subheader("🌱 Additional Context (Optional)")
        col1, col2 = st.columns(2)
        
        with col1:
            plant_age = st.text_input(
                "Plant age/time since acquisition",
                placeholder="e.g., 6 months, 2 years, new plant"
            )
            
        with col2:
            recent_changes = st.text_input(
                "Recent changes in care/environment",
                placeholder="e.g., moved location, changed watering, repotted"
            )
        
        submitted = st.form_submit_button("🔬 Analyze Plant Health", type="primary")
    
    if submitted:
        if not uploaded_file:
            st.error("Please upload a plant image for visual analysis.")
            return
        
        if not symptoms.strip():
            st.error("Please describe the symptoms you've observed.")
            return
        
        # Show uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Plant Health Image", use_container_width=True)
        
        with col2:
            st.info("🔍 Analyzing both visual and textual information for comprehensive health assessment...")
        
        with st.spinner("Performing comprehensive health analysis..."):
            from utils.image_utils import prepare_image_for_api
            try:
                image_base64 = prepare_image_for_api(uploaded_file)
                input_data = {
                    "image_base64": image_base64,
                    "symptoms": symptoms,
                    "analysis_type": "comprehensive",
                    "plant_age": plant_age if plant_age else "",
                    "recent_changes": recent_changes if recent_changes else ""
                }
            except ValueError as e:
                st.error(f"Error processing image: {str(e)}")
                return
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                health_data = result["data"]
                st.session_state.plant_data[st.session_state.current_plant_id]["health_status"] = health_data
                
                display_health_results(health_data)
            else:
                st.error(f"Health analysis failed: {result.get('error', 'Unknown error')}")

def display_health_results(health_data):
    """Display health assessment results"""
    st.success("Health analysis completed!")
    
    # Health score
    if health_data.get("health_score"):
        score = health_data["health_score"] * 100
        st.metric("Health Score", f"{score:.1f}%")
        
        # Health status indicator
        if score >= 80:
            st.success("🌟 Your plant appears healthy!")
        elif score >= 60:
            st.warning("⚠️ Some health concerns detected")
        else:
            st.error("🚨 Significant health issues detected")
    
    # Diseases detected
    if health_data.get("diseases"):
        st.subheader("🦠 Diseases Detected")
        for disease in health_data["diseases"]:
            with st.expander(f"🔍 {disease.get('name', 'Unknown Disease')}"):
                if disease.get("probability"):
                    prob = disease["probability"] * 100
                    st.progress(prob / 100)
                    st.caption(f"Probability: {prob:.1f}%")
                
                if disease.get("description"):
                    st.markdown(f"**Description:** {disease['description']}")
                
                if disease.get("treatment"):
                    st.markdown(f"**Treatment:** {disease['treatment']}")
    
    # Pests detected
    if health_data.get("pests"):
        st.subheader("🐛 Pests Detected")
        for pest in health_data["pests"]:
            with st.expander(f"🔍 {pest.get('name', 'Unknown Pest')}"):
                if pest.get("probability"):
                    prob = pest["probability"] * 100
                    st.progress(prob / 100)
                    st.caption(f"Probability: {prob:.1f}%")
                
                if pest.get("description"):
                    st.markdown(f"**Description:** {pest['description']}")
                
                if pest.get("treatment"):
                    st.markdown(f"**Treatment:** {pest['treatment']}")
    
    # Recommendations
    if health_data.get("recommendations"):
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(health_data["recommendations"], 1):
            st.markdown(f"{i}. {rec}")

def show_care_advice_page(agent, vector_db):
    """Display care advice page"""
    st.header("💡 Personalized Care Advice")
    st.markdown("Get tailored care recommendations based on your plant's needs and current conditions.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    plant_data = st.session_state.plant_data.get(st.session_state.current_plant_id, {})
    
    # Input form
    with st.form("care_advice_form"):
        st.subheader("🌱 Plant Information")
        
        # Plant identification
        plant_name = st.text_input(
            "Plant Name",
            value=plant_data.get("identification", {}).get("plant_name", "")
        )
        
        # Care query
        care_query = st.text_area(
            "What would you like to know about caring for your plant?",
            placeholder="e.g., How often should I water? What fertilizer to use? Light requirements?",
            height=100
        )
        
        # Image upload for visual context
        st.subheader("📸 Visual Assessment (Optional but Recommended)")
        uploaded_file = st.file_uploader(
            "Upload a current photo of your plant for better care advice",
            type=['png', 'jpg', 'jpeg'],
            key="care_advice_image",
            help="Providing an image helps the AI analyze your plant's current condition for more accurate advice"
        )
        
        # Image description
        image_description = st.text_area(
            "Describe what you see in the image (optional)",
            placeholder="e.g., yellowing leaves, brown spots, drooping stems, new growth...",
            height=60
        )
        
        # Current conditions
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location (for weather data)", placeholder="e.g., New York, NY")
        with col2:
            season = st.selectbox("Current Season", ["Spring", "Summer", "Fall", "Winter"])
        
        # Plant health status
        health_issues = st.text_area(
            "Any current health issues?",
            placeholder="Describe any problems you've noticed...",
            height=80
        )
        
        submitted = st.form_submit_button("💡 Get Care Advice", type="primary")
    
    if submitted and plant_name and care_query:
        with st.spinner("Generating personalized care advice..."):
            # Prepare input data
            input_data = {
                "plant_name": plant_name,
                "specific_query": care_query,
                "health_issues": [health_issues] if health_issues else [],
                "weather_data": {"location": location, "season": season} if location else {}
            }
            
            # Add image data if provided
            if uploaded_file:
                from utils.image_utils import prepare_image_for_api
                try:
                    image_base64 = prepare_image_for_api(uploaded_file)
                    input_data["image_base64"] = image_base64
                    input_data["image_description"] = image_description
                    
                    # Show uploaded image
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        image = Image.open(uploaded_file)
                        st.image(image, caption="Plant Image for Analysis", use_container_width=True)
                    with col2:
                        st.info("🔍 Image will be analyzed for visual health assessment and care recommendations.")
                        
                except Exception as e:
                    st.warning(f"Could not process image: {e}. Proceeding with text-based advice only.")
            
            # Get care advice
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                advice_data = result["data"]
                
                # Display advice
                success_msg = "Care advice generated successfully!"
                if advice_data.get("visual_analysis_included"):
                    success_msg += " 📸 Visual analysis included for enhanced recommendations."
                st.success(success_msg)
                
                # Main advice
                if advice_data.get("care_advice"):
                    st.subheader("🌿 Care Recommendations")
                    st.markdown(advice_data["care_advice"])
                
                # Specific care aspects
                if advice_data.get("care_aspects"):
                    st.subheader("📋 Detailed Care Guide")
                    
                    aspects = advice_data["care_aspects"]
                    
                    # Create tabs for different aspects
                    tabs = st.tabs(["💧 Watering", "🌞 Light", "🌡️ Temperature", "🌱 Fertilizing", "🪴 Soil"])
                    
                    with tabs[0]:  # Watering
                        if aspects.get("watering"):
                            st.markdown(aspects["watering"])
                    
                    with tabs[1]:  # Light
                        if aspects.get("light"):
                            st.markdown(aspects["light"])
                    
                    with tabs[2]:  # Temperature
                        if aspects.get("temperature"):
                            st.markdown(aspects["temperature"])
                    
                    with tabs[3]:  # Fertilizing
                        if aspects.get("fertilizing"):
                            st.markdown(aspects["fertilizing"])
                    
                    with tabs[4]:  # Soil
                        if aspects.get("soil"):
                            st.markdown(aspects["soil"])
                
                # Seasonal advice
                if advice_data.get("seasonal_advice"):
                    st.subheader(f"🍂 {season} Care Tips")
                    st.info(advice_data["seasonal_advice"])
                
                # Quick tips
                if advice_data.get("quick_tips"):
                    st.subheader("⚡ Quick Tips")
                    for tip in advice_data["quick_tips"]:
                        st.markdown(f"• {tip}")
            else:
                st.error(f"Failed to generate care advice: {result.get('error', 'Unknown error')}")

def show_weather_impact_page(agent):
    """Display weather impact page"""
    st.header("🌤️ Weather Impact Analysis")
    st.markdown("Understand how current weather conditions affect your plant care needs.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    # Location input
    location = st.text_input(
        "Enter your location",
        placeholder="e.g., New York, NY or London, UK",
        help="We'll fetch current weather data for your location"
    )
    
    if st.button("🌤️ Get Weather Analysis", type="primary") and location:
        with st.spinner("Fetching weather data and analyzing impact..."):
            input_data = {
                "location": location,
                "plant_data": st.session_state.plant_data.get(st.session_state.current_plant_id, {})
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                weather_data = result["data"]
                
                # Display current weather
                if weather_data.get("current_weather"):
                    st.subheader("🌡️ Current Weather")
                    weather = weather_data["current_weather"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Temperature", f"{weather.get('temperature', 'N/A')}°C")
                    with col2:
                        st.metric("Humidity", f"{weather.get('humidity', 'N/A')}%")
                    with col3:
                        st.metric("Pressure", f"{weather.get('pressure', 'N/A')} hPa")
                    with col4:
                        st.metric("Wind Speed", f"{weather.get('wind_speed', 'N/A')} m/s")
                    
                    if weather.get("description"):
                        st.info(f"**Conditions:** {weather['description'].title()}")
                
                # Weather impact analysis
                if weather_data.get("impact_analysis"):
                    st.subheader("📊 Impact Analysis")
                    impact = weather_data["impact_analysis"]
                    
                    # Overall stress level
                    if impact.get("overall_stress_level"):
                        stress_level = impact["overall_stress_level"]
                        stress_color = {
                            "low": "green",
                            "moderate": "orange",
                            "high": "red"
                        }.get(stress_level, "gray")
                        
                        st.markdown(f"**Plant Stress Level:** :{stress_color}[{stress_level.upper()}]")
                    
                    # Specific impacts
                    if impact.get("temperature_impact"):
                        st.markdown(f"**Temperature Impact:** {impact['temperature_impact']}")
                    
                    if impact.get("humidity_impact"):
                        st.markdown(f"**Humidity Impact:** {impact['humidity_impact']}")
                    
                    if impact.get("precipitation_impact"):
                        st.markdown(f"**Precipitation Impact:** {impact['precipitation_impact']}")
                
                # Care adjustments
                if weather_data.get("care_adjustments"):
                    st.subheader("🔧 Recommended Care Adjustments")
                    adjustments = weather_data["care_adjustments"]
                    
                    for adjustment in adjustments:
                        st.markdown(f"• {adjustment}")
                
                # Schedule notes
                if weather_data.get("schedule_notes"):
                    st.subheader("📅 Schedule Adjustments")
                    for note in weather_data["schedule_notes"]:
                        st.info(note)
            else:
                st.error(f"Weather analysis failed: {result.get('error', 'Unknown error')}")

def show_care_schedule_page(agent):
    """Display care schedule page"""
    st.header("📅 Smart Care Schedule")
    st.markdown("Generate and manage personalized care schedules for your plants.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    plant_data = st.session_state.plant_data.get(st.session_state.current_plant_id, {})
    plant_name = plant_data.get("name", "Unknown Plant")
    
    # Schedule generation form
    with st.form("schedule_form"):
        st.subheader(f"🌱 Generate Schedule for {plant_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User preferences
            st.markdown("**Your Preferences**")
            watering_time = st.selectbox(
                "Preferred watering time",
                ["morning", "afternoon", "evening"]
            )
            
            watering_frequency = st.selectbox(
                "Watering frequency preference",
                ["as_recommended", "more_frequent", "less_frequent"]
            )
            
            organic_only = st.checkbox("Use only organic fertilizers")
        
        with col2:
            # Environmental conditions
            st.markdown("**Current Conditions**")
            location = st.text_input("Location (for weather adjustment)")
            indoor_outdoor = st.selectbox("Plant location", ["indoor", "outdoor"])
            
            # Health considerations
            health_issues = st.text_area(
                "Any health issues to consider?",
                height=80
            )
        
        generate_schedule = st.form_submit_button("📅 Generate Schedule", type="primary")
    
    if generate_schedule:
        with st.spinner("Generating personalized care schedule..."):
            # Prepare input data
            input_data = {
                "plant_name": plant_name,
                "care_info": plant_data.get("identification", {}).get("care_info", {}),
                "health_status": plant_data.get("health_status", {}),
                "user_preferences": {
                    "watering_time": watering_time,
                    "watering_frequency_adjustment": watering_frequency,
                    "organic_only": organic_only
                },
                "location": location,
                "environment": indoor_outdoor
            }
            
            # Add weather data if location provided
            if location:
                # You could fetch weather data here
                pass
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                schedule_data = result["data"]
                st.session_state.plant_data[st.session_state.current_plant_id]["care_schedule"] = schedule_data
                
                display_care_schedule(schedule_data)
            else:
                st.error(f"Schedule generation failed: {result.get('error', 'Unknown error')}")
    
    # Display existing schedule if available
    existing_schedule = plant_data.get("care_schedule")
    if existing_schedule and not generate_schedule:
        st.subheader("📋 Current Schedule")
        display_care_schedule(existing_schedule)

def display_care_schedule(schedule_data):
    """Display care schedule data"""
    st.success("Care schedule generated successfully!")
    
    # Schedule summary
    if schedule_data.get("schedule_summary"):
        st.subheader("📊 Schedule Summary")
        summary = schedule_data["schedule_summary"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Watering", summary.get("watering_frequency", "N/A"))
        with col2:
            st.metric("Fertilizing", summary.get("fertilizing_frequency", "N/A"))
        with col3:
            st.metric("Monitoring", summary.get("monitoring_frequency", "N/A"))
    
    # Next actions
    if schedule_data.get("next_actions"):
        st.subheader("⚡ Next Actions")
        for action in schedule_data["next_actions"]:
            priority_color = {
                "high": "🔴",
                "medium": "🟡",
                "low": "🟢"
            }.get(action.get("priority", "low"), "⚪")
            
            days_until = action.get("due_in_days", 0)
            if days_until <= 0:
                due_text = "Due today!"
            elif days_until == 1:
                due_text = "Due tomorrow"
            else:
                due_text = f"Due in {days_until} days"
            
            st.markdown(f"{priority_color} **{action.get('description', 'Unknown action')}** - {due_text}")
    
    # Detailed schedule
    if schedule_data.get("schedule"):
        st.subheader("📅 Detailed Schedule")
        schedule = schedule_data["schedule"]
        
        # Create tabs for different care aspects
        tabs = st.tabs(["💧 Watering", "🌱 Fertilizing", "👀 Monitoring", "✂️ Maintenance"])
        
        with tabs[0]:  # Watering
            if schedule.get("watering"):
                watering = schedule["watering"]
                st.markdown(f"**Frequency:** Every {watering.get('frequency_days', 'N/A')} days")
                st.markdown(f"**Amount:** {watering.get('amount', 'N/A')}")
                st.markdown(f"**Method:** {watering.get('method', 'N/A')}")
                st.markdown(f"**Time:** {watering.get('time_of_day', 'N/A')}")
                st.markdown(f"**Next watering:** {watering.get('next_watering', 'N/A')}")
                
                if watering.get("notes"):
                    st.markdown("**Notes:**")
                    for note in watering["notes"]:
                        st.markdown(f"• {note}")
        
        with tabs[1]:  # Fertilizing
            if schedule.get("fertilizing"):
                fertilizing = schedule["fertilizing"]
                st.markdown(f"**Frequency:** Every {fertilizing.get('frequency_days', 'N/A')} days")
                st.markdown(f"**Type:** {fertilizing.get('type', 'N/A')}")
                st.markdown(f"**Strength:** {fertilizing.get('strength', 'N/A')}")
                st.markdown(f"**Next fertilizing:** {fertilizing.get('next_fertilizing', 'N/A')}")
                
                if fertilizing.get("notes"):
                    st.markdown("**Notes:**")
                    for note in fertilizing["notes"]:
                        st.markdown(f"• {note}")
        
        with tabs[2]:  # Monitoring
            if schedule.get("monitoring"):
                monitoring = schedule["monitoring"]
                st.markdown(f"**Frequency:** Every {monitoring.get('frequency_days', 'N/A')} days")
                st.markdown(f"**Next check:** {monitoring.get('next_check', 'N/A')}")
                
                if monitoring.get("checks"):
                    st.markdown("**What to check:**")
                    for check in monitoring["checks"]:
                        st.markdown(f"• {check}")
        
        with tabs[3]:  # Maintenance
            if schedule.get("maintenance"):
                maintenance = schedule["maintenance"]
                
                if maintenance.get("pruning"):
                    pruning = maintenance["pruning"]
                    st.markdown("**Pruning:**")
                    st.markdown(f"• Frequency: Every {pruning.get('frequency_days', 'N/A')} days")
                    st.markdown(f"• Next pruning: {pruning.get('next_pruning', 'N/A')}")
                    st.markdown(f"• Type: {pruning.get('type', 'N/A')}")
                
                if maintenance.get("repotting"):
                    repotting = maintenance["repotting"]
                    st.markdown("**Repotting:**")
                    st.markdown(f"• Frequency: Every {repotting.get('frequency_days', 'N/A')} days")
                    st.markdown(f"• Next repotting: {repotting.get('next_repotting', 'N/A')}")
                    st.markdown(f"• Best season: {repotting.get('season', 'N/A')}")
    
    # Calendar events
    if schedule_data.get("calendar_events"):
        st.subheader("📆 Upcoming Events")
        events_df = pd.DataFrame(schedule_data["calendar_events"])
        
        if not events_df.empty:
            # Sort by date
            events_df['date'] = pd.to_datetime(events_df['date'])
            events_df = events_df.sort_values('date')
            
            # Display as table
            display_df = events_df[['date', 'type', 'title', 'description']].copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df, use_container_width=True)

def show_growth_tracking_page(agent):
    """Display growth tracking page"""
    st.header("📈 Growth Tracking")
    st.markdown("Monitor and analyze your plant's growth over time.")
    
    if not st.session_state.current_plant_id:
        st.warning("Please select or add a plant first using the sidebar.")
        return
    
    plant_data = st.session_state.plant_data.get(st.session_state.current_plant_id, {})
    plant_name = plant_data.get("name", "Unknown Plant")
    
    # Action selection
    action = st.selectbox(
        "What would you like to do?",
        [
            "📏 Record New Measurement",
            "📊 View Growth Analysis",
            "📸 Compare Images",
            "📋 Generate Growth Report",
            "🔮 Growth Predictions"
        ]
    )
    
    if action == "📏 Record New Measurement":
        show_record_measurement_form(agent, plant_name)
    elif action == "📊 View Growth Analysis":
        show_growth_analysis(agent)
    elif action == "📸 Compare Images":
        show_image_comparison(agent)
    elif action == "📋 Generate Growth Report":
        show_growth_report(agent)
    elif action == "🔮 Growth Predictions":
        show_growth_predictions(agent)

def show_record_measurement_form(agent, plant_name):
    """Show form for recording new measurements"""
    st.subheader(f"📏 Record Measurement for {plant_name}")
    
    with st.form("measurement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Physical Measurements**")
            height = st.number_input("Height (cm)", min_value=0.0, step=0.1)
            width = st.number_input("Width (cm)", min_value=0.0, step=0.1)
            leaf_count = st.number_input("Number of leaves", min_value=0, step=1)
            stem_thickness = st.number_input("Stem thickness (mm)", min_value=0.0, step=0.1)
        
        with col2:
            st.markdown("**Growth Stage & Condition**")
            growth_stage = st.selectbox(
                "Growth stage",
                ["seedling", "juvenile", "mature", "flowering", "fruiting"]
            )
            
            new_growth = st.checkbox("New growth observed")
            flowering = st.checkbox("Flowering")
            fruiting = st.checkbox("Fruiting")
        
        # Environmental conditions
        st.markdown("**Environmental Conditions**")
        col3, col4 = st.columns(2)
        
        with col3:
            temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1)
            humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, step=1)
        
        with col4:
            light_hours = st.number_input("Daily light hours", min_value=0.0, max_value=24.0, step=0.5)
            watered_today = st.checkbox("Watered today")
        
        # Image upload
        st.markdown("**Growth Photo (Optional)**")
        growth_image = st.file_uploader(
            "Upload a photo of your plant",
            type=['png', 'jpg', 'jpeg'],
            key="growth_image"
        )
        
        # Notes
        notes = st.text_area(
            "Notes",
            placeholder="Any observations about your plant's condition, changes, or care actions taken...",
            height=100
        )
        
        # Care actions
        care_actions = st.multiselect(
            "Care actions taken today",
            ["watering", "fertilizing", "pruning", "repotting", "pest_treatment", "disease_treatment"]
        )
        
        submitted = st.form_submit_button("📏 Record Measurement", type="primary")
    
    if submitted:
        with st.spinner("Recording measurement..."):
            # Prepare measurement data
            measurements = {}
            if height > 0:
                measurements["height"] = height
            if width > 0:
                measurements["width"] = width
            if leaf_count > 0:
                measurements["leaf_count"] = leaf_count
            if stem_thickness > 0:
                measurements["stem_thickness"] = stem_thickness
            
            measurements["growth_stage"] = growth_stage
            measurements["new_growth"] = new_growth
            measurements["flowering"] = flowering
            measurements["fruiting"] = fruiting
            
            environmental_conditions = {}
            if temperature != 0:
                environmental_conditions["temperature"] = temperature
            if humidity > 0:
                environmental_conditions["humidity"] = humidity
            if light_hours > 0:
                environmental_conditions["light_hours"] = light_hours
            
            input_data = {
                "action": "record_measurement",
                "plant_id": st.session_state.current_plant_id,
                "measurements": measurements,
                "environmental_conditions": environmental_conditions,
                "notes": notes,
                "care_actions": care_actions,
                "image_data": growth_image
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                measurement_data = result["data"]
                st.success("Measurement recorded successfully!")
                
                # Display growth analysis if available
                if measurement_data.get("growth_analysis"):
                    st.subheader("📊 Growth Analysis")
                    analysis = measurement_data["growth_analysis"]
                    
                    if analysis.get("recent_trend"):
                        st.json(analysis["recent_trend"])
                
                # Display recommendations
                if measurement_data.get("recommendations"):
                    st.subheader("💡 Recommendations")
                    for rec in measurement_data["recommendations"]:
                        st.markdown(f"• {rec}")
            else:
                st.error(f"Failed to record measurement: {result.get('error', 'Unknown error')}")

def show_growth_analysis(agent):
    """Show growth analysis"""
    st.subheader("📊 Growth Analysis")
    
    time_period = st.selectbox(
        "Analysis period",
        [7, 14, 30, 60, 90],
        format_func=lambda x: f"Last {x} days"
    )
    
    if st.button("📊 Analyze Growth"):
        with st.spinner("Analyzing growth patterns..."):
            input_data = {
                "action": "analyze_growth",
                "plant_id": st.session_state.current_plant_id,
                "time_period": time_period
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                analysis_data = result["data"]
                
                # Display analysis results
                st.success(f"Growth analysis completed for {time_period} days!")
                
                # Growth trends
                if analysis_data.get("growth_trends"):
                    st.subheader("📈 Growth Trends")
                    trends = analysis_data["growth_trends"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if trends.get("height"):
                            height_trend = trends["height"]
                            st.metric(
                                "Height Trend",
                                height_trend.get("trend", "N/A"),
                                f"{height_trend.get('change', 0):.1f} cm"
                            )
                    
                    with col2:
                        if trends.get("leaf_count"):
                            leaf_trend = trends["leaf_count"]
                            st.metric(
                                "Leaf Count Trend",
                                leaf_trend.get("trend", "N/A"),
                                f"{leaf_trend.get('change', 0)} leaves"
                            )
                
                # Growth rate
                if analysis_data.get("growth_rate"):
                    st.subheader("⚡ Growth Rate")
                    rate = analysis_data["growth_rate"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if rate.get("height_per_week"):
                            st.metric("Height Growth", f"{rate['height_per_week']:.2f} cm/week")
                    with col2:
                        if rate.get("leaves_per_week"):
                            st.metric("Leaf Growth", f"{rate['leaves_per_week']:.1f} leaves/week")
                
                # Health indicators
                if analysis_data.get("health_indicators"):
                    st.subheader("🩺 Health Indicators")
                    health = analysis_data["health_indicators"]
                    
                    st.markdown(f"**Overall Trend:** {health.get('overall_trend', 'Unknown')}")
                    
                    if health.get("positive_signs"):
                        st.success("**Positive Signs:**")
                        for sign in health["positive_signs"]:
                            st.markdown(f"• {sign}")
                    
                    if health.get("concerns"):
                        st.warning("**Concerns:**")
                        for concern in health["concerns"]:
                            st.markdown(f"• {concern}")
                
                # AI insights
                if analysis_data.get("ai_insights"):
                    st.subheader("🤖 AI Insights")
                    st.markdown(analysis_data["ai_insights"])
            else:
                st.error(f"Growth analysis failed: {result.get('error', 'Unknown error')}")

def show_image_comparison(agent):
    """Show image comparison feature"""
    st.subheader("📸 Image Comparison")
    
    current_image = st.file_uploader(
        "Upload current plant image",
        type=['png', 'jpg', 'jpeg'],
        key="comparison_image"
    )
    
    comparison_period = st.selectbox(
        "Compare with images from",
        [7, 14, 30, 60],
        format_func=lambda x: f"{x} days ago"
    )
    
    if st.button("📸 Compare Images") and current_image:
        with st.spinner("Comparing images..."):
            input_data = {
                "action": "compare_images",
                "plant_id": st.session_state.current_plant_id,
                "current_image": current_image,
                "comparison_period": comparison_period
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                comparison_data = result["data"]
                
                st.success("Image comparison completed!")
                
                # Display comparison results
                if comparison_data.get("visual_changes"):
                    st.subheader("👁️ Visual Changes Detected")
                    changes = comparison_data["visual_changes"]
                    
                    if changes.get("comparison_analysis"):
                        st.markdown(changes["comparison_analysis"])
                
                # Growth indicators
                if comparison_data.get("growth_indicators"):
                    st.subheader("📈 Growth Indicators")
                    for indicator in comparison_data["growth_indicators"]:
                        st.markdown(f"• {indicator}")
                
                # Recommendations
                if comparison_data.get("recommendations"):
                    st.subheader("💡 Recommendations")
                    for rec in comparison_data["recommendations"]:
                        st.markdown(f"• {rec}")
            else:
                st.error(f"Image comparison failed: {result.get('error', 'Unknown error')}")

def show_growth_report(agent):
    """Show comprehensive growth report"""
    st.subheader("📋 Comprehensive Growth Report")
    
    report_period = st.selectbox(
        "Report period",
        [30, 60, 90, 180],
        format_func=lambda x: f"Last {x} days"
    )
    
    if st.button("📋 Generate Report"):
        with st.spinner("Generating comprehensive growth report..."):
            input_data = {
                "action": "get_growth_report",
                "plant_id": st.session_state.current_plant_id,
                "report_period": report_period
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                report_data = result["data"]
                
                st.success(f"Growth report generated for {report_period} days!")
                
                # Report summary
                if report_data.get("summary"):
                    st.subheader("📊 Summary")
                    summary = report_data["summary"]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Measurements", summary.get("total_measurements", 0))
                    with col2:
                        st.metric("Tracking Period", summary.get("tracking_period", "N/A"))
                    with col3:
                        st.metric("Overall Progress", "Positive" if summary.get("overall_progress") else "Stable")
                
                # Detailed analysis
                if report_data.get("detailed_analysis"):
                    st.subheader("🔍 Detailed Analysis")
                    analysis = report_data["detailed_analysis"]
                    
                    # Create tabs for different aspects
                    tabs = st.tabs(["📈 Metrics", "📊 Trends", "🩺 Health", "🌱 Care Impact"])
                    
                    with tabs[0]:  # Metrics
                        if analysis.get("growth_metrics"):
                            st.json(analysis["growth_metrics"])
                    
                    with tabs[1]:  # Trends
                        if analysis.get("trend_analysis"):
                            st.json(analysis["trend_analysis"])
                    
                    with tabs[2]:  # Health
                        if analysis.get("health_progression"):
                            st.json(analysis["health_progression"])
                    
                    with tabs[3]:  # Care Impact
                        if analysis.get("care_impact"):
                            st.json(analysis["care_impact"])
                
                # Milestones
                if report_data.get("milestones"):
                    st.subheader("🏆 Milestones")
                    milestones = report_data["milestones"]
                    
                    if milestones.get("achieved"):
                        st.success("**Achieved:**")
                        for milestone in milestones["achieved"]:
                            st.markdown(f"✅ {milestone}")
                    
                    if milestones.get("upcoming"):
                        st.info("**Upcoming:**")
                        for milestone in milestones["upcoming"]:
                            st.markdown(f"🎯 {milestone}")
                
                # Predictions
                if report_data.get("predictions"):
                    st.subheader("🔮 Growth Predictions")
                    predictions = report_data["predictions"]
                    
                    if predictions.get("prediction_text"):
                        st.markdown(predictions["prediction_text"])
                
                # Charts
                if report_data.get("charts_data"):
                    st.subheader("📈 Growth Charts")
                    charts_data = report_data["charts_data"]
                    
                    # Create growth charts
                    if charts_data.get("height_over_time") and charts_data.get("measurement_dates"):
                        # Height chart
                        height_data = [
                            (date, height) for date, height in zip(
                                charts_data["measurement_dates"],
                                charts_data["height_over_time"]
                            ) if height is not None
                        ]
                        
                        if height_data:
                            df_height = pd.DataFrame(height_data, columns=["Date", "Height (cm)"])
                            df_height["Date"] = pd.to_datetime(df_height["Date"])
                            
                            fig_height = px.line(
                                df_height,
                                x="Date",
                                y="Height (cm)",
                                title="Height Growth Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig_height, use_container_width=True)
                    
                    if charts_data.get("leaf_count_over_time") and charts_data.get("measurement_dates"):
                        # Leaf count chart
                        leaf_data = [
                            (date, count) for date, count in zip(
                                charts_data["measurement_dates"],
                                charts_data["leaf_count_over_time"]
                            ) if count is not None
                        ]
                        
                        if leaf_data:
                            df_leaves = pd.DataFrame(leaf_data, columns=["Date", "Leaf Count"])
                            df_leaves["Date"] = pd.to_datetime(df_leaves["Date"])
                            
                            fig_leaves = px.line(
                                df_leaves,
                                x="Date",
                                y="Leaf Count",
                                title="Leaf Count Over Time",
                                markers=True
                            )
                            st.plotly_chart(fig_leaves, use_container_width=True)
            else:
                st.error(f"Report generation failed: {result.get('error', 'Unknown error')}")

def show_growth_predictions(agent):
    """Show growth predictions"""
    st.subheader("🔮 Growth Predictions")
    
    prediction_period = st.selectbox(
        "Prediction period",
        [30, 60, 90],
        format_func=lambda x: f"Next {x} days"
    )
    
    if st.button("🔮 Generate Predictions"):
        with st.spinner("Generating growth predictions..."):
            input_data = {
                "action": "predict_growth",
                "plant_id": st.session_state.current_plant_id,
                "prediction_period": prediction_period
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                prediction_data = result["data"]
                
                st.success(f"Growth predictions generated for {prediction_period} days!")
                
                # Predictions
                if prediction_data.get("predictions"):
                    st.subheader("📊 Predicted Growth")
                    predictions = prediction_data["predictions"]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if predictions.get("height_prediction"):
                            height_pred = predictions["height_prediction"]
                            st.metric(
                                "Predicted Height",
                                f"{height_pred.get('predicted_value', 0):.1f} cm",
                                f"+{height_pred.get('growth_amount', 0):.1f} cm"
                            )
                    
                    with col2:
                        if predictions.get("leaf_prediction"):
                            leaf_pred = predictions["leaf_prediction"]
                            st.metric(
                                "Predicted Leaves",
                                f"{leaf_pred.get('predicted_value', 0)} leaves",
                                f"+{leaf_pred.get('growth_amount', 0)} leaves"
                            )
                
                # Growth factors
                if prediction_data.get("growth_factors"):
                    st.subheader("🌱 Growth Factors")
                    factors = prediction_data["growth_factors"]
                    
                    for factor, impact in factors.items():
                        impact_color = {
                            "positive": "green",
                            "negative": "red",
                            "neutral": "gray"
                        }.get(impact.get("type", "neutral"), "gray")
                        
                        st.markdown(f"**{factor.title()}:** :{impact_color}[{impact.get('description', 'No impact')}]")
                
                # Recommendations
                if prediction_data.get("recommendations"):
                    st.subheader("💡 Optimization Recommendations")
                    for rec in prediction_data["recommendations"]:
                        st.markdown(f"• {rec}")
            else:
                st.error(f"Prediction generation failed: {result.get('error', 'Unknown error')}")

def show_knowledge_base_page(agent, vector_db):
    """Display knowledge base management page"""
    st.header("🧠 Knowledge Base")
    st.markdown("Manage and augment the plant care knowledge base.")
    
    # Action selection
    action = st.selectbox(
        "What would you like to do?",
        [
            "🔍 Search Knowledge Base",
            "➕ Add New Information",
            "🌐 Web Search & Update",
            "📚 Browse Categories"
        ]
    )
    
    if action == "🔍 Search Knowledge Base":
        show_knowledge_search(vector_db)
    elif action == "➕ Add New Information":
        show_add_knowledge(vector_db)
    elif action == "🌐 Web Search & Update":
        show_web_search_update(agent, vector_db)
    elif action == "📚 Browse Categories":
        show_browse_categories(vector_db)

def show_knowledge_search(vector_db):
    """Show knowledge base search"""
    st.subheader("🔍 Search Knowledge Base")
    
    search_query = st.text_input(
        "Search for plant care information",
        placeholder="e.g., watering frequency for succulents, fertilizer for roses..."
    )
    
    if st.button("🔍 Search") and search_query:
        with st.spinner("Searching knowledge base..."):
            try:
                results = vector_db.similarity_search(search_query, k=5)
                
                if results:
                    st.success(f"Found {len(results)} relevant documents")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Result {i}"):
                            # Handle both Document objects and dictionaries
                            if hasattr(result, 'page_content'):
                                content = result.page_content
                                metadata = getattr(result, 'metadata', {})
                            elif isinstance(result, dict):
                                content = result.get('content', str(result))
                                metadata = result.get('metadata', {})
                            else:
                                content = str(result)
                                metadata = {}
                            
                            st.markdown(content)
                            
                            if metadata:
                                st.json(metadata)
                else:
                    st.warning("No relevant information found. Consider adding more content to the knowledge base.")
            except Exception as e:
                st.error(f"Search failed: {e}")

def show_add_knowledge(vector_db):
    """Show form to add new knowledge"""
    st.subheader("➕ Add New Information")
    
    with st.form("add_knowledge_form"):
        title = st.text_input("Title", placeholder="e.g., Watering Guide for Succulents")
        
        category = st.selectbox(
            "Category",
            ["watering", "fertilizing", "light_requirements", "soil", "diseases", "pests", "general_care"]
        )
        
        plant_type = st.text_input("Plant Type (optional)", placeholder="e.g., succulent, rose, fern")
        
        content = st.text_area(
            "Content",
            placeholder="Enter detailed plant care information...",
            height=200
        )
        
        tags = st.text_input(
            "Tags (comma-separated)",
            placeholder="e.g., indoor, beginner-friendly, low-maintenance"
        )
        
        submitted = st.form_submit_button("➕ Add to Knowledge Base", type="primary")
    
    if submitted and title and content:
        with st.spinner("Adding to knowledge base..."):
            try:
                # Prepare metadata
                metadata = {
                    "title": title,
                    "category": category,
                    "source": "user_input",
                    "timestamp": datetime.now().isoformat()
                }
                
                if plant_type:
                    metadata["plant_type"] = plant_type
                
                if tags:
                    metadata["tags"] = [tag.strip() for tag in tags.split(",")]
                
                # Add to vector database
                vector_db.add_documents([content], [metadata])
                
                st.success("Information added to knowledge base successfully!")
            except Exception as e:
                st.error(f"Failed to add information: {e}")

def show_web_search_update(agent, vector_db):
    """Show web search and knowledge base update"""
    st.subheader("🌐 Web Search & Update")
    
    plant_name = st.text_input(
        "Plant name",
        placeholder="e.g., Monstera deliciosa, Peace lily, Snake plant..."
    )
    
    search_topic = st.text_input(
        "Specific care topic (optional)",
        placeholder="e.g., watering, fertilizing, pest control..."
    )
    
    if st.button("🌐 Search & Update") and plant_name:
        with st.spinner("Searching web and updating knowledge base..."):
            input_data = {
                "plant_name": plant_name,
                "specific_query": search_topic if search_topic else "",
                "search_topics": ["care", "watering", "light", "soil"]
            }
            
            result = asyncio.run(agent.execute(input_data))
            
            if result.get("success"):
                update_data = result["data"]
                
                st.success("Knowledge base updated with new information!")
                
                # Show what was added
                if update_data.get("added_documents"):
                    st.subheader("📚 New Information Added")
                    
                    for doc in update_data["added_documents"]:
                        with st.expander(f"📄 {doc.get('title', 'New Document')}"):
                            st.markdown(doc.get('content', 'No content'))
                            
                            if doc.get('source'):
                                st.caption(f"Source: {doc['source']}")
                
                # Show summary
                if update_data.get("summary"):
                    st.subheader("📊 Update Summary")
                    st.info(update_data["summary"])
            else:
                st.error(f"Web search and update failed: {result.get('error', 'Unknown error')}")

def show_browse_categories(vector_db):
    """Show knowledge base categories"""
    st.subheader("📚 Browse Categories")
    
    categories = [
        "watering", "fertilizing", "light_requirements", "soil", 
        "diseases", "pests", "general_care", "propagation", "repotting"
    ]
    
    selected_category = st.selectbox("Select category", categories)
    
    if st.button("📚 Browse Category"):
        with st.spinner(f"Loading {selected_category} information..."):
            try:
                # Search for documents in the selected category
                results = vector_db.similarity_search(selected_category, k=10)
                
                if results:
                    st.success(f"Found {len(results)} documents in {selected_category}")
                    
                    for i, result in enumerate(results, 1):
                        with st.expander(f"📄 Document {i}"):
                            # Handle both Document objects and dictionaries
                            if hasattr(result, 'page_content'):
                                content = result.page_content
                                metadata = getattr(result, 'metadata', {})
                            elif isinstance(result, dict):
                                content = result.get('content', str(result))
                                metadata = result.get('metadata', {})
                            else:
                                content = str(result)
                                metadata = {}
                            
                            st.markdown(content)
                            
                            if metadata:
                                st.json(metadata)
                else:
                    st.warning(f"No documents found in {selected_category} category.")
            except Exception as e:
                st.error(f"Failed to browse category: {e}")

def show_dashboard_page(agents):
    """Display comprehensive dashboard"""
    st.header("📊 Plant Care Dashboard")
    st.markdown("Overview of all your plants and their care status.")
    
    if not st.session_state.plant_data:
        st.info("No plants added yet. Add your first plant using the sidebar!")
        return
    
    # Overall statistics
    st.subheader("📈 Overview")
    
    total_plants = len(st.session_state.plant_data)
    identified_plants = sum(1 for p in st.session_state.plant_data.values() if p.get('identification'))
    healthy_plants = sum(1 for p in st.session_state.plant_data.values() 
                        if p.get('health_status', {}).get('health_score', 0) > 0.7)
    plants_with_schedules = sum(1 for p in st.session_state.plant_data.values() if p.get('care_schedule'))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Plants", total_plants)
    with col2:
        st.metric("Identified", identified_plants, f"{(identified_plants/total_plants*100):.0f}%")
    with col3:
        st.metric("Healthy", healthy_plants, f"{(healthy_plants/total_plants*100):.0f}%")
    with col4:
        st.metric("With Schedules", plants_with_schedules, f"{(plants_with_schedules/total_plants*100):.0f}%")
    
    # Plant list with status
    st.subheader("🌱 Plant Status")
    
    plant_status_data = []
    for plant_id, plant_info in st.session_state.plant_data.items():
        status_row = {
            "Plant": plant_info.get('name', 'Unknown'),
            "Identified": "✅" if plant_info.get('identification') else "❌",
            "Health Score": f"{plant_info.get('health_status', {}).get('health_score', 0)*100:.0f}%" if plant_info.get('health_status') else "N/A",
            "Schedule": "✅" if plant_info.get('care_schedule') else "❌",
            "Last Updated": plant_info.get('created_at', 'Unknown')[:10] if plant_info.get('created_at') else 'Unknown'
        }
        plant_status_data.append(status_row)
    
    if plant_status_data:
        df_status = pd.DataFrame(plant_status_data)
        st.dataframe(df_status, use_container_width=True)
    
    # Recent activity
    st.subheader("📅 Recent Activity")
    
    # This would show recent measurements, care actions, etc.
    # For now, we'll show a placeholder
    st.info("Recent activity tracking will be implemented as you use the growth tracking features.")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Identify New Plant", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🩺 Health Check", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("📏 Record Growth", use_container_width=True):
            st.rerun()
    
    # Health alerts
    st.subheader("🚨 Health Alerts")
    
    alerts = []
    for plant_id, plant_info in st.session_state.plant_data.items():
        health_status = plant_info.get('health_status', {})
        health_score = health_status.get('health_score', 1.0)
        
        if health_score < 0.6:
            alerts.append({
                "plant": plant_info.get('name', 'Unknown'),
                "issue": f"Low health score: {health_score*100:.0f}%",
                "severity": "high" if health_score < 0.4 else "medium"
            })
        
        # Check for diseases
        diseases = health_status.get('diseases', [])
        if diseases:
            for disease in diseases:
                if disease.get('probability', 0) > 0.5:
                    alerts.append({
                        "plant": plant_info.get('name', 'Unknown'),
                        "issue": f"Possible disease: {disease.get('name', 'Unknown')}",
                        "severity": "high"
                    })
    
    if alerts:
        for alert in alerts:
            severity_color = {
                "high": "error",
                "medium": "warning",
                "low": "info"
            }.get(alert["severity"], "info")
            
            getattr(st, severity_color)(f"**{alert['plant']}:** {alert['issue']}")
    else:
        st.success("No health alerts! All plants appear to be doing well.")

def show_chat_page(vector_db):
    """Display AI Chat Assistant page with LangGraph workflow"""
    st.header("💬 AI Chat Assistant")
    st.markdown("Chat with your AI plant care expert! Upload images and ask questions for comprehensive advice.")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_workflow' not in st.session_state:
        st.session_state.chat_workflow = PlantCareWorkflow(vector_db)
    if 'current_workflow_state' not in st.session_state:
        st.session_state.current_workflow_state = None
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                with st.chat_message("user"):
                    if message.get('image'):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(message['image'], caption="Uploaded Image", width=150)
                        with col2:
                            st.markdown(message['content'])
                    else:
                        st.markdown(message['content'])
            
            elif message['type'] == 'assistant':
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
                    
                    # Show sources if available
                    if message.get('sources'):
                        with st.expander("📚 Sources Used"):
                            sources = message['sources']
                            
                            # Handle new workflow steps structure
                            if isinstance(sources, dict) and sources.get('workflow_steps'):
                                # Show workflow steps in order
                                st.markdown("**🔄 Analysis Workflow:**")
                                for step in sources['workflow_steps']:
                                    status_icon = "✅" if step['status'] == "completed" else "⏭️" if step['status'] == "skipped" else "❌"
                                    st.markdown(f"{status_icon} **{step['name']}** - {step['status'].title()}")
                                    
                                    # Show step details
                                    if step['details'] and step['status'] == "completed":
                                        with st.container():
                                            details = step['details']
                                            
                                            if 'plant_name' in details:  # Plant identification
                                                st.caption(f"   🏷️ Plant: {details['plant_name']} ({details['scientific_name']})")
                                                st.caption(f"   📊 Confidence: {details['confidence']}")
                                                st.caption(f"   🩺 Health: {details['health_status']}")
                                            
                                            elif 'sources_found' in details:  # Knowledge base
                                                st.caption(f"   📊 Found {details['sources_found']} relevant sources")
                                                if details.get('preview'):
                                                    for idx, preview in enumerate(details['preview'][:2], 1):
                                                        st.caption(f"   {idx}. {preview}")
                                            
                                            elif 'results_found' in details:  # Web search
                                                st.caption(f"   📊 Found {details['results_found']} web results")
                                                if details.get('top_sources'):
                                                    for idx, source in enumerate(details['top_sources'][:2], 1):
                                                        st.caption(f"   {idx}. {source}")
                                            
                                            elif 'temperature' in details:  # Weather
                                                st.caption(f"   🌡️ Temperature: {details['temperature']}")
                                                st.caption(f"   💧 Humidity: {details['humidity']}")
                                                st.caption(f"   ☁️ Conditions: {details['description']}")
                                            
                                            elif 'description_available' in details:  # Image analysis
                                                if details['description_available']:
                                                    st.caption(f"   👁️ Analysis: {details['preview']}")
                                                else:
                                                    st.caption(f"   👁️ {details['preview']}")
                                            
                                            elif 'total_sources_used' in details:  # Care generation
                                                st.caption(f"   🔗 Combined {details['total_sources_used']} data sources")
                                                st.caption(f"   📝 Response type: {details['response_type']}")
                                
                                # Show summary if available
                                if sources.get('summary'):
                                    st.markdown("---")
                                    st.markdown("**📋 Quick Summary:**")
                                    for source in sources['summary']:
                                        st.write(f"• {source}")
                            
                            # Handle legacy dictionary structure
                            elif isinstance(sources, dict):
                                # Show summary for backward compatibility
                                if sources.get('summary'):
                                    st.markdown("**Summary:**")
                                    for source in sources['summary']:
                                        st.write(f"• {source}")
                                
                                # Show detailed information
                                if sources.get('details'):
                                    details = sources['details']
                                    
                                    # Knowledge base sources
                                    if details.get('knowledge_base_count', 0) > 0:
                                        st.markdown(f"**📚 Knowledge Base:** {details['knowledge_base_count']} sources")
                                        if details.get('knowledge_base_snippets'):
                                            for snippet in details['knowledge_base_snippets'][:3]:  # Show first 3
                                                st.caption(f"• {snippet[:100]}...")
                                    
                                    # Web search sources
                                    if details.get('web_search_count', 0) > 0:
                                        st.markdown(f"**🌐 Web Search:** {details['web_search_count']} sources")
                                        if details.get('web_search_snippets'):
                                            for snippet in details['web_search_snippets'][:3]:  # Show first 3
                                                st.caption(f"• {snippet[:100]}...")
                                    
                                    # Weather data
                                    if details.get('weather_data'):
                                        st.markdown("**🌤️ Weather Data:** Included")
                                    
                                    # Image analysis
                                    if details.get('image_analysis'):
                                        st.markdown("**📸 Image Analysis:** Performed")
                                    
                                    # Plant identification
                                    if details.get('plant_identification'):
                                        plant_id = details['plant_identification']
                                        if plant_id.get('species_name'):
                                            st.markdown(f"**🌱 Plant Identified:** {plant_id['species_name']}")
                                    
                                    # Original response for tracing
                                    if details.get('original_response') and st.checkbox("Show original response for debugging", key=f"debug_{i}"):
                                        st.text_area("Original Response:", details['original_response'], height=100, key=f"orig_resp_{i}")
                            
                            # Handle old list structure for backward compatibility
                            elif isinstance(sources, list):
                                for source in sources:
                                    st.write(f"• {source}")
                    
                    # Show workflow progress if available
                    if message.get('workflow_steps'):
                        with st.expander("🔄 Analysis Steps"):
                            for step in message['workflow_steps']:
                                st.write(f"✅ {step}")
                    
                    # Feedback buttons (only for the last assistant message)
                    if i == len(st.session_state.chat_history) - 1 and not message.get('feedback_given'):
                        st.markdown("**How helpful was this response?**")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        feedback_scores = [20, 40, 60, 80, 100]
                        feedback_labels = ["😞 Poor", "😐 Fair", "🙂 Good", "😊 Great", "🤩 Excellent"]
                        
                        for j, (score, label) in enumerate(zip(feedback_scores, feedback_labels)):
                            with [col1, col2, col3, col4, col5][j]:
                                if st.button(label, key=f"feedback_{i}_{score}"):
                                    # Process feedback
                                    asyncio.run(process_feedback(i, score))
                                    st.rerun()
    
    # Input area
    st.markdown("---")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_area(
                "Ask about your plant:",
                placeholder="e.g., My plant has yellow leaves, what should I do? How often should I water my monstera?",
                height=100,
                key="chat_input"
            )
        
        with col2:
            uploaded_file = st.file_uploader(
                "Upload plant image (optional)",
                type=['png', 'jpg', 'jpeg'],
                key="chat_image"
            )
            
            location = st.text_input(
                "Your location (optional)",
                placeholder="e.g., New York, NY",
                key="chat_location"
            )
        
        submitted = st.form_submit_button("💬 Send Message", type="primary")
    
    # Process user input
    if submitted and user_input.strip():
        # Add user message to chat history
        user_message = {
            'type': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add image if uploaded
        if uploaded_file:
            image = Image.open(uploaded_file)
            user_message['image'] = image
        
        st.session_state.chat_history.append(user_message)
        
        # Process with LangGraph workflow
        with st.spinner("🤖 AI is analyzing your request..."):
            asyncio.run(process_chat_message(user_input, uploaded_file, location))
        
        st.rerun()

async def process_chat_message(user_input: str, uploaded_file, location: str):
    """Process chat message through LangGraph workflow"""
    try:
        # Prepare image data if uploaded
        image_base64 = None
        if uploaded_file:
            from utils.image_utils import prepare_image_for_api
            image_base64 = prepare_image_for_api(uploaded_file)
        
        # Create initial state
        initial_state = ChatState(
            user_query=user_input,
            image_base64=image_base64,
            image_description="",
            location=location or ""
        )
        
        # Run workflow
        workflow = st.session_state.chat_workflow
        
        # Track workflow progress
        workflow_steps = []
        
        # Execute workflow with progress tracking
        final_state = await workflow.run_workflow(initial_state)
        
        # Collect workflow steps
        if final_state.identified_plant:
            workflow_steps.append("Plant identified from image")
        if final_state.knowledge_results:
            workflow_steps.append("Searched knowledge base")
        if final_state.web_search_results:
            workflow_steps.append("Performed web search for additional information")
        if final_state.weather_data and "error" not in final_state.weather_data:
            workflow_steps.append("Retrieved weather information")
        workflow_steps.append("Generated comprehensive response")
        
        # Create assistant response
        assistant_message = {
            'type': 'assistant',
            'content': final_state.final_response,
            'timestamp': datetime.now().isoformat(),
            'sources': final_state.sources_used or [],
            'workflow_steps': workflow_steps,
            'confidence_score': final_state.confidence_score,
            'workflow_state': final_state  # Store for feedback processing
        }
        
        st.session_state.chat_history.append(assistant_message)
        st.session_state.current_workflow_state = final_state
        
    except Exception as e:
        error_message = {
            'type': 'assistant',
            'content': f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try again.",
            'timestamp': datetime.now().isoformat(),
            'error': True
        }
        st.session_state.chat_history.append(error_message)

async def process_feedback(message_index: int, score: int):
    """Process user feedback and update knowledge base if score is high enough"""
    try:
        # Get the message and its workflow state
        message = st.session_state.chat_history[message_index]
        workflow_state = message.get('workflow_state')
        
        if workflow_state:
            # Update workflow state with feedback
            updated_state = await st.session_state.chat_workflow.update_feedback(
                workflow_state, score, ""
            )
            
            # Mark feedback as given
            st.session_state.chat_history[message_index]['feedback_given'] = True
            st.session_state.chat_history[message_index]['feedback_score'] = score
            
            # Show feedback result
            if score >= 70 and updated_state.knowledge_updated:
                st.success("✅ Thank you for your feedback! This information has been added to our knowledge base to help other users.")
            else:
                st.info("✅ Thank you for your feedback! We'll use this to improve our responses.")
        
    except Exception as e:
        st.error(f"Error processing feedback: {str(e)}")

if __name__ == "__main__":
    main()