import base64
import io
from PIL import Image
from typing import Union, Tuple

def encode_image_to_base64(image: Union[Image.Image, str]) -> str:
    """Convert PIL Image or file path to base64 string"""
    if isinstance(image, str):
        # If it's a file path
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # If it's a PIL Image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image

def validate_image_format(image: Image.Image) -> bool:
    """Validate that image is in supported format"""
    supported_formats = ['JPEG', 'PNG', 'JPG']
    return image.format in supported_formats

def prepare_image_for_api(uploaded_file) -> str:
    """Prepare uploaded Streamlit file for API consumption"""
    try:
        # Open the image
        image = Image.open(uploaded_file)
        
        # Validate format
        if not validate_image_format(image):
            raise ValueError(f"Unsupported image format: {image.format}")
        
        # Resize if too large
        image = resize_image(image)
        
        # Convert to base64
        return encode_image_to_base64(image)
    
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")