import time
from typing import Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types

from .image_processor import ImageProcessor


def generate_response_from_images(
    images: Dict[str, Image.Image],
    user_prompt: Optional[str] = None,
    api_key: str = "AIzaSyAlJm0ZrIOTwmHOpxAxLV9mLV6knf3T82M",
    config_gemini: types.GenerateContentConfig = None
) -> Dict[str, Any]:
    """
    Call Gemini API to generate response from multiple images.
    """
    # Load default prompt if not provided
    if user_prompt is None:
        try:
            prompt_path="app/utils/file_processer/ocr_prompt.md"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                user_prompt = f.read()
        except Exception:
            user_prompt = "Describe the image"
    
    # Default config
    if config_gemini is None:
        config_gemini = types.GenerateContentConfig(
            temperature=0.25,
            top_p=0.5,
            response_mime_type="text/plain",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            seed=42,
        )
    
    try:
        start_time = time.time()
        
        # Initialize Gemini client
        client = genai.Client(api_key=api_key)
        processor = ImageProcessor()
        
        # Build contents list
        contents = []
        
        # Add images with numbering
        for idx, (image_id, image) in enumerate(images.items(), start=1):
            image_bytes = processor.image_to_jpg_bytes(image)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(text=f"Image {idx}. Image ID: {image_id}"),
                        types.Part.from_bytes(
                            mime_type="image/jpeg",
                            data=image_bytes,
                        ),
                    ],
                )
            )
        
        # Add main prompt
        if user_prompt:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=user_prompt)]
                )
            )
        
        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config_gemini,
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "data": response.text,
            "gemini_response": {
                "elapsed_time": elapsed_time,
                "token_input": response.usage_metadata.prompt_token_count,
                "token_output": response.usage_metadata.candidates_token_count,
            },
        }
    
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")


if __name__ == "__main__":
    from PIL import Image
    
    # Example usage
    images = {
        "b3a88230-5dc3-47ff-897d-85acf7b009cf": Image.open(
            r"C:\Users\PC\Downloads\138.jpg"
        ),
        "68e3d7e6-9c64-8328-b61d-19926ef83dbf": Image.open(
            r"C:\Users\PC\Downloads\b.jpg"
        ),
    }
    
    response = generate_response_from_images(images=images)
    print(response)