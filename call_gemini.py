import io
import time
import json
from typing import Dict, Any, Optional
from PIL import Image
from google import genai
from google.genai import types
from app.core.config import settings

# Initialize Gemini client
client = genai.Client(api_key="xxx")
with open("app/utils/file_processer/ocr_prompt.md", "r", encoding="utf-8") as f:
    PROMPT_OCR = f.read()

def image_to_jpg_bytes(image: Image.Image) -> bytes:
    """
    Convert a PIL Image to JPEG bytes.
    """
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")  # JPEG không hỗ trợ alpha
    with io.BytesIO() as output:
        image.save(output, format="JPEG", quality=90)
        return output.getvalue()


def generate_response_from_images(
    images: Dict[str, Image.Image],
    user_prompt: Optional[str] = PROMPT_OCR,
    config_gemini: types.GenerateContentConfig = types.GenerateContentConfig(
        temperature=0.25,
        top_p=0.5,
        responseMimeType="text/plain",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        seed=42,
    ),
) -> Dict[str, Any]:
    """
    Gọi Gemini API để sinh phản hồi, kèm nhiều ảnh (dict {image_id: Image.Image}),
    đánh số thứ tự từ 1 và chuyển toàn bộ sang JPEG.
    """
    try:
        start_time = time.time()
        contents: list[types.Content] = []

        # Thêm từng ảnh (đã chuyển JPEG) và đánh số thứ tự
        for idx, (image_id, image) in enumerate(images.items(), start=1):
            image_bytes = image_to_jpg_bytes(image)
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

        # Thêm prompt chính (nếu có)
        if user_prompt:
            contents.append(types.Content(role="user", parts=[types.Part(text=user_prompt)]))

        # Gọi Gemini API
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
    # Ví dụ sử dụng
    from PIL import Image

    # Load ảnh mẫu
    images = {
        "b3a88230-5dc3-47ff-897d-85acf7b009cf": Image.open(fr"c:\Users\four\Downloads\36869aa8-cd77-4e7d-9c77-c5cfa6c22208.jpg"),
        "68e3d7e6-9c64-8328-b61d-19926ef83dbf": Image.open(fr"c:\Users\four\Downloads\102210334_LeVanTu_HoithaoPHATTRIENPHANMEMTRONGKYNGUYENAI.jpg"),
    }

    response = generate_response_from_images( images=images )