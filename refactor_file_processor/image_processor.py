import base64
import io
from PIL import Image
from typing import Tuple


class ImageProcessor:
    """Handles common image processing operations for OCR."""
    
    @staticmethod
    def image_to_base64_string(
        image: Image.Image,
        max_size: Tuple[int, int] = (2048, 2048)
    ) -> Tuple[str, str]:
        """
        Convert PIL Image to Base64 string with optimal format.
        """
        # Resize if needed
        if image.width > max_size[0] or image.height > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        
        # Determine optimal format
        if image.mode in ('RGBA', 'LA', 'P'):
            image.save(buffered, format="PNG", optimize=True)
            media_type = "image/png"
        else:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            media_type = "image/jpeg"
        
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return base64_string, media_type
    
    @staticmethod
    def image_to_jpg_bytes(image: Image.Image) -> bytes:
        """
        Convert PIL Image to JPEG bytes (for Gemini API).
        
        Args:
            image: PIL Image object
            
        Returns:
            JPEG bytes
        """
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        with io.BytesIO() as output:
            image.save(output, format="JPEG", quality=90)
            return output.getvalue()
    
    @staticmethod
    def validate_image_id(img_id: str) -> None:
        """
        Validate image ID format.
        
        Args:
            img_id: Image identifier string
            
        Raises:
            ValueError: If image ID is invalid
        """
        if not img_id or not img_id.strip():
            raise ValueError("Image ID cannot be empty")
        if ']' in img_id or '\n' in img_id:
            raise ValueError(
                f"Image ID '{img_id}' contains invalid characters (']' or newline)"
            )