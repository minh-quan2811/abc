import re
import logging
from typing import Dict, Any, List, Protocol
from PIL import Image
from langchain_core.messages import HumanMessage

from .image_processor import ImageProcessor
from .image_quality_filter import ImageQualityFilter

logger = logging.getLogger(__name__)


class LangChainModel(Protocol):
    """Protocol for LangChain models with invoke method."""
    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        ...


class ImageOcrParser:
    """
    Performs batch OCR using LangChain-wrapped models.
    Optimized for processing multiple images (up to 5) per batch.
    """
    
    def __init__(
        self,
        model_llm: Any,
        model_cnn: Any = None,
        max_image_size: tuple = (2048, 2048),
        min_score: float = 0.5,
        enable_dedup: bool = True,
        east_model_path: str = fr'C:\Users\PC\Downloads\data-ai-marketing\east_detector.pb',
        ocr_prompt_path: str = "app/utils/file_processer/ocr_prompt.md"
    ):
        """
        Initialize OCR parser.
        """
        if not hasattr(model_llm, 'invoke'):
            raise TypeError(
                "model_llm must be a LangChain model with 'invoke' method"
            )
        
        self.model_llm = model_llm
        self.max_image_size = max_image_size
        
        # Initialize shared utilities
        self.processor = ImageProcessor()
        self.filter = ImageQualityFilter(
            model_cnn=model_cnn,
            min_score=min_score,
            enable_dedup=enable_dedup,
            east_model_path=east_model_path
        )
        
        # Load OCR prompt
        self.ocr_prompt = self._load_prompt(ocr_prompt_path)
    
    @staticmethod
    def _load_prompt(prompt_path: str) -> str:
        """Load OCR prompt from file."""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load prompt from {prompt_path}: {e}")
            # Fallback prompt
            return """You are an expert OCR system. Extract content from images.

<output_format>
<image_id>
[Extracted content here]
</image_id>
</output_format>"""
    
    def parse(
        self,
        images_dict: Dict[str, Image.Image],
        debug: bool = False,
        filter_images: bool = True
    ) -> Dict[str, str]:
        """
        Parse images and return OCR content.
        """
        if not images_dict:
            return {}
        
        if len(images_dict) > 5:
            raise ValueError("Maximum 5 images per batch supported")
        
        # Validate image IDs
        for img_id in images_dict:
            self.processor.validate_image_id(img_id)
        
        # Filter images by quality
        if filter_images:
            images_dict = self._filter_images(images_dict, debug)
            if not images_dict:
                logger.info("All images filtered out")
                return {
                    img_id: "[SKIPPED: No meaningful content detected]"
                    for img_id in images_dict
                }
        
        # Build message content
        message_content = self._build_message_content(images_dict)
        messages = HumanMessage(content=message_content)
        
        # Call LLM
        try:
            if debug:
                logger.info(f"Sending {len(images_dict)} images to OCR API...")
            
            response = self.model_llm.invoke([messages])
            response_text = response.content
            
            if debug:
                logger.info(f"Model Response:\n{response_text}")
        
        except Exception as e:
            error_msg = f"[OCR FAILED: API Error - {str(e)}]"
            logger.error(f"OCR API call failed: {e}", exc_info=True)
            return {img_id: error_msg for img_id in images_dict}
        
        # Parse response
        ocr_results = self._parse_response(response_text, images_dict, debug)
        return ocr_results
    
    def _filter_images(
        self,
        images_dict: Dict[str, Image.Image],
        debug: bool
    ) -> Dict[str, Image.Image]:
        """Filter images by quality."""
        filtered = {}
        for img_id, img in images_dict.items():
            is_valid, score, scores = self.filter.is_meaningful_image(img)
            if is_valid:
                filtered[img_id] = img
                if debug:
                    logger.info(f"Image {img_id}: PASS (score={score:.2f})")
            else:
                if debug:
                    logger.info(f"Image {img_id}: SKIP (score={score:.2f})")
        return filtered
    
    def _build_message_content(
        self,
        images_dict: Dict[str, Image.Image]
    ) -> List[Dict[str, Any]]:
        """Build message content for LLM."""
        message_content = [{"type": "text", "text": self.ocr_prompt}]
        
        for img_id, img in images_dict.items():
            base64_string, media_type = self.processor.image_to_base64_string(
                img, self.max_image_size
            )
            
            message_content.extend([
                {
                    "type": "text",
                    "text": f'<file image_id="{img_id}">\n'
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_string
                    }
                },
                {
                    "type": "text",
                    "text": f'\n</file image_id="{img_id}">\n\n'
                }
            ])
        
        return message_content
    
    def _parse_response(
        self,
        response_text: str,
        images_dict: Dict[str, Image.Image],
        debug: bool
    ) -> Dict[str, str]:
        """Parse OCR response text."""
        ocr_results = {}
        pattern = r"<([0-9a-fA-F-]{36})>\s*(.*?)\s*</\1>"
        matches = re.findall(pattern, response_text, flags=re.DOTALL)
        ocr_results = {image_id: content.strip() for image_id, content in matches}
        
        if debug:
            logger.info(f"OCR Results: {list(ocr_results.keys())}")
            logger.info(f"Expected IDs: {list(images_dict.keys())}")
        
        # Add failure messages for missing IDs
        for img_id in images_dict:
            if img_id not in ocr_results:
                logger.warning(f"No content returned for image ID: {img_id}")
                ocr_results[img_id] = (
                    "[OCR FAILED: Model did not return content for this ID]"
                )
        
        return ocr_results