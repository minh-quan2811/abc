import base64
import re
import io
from PIL import Image, ImageFilter, ImageStat
from typing import Dict, Any, List, Protocol, Tuple
import logging
import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LangChainModel(Protocol):
    """Protocol for LangChain models with invoke method."""
    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        ...






class ImageOcrParser:
    """
    Performs batch OCR on a dictionary of images using a LangChain-wrapped model.
    Optimized for processing multiple images (up to 5) in a single API call.
    Includes intelligent image filtering to skip non-informative images.
    """
    MIN_WIDTH = 32
    MIN_HEIGHT = 32

    def __init__(
        self,
        model_llm: Any,
        model_cnn: Any = None,
        max_image_size: tuple = (2048, 2048),
        min_score: float = 0.5,
        enable_dedup: bool = True,
        east_model_path: str = fr'C:\Users\PC\Downloads\data-ai-marketing\east_detector.pb'
    ):
        """
        Initialize parser with a configured LangChain model.

        Args:
            model_llm: A LangChain model object with an `invoke` method.
            model_cnn: Optional pre-loaded CNN model for text detection.
            max_image_size: Maximum dimensions (width, height) to resize images.
            min_score: Minimum quality score for image to be processed (0-1).
            enable_dedup: Enable duplicate detection using imagehash.
            east_model_path: Path to EAST text detection model.
        """
        if not hasattr(model_llm, 'invoke'):
            raise TypeError("The 'model_llm' object must be a LangChain model with an 'invoke' method.")
        
        self.model_llm = model_llm
        self.max_image_size = max_image_size
        self.min_score = min_score
        self.enable_dedup = enable_dedup
        self.seen_hashes = set()
        self.imagehash_available = False
        
        # Load text detection model
        if model_cnn:
            self.model_cnn = model_cnn
        else:
            try:
                self.model_cnn = cv2.dnn.readNet(east_model_path)
            except Exception as e:
                logger.warning(f"Could not load EAST model: {e}. Text detection will be disabled.")
                self.model_cnn = None
        
        # Try to import imagehash for deduplication
        if self.enable_dedup:
            try:
                import imagehash
                self.imagehash_available = True
            except ImportError:
                logger.warning("imagehash not installed. Deduplication disabled.")
                self.enable_dedup = False

    def is_meaningful_image(self, image: Image.Image) -> Tuple[bool, float, Dict[str, float]]:
        # Skip very small images
        if image.width < self.MIN_WIDTH or image.height < self.MIN_HEIGHT:
            return False, 0.0, {}

        if self._should_skip_by_dimensions(image.width, image.height):
            return False, 0.0, {}

        is_dup, _ = self._is_duplicate(image)
        if is_dup:
            return False, 0.0, {}

        # Calculate quality scores
        scores = {
            'brightness': self._check_brightness(image),
            'color': self._check_color_diversity(image),
            'entropy': self._check_histogram_entropy(image),
            'edges': self._check_edge_density(image),
            'variance': self._check_variance(image),
            'uniqueness': self._check_unique_colors(image),
            'texture': self._check_texture(image),
            'density': self._check_content_density(image)
        }

        weights = {
            'brightness': 0.15,
            'color': 0.10,
            'entropy': 0.15,
            'edges': 0.20,
            'variance': 0.05,
            'uniqueness': 0.05,
            'texture': 0.10,
            'density': 0.20
        }

        total_score = sum(scores[k] * weights[k] for k in scores)

        if total_score >= self.min_score:
            if self._has_text_in_image(image):
                return True, total_score, scores

        return False, total_score, scores
    

    def _should_skip_by_dimensions(self, width: int, height: int) -> bool:
        """Skip images that are too small or have extreme aspect ratios."""
        MIN_SIZE = 150
        MIN_ASPECT_RATIO = 0.35
        
        if width < MIN_SIZE and height < MIN_SIZE:
            return True
        if min(width, height) / max(width, height) < MIN_ASPECT_RATIO:
            return True
        if (width > 500 and height < 80) or (height > 500 and width < 80):
            return True
        return False

    def _is_duplicate(self, image: Image.Image) -> Tuple[bool, str]:
        """Check if image is a duplicate using perceptual hashing."""
        if not self.enable_dedup or not self.imagehash_available:
            return False, ""
        
        try:
            import imagehash
            img_hash = str(imagehash.dhash(image, hash_size=8))
            if img_hash in self.seen_hashes:
                return True, img_hash
            self.seen_hashes.add(img_hash)
            return False, img_hash
        except Exception:
            return False, ""

    def _check_brightness(self, image: Image.Image) -> float:
        """Check if image brightness is in acceptable range."""
        stat = ImageStat.Stat(image.convert("L"))
        mean = stat.mean[0]
        
        if 30 < mean < 230:
            return 1.0
        if mean >= 245 or mean <= 15:
            return 0.0
        
        return (245 - mean) / 15 if mean >= 230 else (mean - 15) / 15

    def _check_color_diversity(self, image: Image.Image) -> float:
        """Measure color diversity in the image."""
        colors = image.convert('RGB').getcolors(10000)
        if colors is None:
            return 1.0
        return min(len(colors) / 500, 1.0)

    def _check_histogram_entropy(self, image: Image.Image) -> float:
        """Calculate histogram entropy to measure information content."""
        hist = image.convert("L").histogram()
        size = sum(hist)
        
        if size == 0:
            return 0.0
        
        p = [h / size for h in hist if h > 0]
        entropy = -sum(pi * np.log2(pi) for pi in p)
        return min(entropy / 8.0, 1.0)

    def _check_edge_density(self, image: Image.Image) -> float:
        """Measure edge density to detect structured content."""
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        ratio = np.array(edges).sum() / 255 / (image.width * image.height)
        
        if ratio < 0.03:
            return ratio / 0.03
        return 1.0 if ratio > 0.10 else (0.5 + (ratio - 0.03) / 0.14)

    def _check_variance(self, image: Image.Image) -> float:
        """Check pixel value variance."""
        return min(ImageStat.Stat(image.convert("L")).stddev[0] / 40, 1.0)

    def _check_unique_colors(self, image: Image.Image) -> float:
        """Count unique colors in downsampled image."""
        colors = image.resize((100, 100), Image.LANCZOS).getcolors(10000)
        if colors is None:
            return 1.0
        return min(len(colors) / 500, 1.0)

    def _check_texture(self, image: Image.Image) -> float:
        """Measure texture complexity using Laplacian."""
        laplacian = image.convert("L").filter(
            ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 0)
        )
        return min(ImageStat.Stat(laplacian).var[0] / 300, 1.0)

    def _check_content_density(self, image: Image.Image) -> float:
        """Check content distribution across image grid."""
        w, h = image.width, image.height
        grid = 4
        cw, ch = w // grid, h // grid
        
        if cw < 10 or ch < 10:
            return 0.5
        
        gray = image.convert("L")
        cell_vars = []
        
        for i in range(grid):
            for j in range(grid):
                cell = gray.crop((j * cw, i * ch, (j + 1) * cw, (i + 1) * ch))
                cell_vars.append(ImageStat.Stat(cell).var[0])
        
        if len(cell_vars) < 2:
            return 0.5
        
        return min(float(np.var(cell_vars)) / 500, 1.0)
    
    def _has_text_in_image(self, image: Image.Image,
                           input_size: tuple = (224, 224),
                           conf_threshold: float = 0.8) -> bool:
        if self.model_cnn is None:
            return True
        try:
            img_cv = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB -> BGR
            # Resize keeping aspect ratio
            h, w = img_cv.shape[:2]
            scale = min(input_size[0]/w, input_size[1]/h)
            new_w, new_h = max(1,int(w*scale)), max(1,int(h*scale))
            resized = cv2.resize(img_cv, (new_w, new_h))
            resized = np.ascontiguousarray(resized, dtype=np.float32)

            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0,
                size=input_size,
                mean=(123.68, 116.78, 103.94),
                swapRB=True,
                crop=False
            )
            self.model_cnn.setInput(blob)
            scores = self.model_cnn.forward("feature_fusion/Conv_7/Sigmoid")[0,0]
            return np.any(scores >= conf_threshold)
        except cv2.error as e:
            logger.warning(f"OpenCV DNN error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return False
    @staticmethod
    def _image_to_base64_string(image: Image.Image, max_size: tuple = (2048, 2048)) -> tuple:
        """
        Convert a PIL Image to a Base64 string and return media type.
        
        Returns:
            Tuple of (base64_string, media_type)
        """
        if image.width > max_size[0] or image.height > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        
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
    def _validate_image_id(img_id: str) -> None:
        """Validate that image ID doesn't contain problematic characters."""
        if not img_id or not img_id.strip():
            raise ValueError("Image ID cannot be empty")
        if ']' in img_id or '\n' in img_id:
            raise ValueError(f"Image ID '{img_id}' contains invalid characters (']' or newline)")

    def parse(
        self,
        images_dict: Dict[str, Image.Image],
        debug: bool = False,
        filter_images: bool = True
    ) -> Dict[str, str]:
        """
        Parse a dictionary of images and return OCR content.

        Args:
            images_dict: Dictionary with image IDs (str) as keys and PIL Images as values.
            debug: If True, print debug information.
            filter_images: If True, apply quality filtering before OCR.

        Returns:
            Dictionary with same keys as input and extracted text as values.
        """
        if not images_dict:
            return {}

        if len(images_dict) > 5:
            raise ValueError("Maximum 5 images supported per batch for optimal performance.")

        # Validate all image IDs
        for img_id in images_dict:
            self._validate_image_id(img_id)

        # Filter images if enabled
        if filter_images:
            filtered_images = {}
            for img_id, img in images_dict.items():
                is_valid, score, scores = self.is_meaningful_image(img)
                if is_valid:
                    filtered_images[img_id] = img
                    if debug:
                        logger.info(f"Image {img_id}: PASS (score={score:.2f})")
                else:
                    if debug:
                        logger.info(f"Image {img_id}: SKIP (score={score:.2f})")
            
            if not filtered_images:
                logger.info("All images filtered out - no meaningful content detected")
                return {img_id: "[SKIPPED: No meaningful content detected]" for img_id in images_dict}
            
            images_dict = filtered_images

        # Build message content
        message_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": """You are an expert OCR system. Extract content from images containing graphs, tables, diagrams, text, and photos.

<input_format>
Images provided as: [ID: <image_id>]
</input_format>

<extraction_rules>
1. SKIP: Product photos or people photos without informative text
2. TABLES: Extract complete table in Markdown format - all headers, rows, columns
3. GRAPHS/CHARTS: Describe key insights and trends with specific data points
4. DIAGRAMS/FLOWCHARTS: Describe structure, extract all labels, explain relationships
5. TEXT: Extract all visible text, preserve formatting and structure
</extraction_rules>

<output_format>
<image_id>
[Extracted content here]
</image_id>
</output_format>

<examples>
<101>
| Product | Q1 Sales | Q2 Sales |
|---------|----------|----------|
| Widget A| $50K     | $75K     |
</101>

<102>
Revenue grew from $100K (Q1) to $250K (Q4), showing 150% increase. Peak growth occurred in Q2-Q3.
</102>

<103>
**Marketing Mix**
- Product: Features and benefits
- Price: Competitive positioning
- Place: Distribution channels
- Promotion: Communication strategy
</103>
</examples>""",
            }
        ]

        for img_id, img in images_dict.items():
            base64_string, media_type = self._image_to_base64_string(img, self.max_image_size)
            
            message_content.extend([
                {
                        "type": "text",
                        "text": f'<file image_id="{img_id}"">\n'
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"{media_type}",
                        "data": base64_string
                    }
                },
                {
                        "type": "text",
                        "text": f'\n</file image_id="{img_id}">\n\n'
                }
            ])

        # messages = [{"role": "user", "content": message_content}]
        messages = HumanMessage(content=message_content)

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
        ocr_results: Dict[str, str] = {}
        pattern = r"<([0-9a-fA-F-]{36})>\s*(.*?)\s*</\1>"
        matches = re.findall(pattern, response_text, flags=re.DOTALL)
        ocr_results = {image_id: content.strip() for image_id, content in matches}

        if debug:
            logger.info(f"OCR Results: {list(ocr_results.keys())}")
            logger.info(f"Expected IDs: {list(images_dict.keys())}")

        # Add failure messages for missing IDs
        for img_id in images_dict:
            if img_id not in ocr_results:
                logger.warning(f"Model did not return content for image ID: {img_id}")
                ocr_results[img_id] = "[OCR FAILED: Model did not return content for this ID]"
        
        return ocr_results