import logging
from typing import Dict, Tuple
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ImageQualityFilter:
    """
    Filters images based on quality metrics and text detection.
    """
    
    MIN_WIDTH = 32
    MIN_HEIGHT = 32
    
    def __init__(
        self,
        model_cnn=None,
        min_score: float = 0.5,
        enable_dedup: bool = True,
        east_model_path: str = fr'C:\Users\PC\Downloads\data-ai-marketing\east_detector.pb'
    ):
        """
        Initialize image quality filter.
        """
        self.min_score = min_score
        self.enable_dedup = enable_dedup
        self.seen_hashes = set()
        self.imagehash_available = False
        
        # Load text detection model
        if model_cnn:
            self.model_cnn = model_cnn
        elif east_model_path:
            try:
                self.model_cnn = cv2.dnn.readNet(east_model_path)
            except Exception as e:
                logger.warning(f"Could not load EAST model: {e}")
                self.model_cnn = None
        else:
            self.model_cnn = None
        
        # Try to import imagehash
        if self.enable_dedup:
            try:
                import imagehash
                self.imagehash_available = True
            except ImportError:
                logger.warning("imagehash not installed. Deduplication disabled.")
                self.enable_dedup = False
    
    def is_meaningful_image(
        self,
        image: Image.Image
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Determine if image contains meaningful content.
        """
        # Basic dimension checks
        if image.width < self.MIN_WIDTH or image.height < self.MIN_HEIGHT:
            return False, 0.0, {}
        
        if self._should_skip_by_dimensions(image.width, image.height):
            return False, 0.0, {}
        
        # Duplicate check
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
        
        # Final validation with text detection
        if total_score >= self.min_score:
            if self._has_text_in_image(image):
                return True, total_score, scores
        
        return False, total_score, scores
    
    def _should_skip_by_dimensions(self, width: int, height: int) -> bool:
        """Check if dimensions indicate non-informative image."""
        MIN_SIZE = 100
        MIN_ASPECT_RATIO = 0.35
        
        if width < MIN_SIZE and height < MIN_SIZE:
            return False
        if min(width, height) / max(width, height) < MIN_ASPECT_RATIO:
            return False
        if (width > 500 and height < 80) or (height > 500 and width < 80):
            return False
        return False
    
    def _is_duplicate(self, image: Image.Image) -> Tuple[bool, str]:
        """Detect duplicate images using perceptual hashing."""
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
        """Check brightness level."""
        stat = ImageStat.Stat(image.convert("L"))
        mean = stat.mean[0]
        
        if 30 < mean < 230:
            return 1.0
        if mean >= 245 or mean <= 15:
            return 0.0
        
        return (245 - mean) / 15 if mean >= 230 else (mean - 15) / 15
    
    def _check_color_diversity(self, image: Image.Image) -> float:
        """Measure color diversity."""
        colors = image.convert('RGB').getcolors(10000)
        if colors is None:
            return 1.0
        return min(len(colors) / 500, 1.0)
    
    def _check_histogram_entropy(self, image: Image.Image) -> float:
        """Calculate histogram entropy."""
        hist = image.convert("L").histogram()
        size = sum(hist)
        
        if size == 0:
            return 0.0
        
        p = [h / size for h in hist if h > 0]
        entropy = -sum(pi * np.log2(pi) for pi in p)
        return min(entropy / 8.0, 1.0)
    
    def _check_edge_density(self, image: Image.Image) -> float:
        """Measure edge density."""
        edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
        ratio = np.array(edges).sum() / 255 / (image.width * image.height)
        
        if ratio < 0.03:
            return ratio / 0.03
        return 1.0 if ratio > 0.10 else (0.5 + (ratio - 0.03) / 0.14)
    
    def _check_variance(self, image: Image.Image) -> float:
        """Check pixel variance."""
        return min(ImageStat.Stat(image.convert("L")).stddev[0] / 40, 1.0)
    
    def _check_unique_colors(self, image: Image.Image) -> float:
        """Count unique colors."""
        colors = image.resize((100, 100), Image.LANCZOS).getcolors(10000)
        if colors is None:
            return 1.0
        return min(len(colors) / 500, 1.0)
    
    def _check_texture(self, image: Image.Image) -> float:
        """Measure texture complexity."""
        laplacian = image.convert("L").filter(
            ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], 1, 0)
        )
        return min(ImageStat.Stat(laplacian).var[0] / 300, 1.0)
    
    def _check_content_density(self, image: Image.Image) -> float:
        """Check content distribution across grid."""
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
    
    def _has_text_in_image(
        self,
        image: Image.Image,
        input_size: Tuple[int, int] = (224, 224),
        conf_threshold: float = 0.8
    ) -> bool:
        """Detect text presence using EAST model."""
        if self.model_cnn is None:
            return True
        
        try:
            img_cv = np.array(image.convert("RGB"))[:, :, ::-1]
            h, w = img_cv.shape[:2]
            scale = min(input_size[0]/w, input_size[1]/h)
            new_w, new_h = max(1, int(w*scale)), max(1, int(h*scale))
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
            scores = self.model_cnn.forward("feature_fusion/Conv_7/Sigmoid")[0, 0]
            return np.any(scores >= conf_threshold)
        except cv2.error as e:
            logger.warning(f"OpenCV DNN error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return False