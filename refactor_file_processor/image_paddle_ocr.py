# import logging
# from typing import Dict, Any
# from PIL import Image
# import numpy as np
# from paddleocr import PaddleOCR

# from .image_processor import ImageProcessor
# from .image_quality_filter import ImageQualityFilter

# logger = logging.getLogger(__name__)


# class ImageOcrParserPaddle:
#     """
#     Performs batch OCR using PaddleOCR.
#     Returns results in the same format as LLM-based parser for compatibility.
#     """
    
#     def __init__(
#         self,
#         model_cnn: Any = None,
#         max_image_size: tuple = (2048, 2048),
#         min_score: float = 0.5,
#         enable_dedup: bool = True,
#         east_model_path: str = None,
#         lang: str = 'en',
#         use_angle_cls: bool = True
#     ):
#         """
#         Initialize PaddleOCR parser.
#         """
#         self.max_image_size = max_image_size
        
#         # Initialize PaddleOCR
#         self.paddle_ocr = PaddleOCR(
#             lang=lang,
#             use_angle_cls=use_angle_cls
#         )
        
#         # Initialize shared utilities
#         self.processor = ImageProcessor()
#         self.filter = ImageQualityFilter(
#             model_cnn=model_cnn,
#             min_score=min_score,
#             enable_dedup=enable_dedup,
#             east_model_path=east_model_path
#         )
    
#     def parse(
#         self,
#         images_dict: Dict[str, Image.Image],
#         debug: bool = False,
#         filter_images: bool = True
#     ) -> Dict[str, str]:
#         """
#         Parse images and return OCR content using PaddleOCR.
#         """
#         if not images_dict:
#             return {}
        
#         # Validate image IDs
#         for img_id in images_dict:
#             self.processor.validate_image_id(img_id)
        
#         # Track original IDs for filtered images
#         original_ids = set(images_dict.keys())
#         ocr_results = {}
        
#         # Filter images by quality
#         if filter_images:
#             filtered_dict = self._filter_images(images_dict, debug)
            
#             # Add skip messages for filtered-out images
#             for img_id in original_ids:
#                 if img_id not in filtered_dict:
#                     ocr_results[img_id] = "[Image skipped: Low quality/no text detected]"
#                     if debug:
#                         logger.info(f"Image {img_id}: FILTERED OUT")
            
#             images_dict = filtered_dict
            
#             if not images_dict:
#                 logger.info("All images filtered out")
#                 return ocr_results
        
#         # Process each image with PaddleOCR
#         for img_id, img in images_dict.items():
#             try:
#                 if debug:
#                     logger.info(f"Processing image {img_id} with PaddleOCR...")
                
#                 # Convert PIL Image to numpy array
#                 img_array = np.array(img)
                
#                 # Run PaddleOCR (cls is set during initialization)
#                 result = self.paddle_ocr.ocr(img_array)
                
#                 # Extract text from results
#                 extracted_text = self._extract_text_from_result(result, debug)
                
#                 ocr_results[img_id] = extracted_text
                
#                 if debug:
#                     logger.info(f"Image {img_id}: Extracted {len(extracted_text)} characters")
            
#             except Exception as e:
#                 error_msg = f"[OCR FAILED: {str(e)}]"
#                 logger.error(f"PaddleOCR failed for image {img_id}: {e}", exc_info=True)
#                 ocr_results[img_id] = error_msg
        
#         return ocr_results
    
#     def _filter_images(
#         self,
#         images_dict: Dict[str, Image.Image],
#         debug: bool
#     ) -> Dict[str, Image.Image]:
#         """Filter images by quality."""
#         filtered = {}
#         for img_id, img in images_dict.items():
#             is_valid, score, scores = self.filter.is_meaningful_image(img)
#             if is_valid:
#                 filtered[img_id] = img
#                 if debug:
#                     logger.info(f"Image {img_id}: PASS (score={score:.2f})")
#             else:
#                 if debug:
#                     logger.info(f"Image {img_id}: SKIP (score={score:.2f})")
#         return filtered
    
#     def _extract_text_from_result(
#         self,
#         result: list,
#         debug: bool = False
#     ) -> str:
#         """
#         Extract text from PaddleOCR result.
        
#         PaddleOCR returns format:
#         [
#             [
#                 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],  # bounding box
#                 ('text content', confidence_score)
#             ],
#             ...
#         ]
#         """
#         if not result or not result[0]:
#             return ""
        
#         text_lines = []
        
#         for line in result[0]:
#             if line and len(line) >= 2:
#                 # line[0] is bounding box, line[1] is (text, confidence)
#                 text_content = line[1][0] if isinstance(line[1], tuple) else str(line[1])
#                 confidence = line[1][1] if isinstance(line[1], tuple) and len(line[1]) > 1 else 1.0
                
#                 if debug:
#                     logger.debug(f"  Line: '{text_content}' (conf: {confidence:.2f})")
                
#                 text_lines.append(text_content)
        
#         # Join lines with newlines
#         extracted_text = '\n'.join(text_lines)
        
#         return extracted_text.strip()


# # Standalone test function
# def test_paddle_ocr():
#     """Test PaddleOCR parser with sample images."""
#     from PIL import Image
    
#     # Initialize parser
#     parser = ImageOcrParserPaddle(
#         use_angle_cls=True,
#         lang='en'
#     )
    
#     # Test with images
#     test_images = {
#         "b3a88230-5dc3-47ff-897d-85acf7b009cf": Image.open(
#             r"C:\Users\PC\Downloads\138.jpg"
#         ),
#         "68e3d7e6-9c64-8328-b61d-19926ef83dbf": Image.open(
#             r"C:\Users\PC\Downloads\b.jpg"
#         ),
#     }
    
#     # Parse images
#     results = parser.parse(test_images, debug=True)
    
#     # Print results
#     for img_id, text in results.items():
#         print(f"\n{'='*50}")
#         print(f"Image ID: {img_id}")
#         print(f"{'='*50}")
#         print(text)


# if __name__ == "__main__":
#     test_paddle_ocr()


from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)
result = ocr.ocr(input=fr"C:\Users\PC\Downloads\abc.png")
print(result)