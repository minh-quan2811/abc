# import base64
# import re
# import io
# from PIL import Image
# from typing import Dict, Any, List

# # Import thành phần cốt lõi từ LangChain
# from langchain_core.messages import HumanMessage

# class ImageOcrParser:
#     """
#     Thực hiện OCR hàng loạt trên một dictionary ảnh sử dụng một mô hình Gemini
#     được bao bọc bởi LangChain (ví dụ: ChatGoogleGenerativeAI).
    
#     Class này định dạng lại dữ liệu ảnh sang Base64 và xây dựng một
#     HumanMessage duy nhất để tương thích với API của LangChain.
#     """

#     def __init__(self, model: Any):
#         """
#         Khởi tạo parser với một mô hình LangChain đã được cấu hình.

#         Args:
#             model: Một đối tượng mô hình LangChain (ví dụ: ChatGoogleGenerativeAI)
#                    có phương thức `invoke`.
#         """
#         if not hasattr(model, 'invoke'):
#             raise TypeError("Đối tượng 'model' phải là một mô hình LangChain và có phương thức 'invoke'.")
#         self.model = model
        
#     @staticmethod
#     def _image_to_base64_url(image: Image.Image) -> str:
#         """
#         Chuyển đổi một đối tượng PIL Image thành chuỗi data URL Base64.
#         Thao tác này hoàn toàn diễn ra trong bộ nhớ.
#         """
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
#         encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
#         return f"data:image/png;base64,{encoded_string}"

#     def parse(self, images_dict: Dict[str, Image.Image]) -> Dict[str, str]:
#         """
#         Phân tích một dictionary ảnh và trả về nội dung OCR.

#         Args:
#             images_dict: Một dictionary với key là ID của ảnh (str) 
#                          và value là đối tượng PIL Image.

#         Returns:
#             Một dictionary với các key giống hệt đầu vào và value là
#             nội dung văn bản được OCR từ ảnh tương ứng.
#         """
#         if not images_dict:
#             return {}

#         message_content: List[Dict[str, Any]] = [
#             {
#                 "type": "text",
#                 "text": (
#                     "You are an OCR expert. Please extract text from the following images.\n\n"
#                     "IMPORTANT REQUIREMENTS:\n"
#                     "For each image, follow this strict format:\n"
#                     "1. Start with a line containing the image ID, in the format: `[ID: <image_id>]`\n"
#                     "2. If the image content is clear and readable, extract all visible text. Preserve original formatting, special characters, and line breaks as they appear in the image.\n"
#                     "3. If the image contains a table, extract the table in **Markdown format**.\n"
#                     "4. If the image is too blurry, obstructed, or mostly unreadable, **skip it and do not return any output for that image**.\n"
#                 ),
#             }
#         ]


#         for img_id, img in images_dict.items():
#             message_content.append({
#                 "type": "image_url",
#                 "image_url": {"url": self._image_to_base64_url(img)}
#             })
#             message_content.append({"type": "text", "text": f"[ID: {img_id}]"})

#         message = HumanMessage(content=message_content)

#         try:
#             response = self.model.invoke([message])
#             response_text = response.content
#         except Exception as e:
#             return {img_id: f"[OCR FAILED: API Error - {e}]" for img_id in images_dict}
        
#         print(response_text)

#         ocr_results: Dict[str, str] = {}
#         pattern = r"\[ID: (.*?)\].*?\n(.*?)(?=\[ID: |$)"
#         matches = re.findall(pattern, response_text, re.DOTALL)

#         for match in matches:
#             img_id = match[0].strip()
#             ocr_text = match[1].strip()
#             ocr_results[img_id] = ocr_text

#         for img_id in images_dict:
#             if img_id not in ocr_results:
#                 ocr_results[img_id] = "[OCR FAILED: Model did not return content for this ID]"
        
#         return ocr_results



import base64
import re
import io
from PIL import Image
from typing import Dict, Any, List, Protocol
import logging

logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage


class LangChainModel(Protocol):
    """Protocol for LangChain models with invoke method."""
    def invoke(self, messages: List[Dict[str, Any]]) -> Any:
        ...


class ImageOcrParser:
    """
    Performs batch OCR on a dictionary of images using a LangChain-wrapped model.
    Optimized for processing multiple images (up to 5) in a single API call.
    """

    def __init__(self, model: Any, max_image_size: tuple = (2048, 2048)):
        """
        Initialize parser with a configured LangChain model.

        Args:
            model: A LangChain model object with an `invoke` method.
            max_image_size: Maximum dimensions (width, height) to resize images.
                           Helps reduce token usage and improve performance.
        """
        if not hasattr(model, 'invoke'):
            raise TypeError("The 'model' object must be a LangChain model with an 'invoke' method.")
        self.model = model
        self.max_image_size = max_image_size
        
    @staticmethod
    def _image_to_base64_string(image: Image.Image, max_size: tuple = (2048, 2048)) -> tuple:
        """
        Convert a PIL Image to a Base64 string (without data URL prefix) and return media type.
        
        Returns:
            Tuple of (base64_string, media_type)
        """
        # Resize if needed
        if image.width > max_size[0] or image.height > max_size[1]:
            image = image.copy()
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        
        # Determine format and media type
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

    def parse(self, images_dict: Dict[str, Image.Image], debug: bool = False) -> Dict[str, str]:
        """
        Parse a dictionary of images and return OCR content.

        Args:
            images_dict: Dictionary with image IDs (str) as keys and PIL Images as values.
            debug: If True, print debug information including base64 strings and response.

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

        # message_content: List[Dict[str, Any]] = [
        #     {
        #         "type": "text",
        #         "text": ("""You are an OCR expert. You are reading images from a document that contains mixed content including graphs, tables, book covers, diagrams, text pages, and photos. Some content may be related to marketing knowledge. Some may not

        #     <instructions>
        #     For each image, follow this strict format:

        #     1. Start with a line containing the image ID in the format: [ID: <image_id>]

        #     2. Analyze the image type and extract accordingly:

        #     <image_types>
        #     <type name="text_document">
        #     Image contains clear and readable text (paragraphs, sentences, captions):
        #     - Extract ALL visible text
        #     - Preserve all the original formatting, special characters, and line breaks exactly as they appear
        #     </type>

        #     <type name="table">
        #     Image contains a table with rows and columns:
        #     - Extract the ENTIRE table in Markdown format include all rows and columns, not just the beginning or a summary
        #     - Do NOT describe what the table is about, just extract the data in Markdown format.
        #     <markdown_table_format>
        #     Use | for columns, --- for header separator
        #     Example:
        #     | Header 1 | Header 2 |
        #     |----------|----------|
        #     | Data 1   | Data 2   |
        #     </markdown_table_format>
        #     </type>

        #     <type name="graph_chart">
        #     Image is a graph, chart, or plot (bar chart, line graph, pie chart, etc.):
        #     - Extract key insights and trends from the visualization
        #     - Include specific data points from the graph to support your insights
        #     - Examples: "Revenue increased from $100K to $250K between Q1 and Q4"
        #     - Do NOT just read axis labels or legends as plain text
        #     </type>

        #     <type name="diagram">
        #     Image is a diagram, flowchart, or conceptual illustration:
        #     - Describe the structure and relationships shown
        #     - Extract any text labels or annotations
        #     - Explain the concept being illustrated
        #     </type>

        #     <type name="book_cover_title_page">
        #     Image is a book cover, title page, or chapter heading:
        #     - Extract the title, subtitle, author name, and any other visible text
        #     - Preserve the hierarchy (title larger than subtitle, etc.)
        #     </type>

        #     <type name="product_photo">
        #     Image shows a product, item, or object without informative text:
        #     - Skip it and do not return any output for that image
        #     </type>

        #     <type name="person_people_photo">
        #     Image shows a person or crowd of people without informative content:
        #     - Skip it and do not return any output for that image
        #     </type>

        #     </image_types>

        #     <critical_rules>
        #     - For tables: Output ONLY the markdown table with ALL rows and columns. NEVER summarize or explain the content.
        #     - For graphs/charts: ALWAYS include specific data points to support insights. NEVER just read axis labels.
        #     - For text: ALWAYS preserve original formatting, special characters, and line breaks EXACTLY as they appear. NEVER explaining them.
        #     - For images without informative content (product photos, people photos): DO NOT return any output for that image.
        #     </critical_rules>              
                         
        #     <output_format>
        #     For each image, your response should be:
        #     [ID: image_id]
        #     <extracted content or description here>

        #     </output_format>
        #     </instructions>
        #     """),
        #     }
        # ]

        message_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": ("""You are an expert OCR system. Extract content from images containing graphs, tables, diagrams, text, and photos.

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
</examples> 
                 """),
            }
        ]
        for img_id, img in images_dict.items():
            base64_string, media_type = self._image_to_base64_string(img, self.max_image_size)
            
            message_content.extend([
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
                    "text": f"[ID: {img_id}]"
                }
            ])

        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]

        try:
            print(messages)
            response = self.model.invoke(messages)
            response_text = response.content
            print(response_text)
            
            if debug:
                print(f"\n[DEBUG] Model Response:\n{response_text}\n")
                
        except Exception as e:
            error_msg = f"[OCR FAILED: API Error - {str(e)}]"
            logger.error(f"OCR API call failed: {e}", exc_info=True)
            return {img_id: error_msg for img_id in images_dict}

        # Parse response with improved regex that handles optional whitespace
        ocr_results: Dict[str, str] = {}
        
        # More flexible pattern that handles various spacing and newline variations
        pattern = r"<([0-9a-fA-F-]{36})>\s*(.*?)\s*</\1>"

        matches = re.findall(pattern, response_text, flags=re.DOTALL)

        ocr_results = {image_id: content.strip() for image_id, content in matches}
        print(" OCR Results:", ocr_results)
        print(" Images Dict:", images_dict.keys())

        # Add failure messages for missing IDs
        for img_id in images_dict:
            if img_id not in ocr_results:
                logger.warning(f"Model did not return content for image ID: {img_id}")
                ocr_results[img_id] = "[OCR FAILED: Model did not return content for this ID]"
        
        return ocr_results
    
