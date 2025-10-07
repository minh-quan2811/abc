import os
import re
import asyncio
import tempfile
from pathlib import Path
from typing import Union, List
from markitdown import MarkItDown
from langchain_google_genai import ChatGoogleGenerativeAI

# Import refactored components
from app.utils.refactor_file_processor.image_ocr_parser_gemini import ImageOcrParser
from app.utils.refactor_file_processor.pdf_parser import PdfParser
from app.core.config import settings

class FileProcessor:
    """
    Unified file processor that handles multiple document types.
    Uses Gemini OCR for PDF/DOCX, MarkItDown for text files, and Excel parsing.
    """

    def __init__(
        self,
        enable_ocr: bool = True
    ):
        self.md = MarkItDown(enable_plugins=False)
        
        # Setup Gemini OCR pipeline
        self.enable_ocr = enable_ocr
        if enable_ocr:
            self.gemini_model = ChatGoogleGenerativeAI(
                google_api_key=settings.GOOGLE_API_KEY,
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
            self.ocr_parser = ImageOcrParser(model_llm=self.gemini_model)
        else:
            self.ocr_parser = None

    def _clean_text(self, text: str) -> str:
        """Remove Unnamed columns and NaN values."""
        text = re.sub(r"Unnamed:\s*\d+", "", text)
        text = re.sub(r"\bNaN\b", "", text)
        return text

    def _split_sheets(self, markdown_text: str) -> List[str]:
        """Split markdown text into sheets by ## headers."""
        sheets = re.split(r"(?=^## )", markdown_text, flags=re.MULTILINE)
        return [part.strip() for part in sheets if part.strip().startswith("##")]

    def read_excel(self, file_path: str) -> List[str]:
        """
        Read Excel file, convert to Markdown, clean, and split by sheets.
        """
        markdown_text = self.md.convert(file_path).text_content
        markdown_text_clean = self._clean_text(markdown_text)
        return self._split_sheets(markdown_text_clean)

    def read_text_file(self, file_path: str) -> str:
        """
        Read plain text files (txt, md) using MarkItDown.
        """
        return self.md.convert(file_path).text_content

    def _docx_to_pdf(self, docx_path: str) -> str:
        """
        Convert DOCX to PDF.
        """
        try:
            from docx2pdf import convert
            
            # Create temp directory for PDF output
            temp_dir = tempfile.mkdtemp()
            pdf_path = os.path.join(temp_dir, Path(docx_path).stem + ".pdf")
            
            # Convert DOCX to PDF
            convert(docx_path, pdf_path)
            
            if os.path.exists(pdf_path):
                return pdf_path
            else:
                raise RuntimeError("docx2pdf conversion failed - PDF not created")
                
        except ImportError:
            raise RuntimeError(
                "docx2pdf not installed. Install with: pip install docx2pdf"
            )
        except Exception as e:
            raise RuntimeError(f"DOCX to PDF conversion failed: {e}")

    async def _read_pdf_with_ocr(
        self,
        file_path: str,
        batch_size: int = 5,
        filter_images: bool = True
    ) -> str:
        """
        Read PDF using Gemini OCR parser.
        """
        if not self.enable_ocr or self.ocr_parser is None:
            raise RuntimeError("OCR is not enabled. Initialize with enable_ocr=True")
        
        pdf_parser = PdfParser(ocr_parser=self.ocr_parser)
        pdf_parser.load_pdf(file_path)
        await pdf_parser.parse_and_ocr(
            batch_size=batch_size,
            filter_images=filter_images
        )
        return pdf_parser.get_parsed_text()

    def read_pdf(
        self,
        file_path: str,
        batch_size: int = 5,
        filter_images: bool = True
    ) -> str:
        """
        Read PDF file with OCR support (synchronous wrapper).
        """
        return asyncio.run(
            self._read_pdf_with_ocr(file_path, batch_size, filter_images)
        )

    def read_docx(
        self,
        file_path: str,
        batch_size: int = 5,
        filter_images: bool = True
    ) -> str:
        """
        Read DOCX file by converting to PDF and using OCR.
        """
        # Convert DOCX to PDF
        pdf_path = self._docx_to_pdf(file_path)
        
        try:
            # Read PDF with OCR
            content = self.read_pdf(pdf_path, batch_size, filter_images)
            return content
        finally:
            # Cleanup temporary PDF
            if os.path.exists(pdf_path):
                try:
                    os.remove(pdf_path)
                    temp_dir = os.path.dirname(pdf_path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                except Exception:
                    pass

    def read_file(self, file_path: str) -> Union[str, List[str]]:
        """
        Universal file reader that routes to appropriate handler.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = Path(file_path).suffix.lower()
        
        # Excel files
        if file_ext in [".xlsx", ".xls"]:
            return self.read_excel(file_path)
        
        # PDF files
        elif file_ext == ".pdf":
            if self.enable_ocr:
                return self.read_pdf(file_path)
            else:
                return self.md.convert(file_path).text_content
        
        # DOCX files
        elif file_ext == ".docx":
            if self.enable_ocr:
                return self.read_docx(file_path)
            else:
                return self.md.convert(file_path).text_content
        
        # Plain text files
        elif file_ext in [".txt", ".md"]:
            return self.read_text_file(file_path)
        
        # Image files
        elif file_ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"]:
            raise ValueError(
                f"Direct image file processing not supported. "
                f"Images should be embedded in PDF/DOCX files."
            )
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

file_processor = FileProcessor()



# # Example usage
# if __name__ == "__main__":
#     # Initialize processor
#     processor = FileProcessor(
#         gemini_api_key=settings.GOOGLE_API_KEY,
#         enable_ocr=True
#     )
    
#     # Test PDF
#     print("Testing PDF...")
#     pdf_content = processor.read_file(r"C:\Users\PC\Downloads\data-ai-marketing\data-ai-marketing_2209\６１６ＭＫ戦略目標データ活用③.pdf")
#     with open("text.md", 'w', encoding='utf-8') as f:
#         f.write(pdf_content)
#     print(f"✓ PDF parsed successfully! Output saved to text.md ({len(pdf_content)} chars)")
    
    
#     # Test DOCX
#     print("\nTesting DOCX...")
#     docx_content = processor.read_file(r"C:\Users\PC\Downloads\data-ai-marketing\data-ai-marketing_2209\ＪＣＡＩ００５棚割.docx")
#     with open("text.md", 'w', encoding='utf-8') as f:
#         f.write(docx_content)
#     print(f"✓ PDF parsed successfully! Output saved to text.md ({len(docx_content)} chars)")

#     # Test Excel
#     print("\nTesting Excel...")
#     excel_sheets = processor.read_file("test.xlsx")
#     print(f"Excel sheets: {len(excel_sheets)}")
    
#     # Test text file
#     print("\nTesting TXT...")
#     txt_content = processor.read_file(r"C:\Users\PC\Documents\Semantic Chunking.txt")
#     with open("text.md", 'w', encoding='utf-8') as f:
#         f.write(txt_content)
#     print(f"✓ PDF parsed successfully! Output saved to text.md ({len(txt_content)} chars)")