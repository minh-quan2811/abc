import fitz
from PIL import Image
import io
from typing import List, Dict, Any, Generator
import logging
import uuid
import asyncio
import itertools
import re

logging.basicConfig(level=logging.INFO)


class PdfParser:
    """
    Simplified PDF parser that extracts text and images with placeholders.
    Image quality filtering is delegated to ImageOcrParser.
    """
    
    def __init__(self, ocr_parser=None):
        """
        Initialize PDF parser.
        
        Args:
            ocr_parser: ImageOcrParser instance for OCR processing
        """
        self.ocr_parser = ocr_parser
        self.pdf_path = None
        self.doc = None
        self.parsed_document = None
        self.document_image = {}

        if self.ocr_parser is None:
            logging.warning(
                "ocr_parser not provided. OCR functionality will not work."
            )

    def load_pdf(self, pdf_path: str):
        """Load a PDF file and initialize parsing state."""
        self.pdf_path = pdf_path
        try:
            self.doc = fitz.open(pdf_path)
            self.parsed_document = []
            self.document_image = {}
        except Exception as e:
            logging.error(f"Cannot open PDF file: {e}")
            self.doc = None
            self.parsed_document = None

    def parse_pdf(self):
        """Parse PDF and extract text with image placeholders."""
        if self.doc is None:
            logging.error(
                "PDF document not loaded. Please call load_pdf() first."
            )
            return

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            parsed_text = self._process_page(page, page_num)
            self.parsed_document.append(parsed_text)

    async def parse_and_ocr(self, batch_size: int = 5, filter_images: bool = True):
        """
        Parse PDF and perform OCR on extracted images.
        
        Args:
            batch_size: Number of images to process per batch
            filter_images: If True, apply quality filtering in OCR parser
            
        Returns:
            List of page texts with OCR results integrated
        """
        if self.ocr_parser is None:
            logging.error(
                "Cannot perform OCR because ocr_parser was not provided."
            )
            self.parse_pdf()
            return self.parsed_document
            
        self.parse_pdf()

        if not self.document_image:
            logging.info("No images found for OCR processing.")
            return self.parsed_document

        logging.info(
            f"Starting OCR processing for {len(self.document_image)} images "
            f"in batches of {batch_size}..."
        )
        
        ocr_results = await self._perform_ocr_on_images(
            batch_size, filter_images
        )
        
        logging.info(
            f"Completed. Received results for {len(ocr_results)} images."
        )
        
        self._integrate_ocr_results(ocr_results)
        logging.info("parser.parsed_document successfully updated.")
        
        return self.parsed_document

    def _process_page(self, page, page_num: int) -> str:
        """
        Extract text and images from a single page.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number
            
        Returns:
            Page text with image placeholders
        """
        page_content = page.get_text('dict', sort=True)
        page_full_text = ""

        for block in page_content["blocks"]:
            # Skip content in margins (50px from edges)
            bbox = block["bbox"]
            if (bbox[0] < 50 or bbox[1] < 50 or 
                bbox[2] > page.rect.width - 50 or 
                bbox[3] > page.rect.height - 50):
                continue

            # Process text blocks
            if block["type"] == 0:
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "
                block_text += "\n"
                block_text = ' '.join(block_text.split()) + "\n"
                page_full_text += block_text
            
            # Process image blocks
            elif block["type"] == 1:
                try:
                    image_bytes = block["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Store all images - filtering will be done by OCR parser
                    image_id = str(uuid.uuid4())
                    self.document_image[image_id] = image
                    block_placeholder_text = f'Image-Placeholder{{{image_id}}}'
                    page_full_text += block_placeholder_text + "\n"
                    
                except Exception as e:
                    logging.warning(f"Failed to process image on page {page_num}: {e}")

        return page_full_text

    def _create_image_batches(
        self, batch_size: int
    ) -> Generator[Dict[str, Any], None, None]:
        """Split images into batches for processing."""
        it = iter(self.document_image.items())
        while True:
            batch_slice = itertools.islice(it, batch_size)
            batch_dict = dict(batch_slice)
            if not batch_dict:
                break
            yield batch_dict

    async def _perform_ocr_on_images(
        self, batch_size: int, filter_images: bool
    ) -> Dict[str, str]:
        """
        Perform OCR on all images in batches.
        
        Args:
            batch_size: Number of images per batch
            filter_images: Whether to apply quality filtering
            
        Returns:
            Dictionary mapping image IDs to OCR text
        """
        all_results = {}
        tasks = []
        
        for batch in self._create_image_batches(batch_size):
            task = asyncio.to_thread(
                self.ocr_parser.parse,
                batch,
                filter_images=filter_images
            )
            tasks.append(task)
        
        batch_results_list = await asyncio.gather(*tasks)
        
        for result_dict in batch_results_list:
            all_results.update(result_dict)
            
        return all_results

    def _integrate_ocr_results(self, ocr_results: Dict[str, str]):
        """
        Replace image placeholders with OCR text.
        
        Args:
            ocr_results: Dictionary mapping image IDs to OCR text
        """
        for i in range(len(self.parsed_document)):
            modified_text = self.parsed_document[i]
            found_placeholders = re.findall(
                r'Image-Placeholder{([a-fA-F0-9-]+)}',
                modified_text
            )
            
            for img_id in found_placeholders:
                full_placeholder_string = f"Image-Placeholder{{{img_id}}}"
                ocr_text = ocr_results.get(img_id, "").strip()
                modified_text = modified_text.replace(
                    full_placeholder_string,
                    ocr_text
                )
            
            self.parsed_document[i] = modified_text

    def get_parsed_text(self) -> str:
        """Get full parsed document as single string."""
        if self.parsed_document is None:
            return ""
        return "\n\n".join(self.parsed_document)

    def get_page_text(self, page_num: int) -> str:
        """Get text for specific page."""
        if (self.parsed_document is None or 
            page_num < 0 or 
            page_num >= len(self.parsed_document)):
            return ""
        return self.parsed_document[page_num]

    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None