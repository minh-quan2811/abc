import asyncio
from app.utils.refactor_file_processor.image_ocr_parser_gemini import ImageOcrParser
from app.utils.refactor_file_processor.pdf_parser import PdfParser
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
# from app.utils.refactor_file_processor.image_paddle_ocr import ImageOcrParserPaddle


# Setup Gemini model via LangChain
model = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyAlJm0ZrIOTwmHOpxAxLV9mLV6knf3T82M",
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
ocr_parser = ImageOcrParser(model_llm=model)

# ocr_parser = ImageOcrParserPaddle(
#     lang='en',
# )

# Initialize parsers
pdf_parser = PdfParser(ocr_parser=ocr_parser)


# Parse PDF
async def parse_pdf():
    pdf_parser.load_pdf(r"C:\Users\PC\Downloads\data-ai-marketing\data-ai-marketing_2209\６１６ＭＫ戦略目標データ活用③.pdf")
    await pdf_parser.parse_and_ocr(batch_size=5, filter_images=False)
    return pdf_parser.get_parsed_text()


# Run
if __name__ == "__main__":
    text = asyncio.run(parse_pdf())
    with open("text.md", 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✓ PDF parsed successfully! Output saved to text.md ({len(text)} chars)")


# Test single image (optional)
# from PIL import Image
# text = ocr_parser.parse(
#     {'test-id-001': Image.open(r"c:\Users\four\Downloads\Untitled.jpg")}, 
#     debug=True
# )
# print(text)


# prompt_path="app/utils/file_processer/ocr_prompt.md"
# with open(prompt_path, 'r', encoding='utf-8') as f:
#     user_prompt = f.read()

# print(user_prompt)