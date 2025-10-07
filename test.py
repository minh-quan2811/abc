from app.utils.file_processer.image_ocr_parser import ImageOcrParser
from app.utils.file_processer.pdf_parser import PdfParser
# from langchain_anthropic import ChatAnthropic
from app.services.llm_agent.model_manager import model_manager
from app.core.config import settings
from langchain_google_genai import ChatGoogleGenerativeAI

import asyncio
from langchain_aws import BedrockLLM, ChatBedrock
# model_bed = ChatBedrock(
#             aws_access_key_id="",
#             aws_secret_access_key="",
#             model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
#             region='us-west-2',
#         )


model_bed = ChatGoogleGenerativeAI(
    google_api_key=settings.GOOGLE_API_KEY,
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# 1. Setup model
# llm = model_manager.get_llm_ocr()
# 2. Khởi tạo
ocr_parser = ImageOcrParser(model_llm=model_bed)
pdf_parser = PdfParser(ocr_parser=ocr_parser)

# 3. Parse PDF
async def parse_pdf():
    pdf_parser.load_pdf(fr"C:\Users\PC\Downloads\data-ai-marketing\data-ai-marketing_2209\Lewis.pdf")
    await pdf_parser.parse_and_ocr()
    return pdf_parser.get_parsed_text()

# 4. Chạy
text = asyncio.run(parse_pdf())
# print(text)
with open("text.md", 'w', encoding='utf-8') as f:
    f.write(text)


# from PIL import Image
# text = ocr_parser.parse({'1':Image.open(fr"c:\Users\four\Downloads\Untitled.jpg")}, debug=True)
# print(text)