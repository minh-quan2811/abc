from app.core.config import settings
from langchain_aws import BedrockLLM, ChatBedrock

model_bed = ChatBedrock(
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        model_id="us.anthropic.claude-3-haiku-20240307-v1:0",
        region='us-west-2',
    )
