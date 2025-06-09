import google.generativeai as genai
from ..config import settings

class GeminiModel:
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text response from Gemini model"""
        response = self.model.generate_content(prompt, **kwargs)
        return response.text

    def generate_stream(self, prompt: str, **kwargs):
        """Stream response from Gemini model"""
        response = self.model.generate_content(prompt, stream=True, **kwargs)
        for chunk in response:
            yield chunk.text

gemini_model = GeminiModel()
