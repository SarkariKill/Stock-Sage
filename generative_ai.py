import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

import os
os.environ['GOOGLE_API_KEY'] = "AIzaSyCihuiYQK_yBKxHtduqrMIM8_BtaBOxYKo"

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

## Yeh Rha Hamara Gpt Model
generative_model = genai.GenerativeModel('gemini-pro')


response = generative_model.generate_content("""Write about: VIT Vellore""")
text = response._result.candidates[0].content.parts[0].text
print(text)