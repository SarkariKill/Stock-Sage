import pathlib
import textwrap

import google.generativeai as genai


from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
    text = text.replace('.','*')
    return Markdown(textwrap.indent(text,'>',predicate = lambda _:True))

import os
os.environ['GOOGLE_API_KEY'] = "AIzaSyBNkMTIJmEFh02PmRf18lEzst6-IEUUKwo"
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

model = genai.GenerativeModel('gemini-pro')

print(model)

respose = model.generate_content("What is the meaning of life ?")

to_markdown(respose.text)