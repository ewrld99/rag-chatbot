import os
import sys

# Change to project dir
os.chdir(r"c:\Users\hp\myfinalproject\rag-chatbot")
sys.path.insert(0, r"c:\Users\hp\myfinalproject\rag-chatbot")

from app.core.config import settings

with open("scratch_output.txt", "w") as f:
    f.write(f"GROQ_MODEL = {settings.GROQ_MODEL}\n")
    f.write(f"ENV_GROQ_MODEL = {os.environ.get('GROQ_MODEL')}\n")
