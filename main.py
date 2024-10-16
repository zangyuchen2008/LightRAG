from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    PDFMinerLoader,
)

load_dotenv()

class DocumentLoader:
    @staticmethod
    def load_document(file_path: str):
        _, file_extension = os.path.splitext(file_path)
        if file_extension == ".pdf":
            return PDFMinerLoader(file_path).load()
        elif file_extension == ".txt":
            return TextLoader(file_path).load()
        elif file_extension == ".csv":
            return CSVLoader(file_path).load()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")


WORKING_DIR = "./千脑智能"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

# Load documents
docs = DocumentLoader.load_document("data/千脑智能.pdf")
# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=0, length_function=len
)

splits = text_splitter.split_documents(docs)


rag.insert([each.page_content for each in splits])

# Perform naive search
import json
with open("data/千脑智能_qa.json") as f:
    data = json.load(f)
contexts = []
responses = []
from tqdm import tqdm
for question in tqdm(data["question"]):
    try:
        response = rag.query(question, param=QueryParam(mode="naive"))
        contexts.append([response["context"]])
        responses.append(response["response"])
    except Exception as e:
        contexts.append(["error"])
        responses.append("error")

data["contexts"] = contexts
data["answer"] = responses  
with open("data/千脑智能_qa_result.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)