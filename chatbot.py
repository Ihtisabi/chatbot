import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import re
from langchain_community.embeddings import HuggingFaceEmbeddings


# === 1. Load API Key dari .env ===
load_dotenv("GOOGLE_API_KEY.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("⚠️ GOOGLE_API_KEY tidak ditemukan! Pastikan ada di file GOOGLE_API_KEY.env")

# Konfigurasi Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# === CUSTOM PROMPT TEMPLATE ===
CUSTOM_PROMPT_TEMPLATE = """Kamu adalah asisten AI yang membantu menjelaskan tentang penggunaan Generative AI dalam pembelajaran di perguruan tinggi.

Gunakan konteks berikut untuk menjawab pertanyaan dengan akurat dan informatif:

{context}

Riwayat Percakapan:
{chat_history}

Pertanyaan: {question}

Instruksi Penting:
1. Jawab dengan bahasa Indonesia yang jelas dan mudah dipahami
2. Gunakan poin-poin atau numbering jika menjelaskan beberapa hal
3. Berikan contoh konkret jika relevan
4. Jika informasi tidak ada dalam konteks, katakan dengan jujur
5. Jangan membuat informasi yang tidak ada dalam dokumen
6. Gunakan format yang rapi dan terstruktur
7. Akhiri dengan pertanyaan follow-up jika relevan untuk membantu pengguna

Jawaban:"""

# === 2. Baca data dari PDF ===
def extract_text_from_pdf(pdf_path):
    """Ekstrak teks dari file PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"❌ Error membaca PDF: {e}")
        return ""

def clean_text(text):
    text = re.sub(r'[*◆▪▫➢➤→◉○■□✓✔✗✘]', '•', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n', text)
    
    text = text.replace('…', '...')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    text = text.strip()
    
    return text

# Path ke file PDF Anda
PDF_FILE = "Buku-Panduan-_-Penggunaan-Generative-AI-pada-Pembelajaran-di-Perguruan-Tinggi-cetak.pdf"

pdf_text = extract_text_from_pdf(PDF_FILE)

if not pdf_text:
    print("⚠️ Tidak ada teks yang diekstrak dari PDF!")
    exit()
    
pdf_text = clean_text(pdf_text)

# === 3. Split teks menjadi chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(pdf_text)
docs = [Document(page_content=chunk) for chunk in chunks]

# === 4. Buat vectorstore dengan Embeddings LOKAL (Gratis & Unlimited) ===

# Gunakan model lokal yang ringan dan gratis
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = FAISS.from_documents(docs, embeddings)
print("✅ Vector store berhasil dibuat\n")

custom_prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "chat_history", "question"]
)

# === 5. Setup memory untuk chatbot ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# === 6. Buat chatbot dengan Gemini ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    convert_system_message_to_human=True
)

chatbot = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt},
    verbose=False
)

# === 7. Simulasi percakapan ===
print("🤖 Chatbot Generative AI - Panduan Pembelajaran")
print("=" * 60)
print("Tanyakan tentang penggunaan AI dalam pembelajaran!")
print("Ketik 'exit', 'quit', atau 'keluar' untuk mengakhiri\n")

while True:
    query = input("Anda: ")
    
    if query.lower() in ["exit", "quit", "keluar"]:
        print("🤖 Chatbot: Terima kasih, sampai jumpa!")
        break
    
    if not query.strip():
        continue
    
    try:
        response = chatbot.invoke({"question": query})
        # Bersihkan Markdown formatting
        answer = response['answer']
        answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)  # Hapus bold **text**
        answer = re.sub(r'\*(.+?)\*', r'\1', answer)      # Hapus italic *text*
        answer = re.sub(r'^[\*\-\+]\s+', '• ', answer, flags=re.MULTILINE)  # Ganti * dengan •
        
        print(f"\n🤖 Chatbot: {answer}\n")

        
    except Exception as e:
        print(f"❌ Error: {e}\n")