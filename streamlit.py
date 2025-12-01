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
import streamlit as st

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Chatbot Generative AI",
    page_icon="🤖",
    layout="wide"
)


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
8. PENTING: Gunakan simbol • (bullet) untuk list, BUKAN tanda bintang (*) atau dash (-)
9. JANGAN gunakan markdown formatting seperti **bold** atau *italic* dalam jawaban
10. Format list harus seperti ini:
    • Poin pertama
    • Poin kedua
      • Sub-poin jika ada

Jawaban:"""

# === FUNGSI UTILITY ===
@st.cache_data
def extract_text_from_pdf(pdf_path):
    """Ekstrak teks dari file PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error membaca PDF: {e}")
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

# === INISIALISASI CHATBOT ===
@st.cache_resource
def initialize_chatbot():
    """Inisialisasi chatbot dengan vectorstore dan memory"""
    
    # === 1. Load API Key dari .env ===
    try:
        # Coba dari Streamlit Cloud Secrets dulu
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except:
        # Kalau gagal, coba dari .env (untuk local development)
        load_dotenv("GOOGLE_API_KEY.env")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        st.error("⚠️ GOOGLE_API_KEY tidak ditemukan! Pastikan ada di file GOOGLE_API_KEY.env")
        st.stop()

    # Konfigurasi Gemini
    genai.configure(api_key=GOOGLE_API_KEY)

    # === 2. Baca data dari PDF ===
    PDF_FILE = "Buku-Panduan-_-Penggunaan-Generative-AI-pada-Pembelajaran-di-Perguruan-Tinggi-cetak_compressed.pdf"

    pdf_text = extract_text_from_pdf(PDF_FILE)

    if not pdf_text:
        st.error("⚠️ Tidak ada teks yang diekstrak dari PDF!")
        st.stop()
        
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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

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
    model_gemini = st.session_state.get('selected_model', 'gemini-2.0-flash')
    llm = ChatGoogleGenerativeAI(
        model=model_gemini,
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
    
    return chatbot

# === MAIN APP ===
def main():
    # Header
    st.markdown('<div class="main-header">🤖 Chatbot Generative AI - Panduan Pembelajaran</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ Informasi")
        st.write("Tanyakan tentang penggunaan AI dalam pembelajaran!")
        st.write("Sumber: Buku Panduan - Penggunaan Generative AI pada Pembelajaran di Perguruan Tinggi")
        
        st.divider()
        st.subheader("⚙️ Pengaturan Model")
        
        model_options = {
            "Gemini 2.0 Flash (Rekomendasi)": "gemini-2.0-flash",
            "Gemini 2.0 Flash Lite (Lebih Ringan)": "gemini-2.0-flash-lite",
            "Gemini 2.5 Flash Lite (Alternatif)": "gemini-2.5-flash-lite"
        }
        
        selected_model_display = st.selectbox(
            "Pilih Model AI:",
            options=list(model_options.keys()),
            index=0
        )
        
        # Simpan model yang dipilih
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_options[selected_model_display]
        
        # Update jika ada perubahan
        new_model = model_options[selected_model_display]
        if new_model != st.session_state.selected_model:
            st.session_state.selected_model = new_model
            if 'chatbot' in st.session_state:
                del st.session_state.chatbot  # Reset chatbot dengan model baru
            st.rerun()
        
        st.info("💡 **Tips:** Jika mendapat error limit/quota, coba ganti ke model Lite atau tunggu beberapa saat.")
        
        st.divider()
        
        if st.button("🔄 Reset Percakapan"):
            st.session_state.messages = []
            st.rerun()
    
    # Inisialisasi chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Memuat chatbot..."):
            st.session_state.chatbot = initialize_chatbot()
        st.success("Silahkan mulai bertanya")
    
    # Inisialisasi chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><b>Anda:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><b>🤖 Chatbot:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
    
    # Input pertanyaan
    query = st.chat_input("Anda:")
    
    if query:
        # Tambah pertanyaan user ke history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Proses dengan chatbot
        try:
            response = st.session_state.chatbot.invoke({"question": query})
            
            # Bersihkan Markdown formatting
            answer = response['answer']
            answer = re.sub(r'^[\*\-\+]\s+', '• ', answer, flags=re.MULTILINE)

            answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)  # Hapus bold **text**
            answer = re.sub(r'(?<!^)(?<!\n)\*([^\*\n]+?)\*', r'\1', answer, flags=re.MULTILINE)      # Hapus italic *text*
            answer = re.sub(r'\n•', '<br>•', answer)
            
            # Tambah jawaban bot ke history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # Rerun untuk update chat
            st.rerun()
            
        except Exception as e:
            error_message = str(e)
            
            if "quota" in error_message.lower() or "limit" in error_message.lower() or "429" in error_message:
                st.error("⚠️ **Limit API Tercapai!**\n\nSilakan:\n• Ganti model ke versi Lite di sidebar\n• Atau tunggu beberapa saat dan coba lagi")
            else:
                st.error(f"❌ Error: {e}")

if __name__ == "__main__":

    main()



