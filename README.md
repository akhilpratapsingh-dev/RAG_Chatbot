# 🤖 GDGC Club Information RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built for the GDGC ML Team Task Round.  
It answers questions strictly using the official GDGC club data.

---

## 🚀 Features

- 🔍 *RAG Pipeline* (Embeddings + Vector Search + LLM)
- 🧠 *Anti-Hallucination* (Answers only from club_data.txt)
- 📅 *Past Events Mode* → show past events
- 💡 *Event Suggestions Mode* → suggest an event
- 📝 *Club Summary Mode* → summarize the club
- 🌐 *Streamlit UI* + Terminal chatbot

---

## 🗂 Project Files

- chatbot.py → Main RAG chatbot logic  
- app.py → Streamlit UI  
- club_data.txt → Club knowledge base  
- requirements.txt → Dependencies  
- .env → API Key (not uploaded)  
- .gitignore → Hides .env  

---

## 🔧 How to Run

### 1️⃣ Install requirements
pip install -r requirements.txt

### 2️⃣ Add API key in .env

OPENROUTER_API_KEY=your-key-here

### 3️⃣ Run Terminal Chatbot

python chatbot.py

### 4️⃣ Run Streamlit UI

streamlit run app.py

Open in browser:  
http://localhost:8501

---

## 📌 Built With
- Sentence Transformers  
- NumPy  
- OpenRouter (Gemma-2-9B-IT)  
- Streamlit  
- Python  

---

## ✨ Author  
*Akhil Pratap Singh*  
GDGC - ML Team Task Round
