from dotenv import load_dotenv
load_dotenv()
import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------------
# LOAD API KEY
# -------------------------------
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise Exception("Error: OPENROUTER_API_KEY not found in .env file.")

MODEL_NAME = "google/gemma-2-9b-it"   # reliable free model

# -------------------------------
# LOAD CLUB DATA
# -------------------------------
with open("club_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = [c.strip() for c in raw_text.split("\n") if len(c.strip()) > 5]

# Extract past events manually (unique feature)
past_events = [c for c in chunks if "TECH WINTER BREAK" in c.upper() or
                                      "GOOGLE OLYMPICS" in c.upper() or
                                      "BUILD WITH AI" in c.upper()]

# -------------------------------
# EMBEDDING MODEL
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedder.encode(chunks, normalize_embeddings=True)

# -------------------------------
# RETRIEVAL FUNCTION
# -------------------------------
def retrieve_context(query, top_k=3):
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, q_emb)
    top_idx = scores.argsort()[-top_k:][::-1]
    context = "\n".join([chunks[i] for i in top_idx])
    return context


# -------------------------------
# OPENROUTER QUERY
# -------------------------------
def ask_llm(query, context):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system = (
        "You are a GDGC Club Assistant. You MUST answer ONLY using the given context. "
        "If the information is not present, reply: 'Information not available in club data.'"
    )

    user = f"Context:\n{context}\n\nQuestion: {query}"

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    }

    r = requests.post(url, json=data, headers=headers)
    j = r.json()

    if "choices" not in j:
        return f"API Error: {j}"

    return j["choices"][0]["message"]["content"].strip()


# -------------------------------
# UNIQUE FEATURE HANDLING
# -------------------------------
def handle_special_modes(query):
    q = query.lower()

    # --- Past Events Memory Mode ---
    if "past events" in q or "previous events" in q:
        if past_events:
            result = "Here are the past events:\n\n"
            for e in past_events:
                result += f"- {e}\n\n"
            return result
        else:
            return "No past events found in club data."

    # --- Event Suggestion Mode ---
    if "suggest" in q or "event idea" in q:
        return (
            "Here are some event suggestions based on your club:\n\n"
            "• AI for Beginners Workshop\n"
            "• Flutter App Building Marathon\n"
            "• Hackathon: Build for Bhopal\n"
            "• ML Bootcamp for Freshers\n"
            "• Cloud Computing with Google Cloud\n"
        )

    # --- Smart Summary Mode ---
    if "summary" in q or "summarize" in q:
        context = " ".join(chunks[:10])
        return ask_llm("Give a short summary of this club.", context)

    return None  # no special mode activated


# -------------------------------
# CHAT LOOP
# -------------------------------
def start_chat():
    print("\n==============================")
    print(" Unique GDGC RAG Chatbot Ready!")
    print(" Type 'exit' to finish.")
    print("==============================\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("\nBot: Goodbye! 👋")
            break

        # Check for unique modes
        special = handle_special_modes(user_input)
        if special:
            print("\nBot:", special, "\n")
            continue

        # Else do normal RAG
        context = retrieve_context(user_input)
        answer = ask_llm(user_input, context)

        print("\nBot:", answer, "\n")


# -------------------------------
# RUN PROGRAM
# -------------------------------
if __name__ == "__main__":
    start_chat()