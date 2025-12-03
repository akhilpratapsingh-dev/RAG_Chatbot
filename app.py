import streamlit as st
from chatbot import retrieve_context, ask_llm, handle_special_modes

# -------------------------------
# STREAMLIT APP UI
# -------------------------------

st.set_page_config(
    page_title="GDGC RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 GDGC Club Information RAG Chatbot")
st.write("Ask anything about your GDGC club. The bot will answer ONLY using club data.")

# Input box
user_query = st.text_input("Enter your question:")

# Ask button
if st.button("Ask"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Check for special modes
        special = handle_special_modes(user_query)
        if special:
            st.subheader("Response")
            st.write(special)
        else:
            # Retrieve context normally
            context = retrieve_context(user_query)
            response = ask_llm(user_query, context)

            st.subheader("Relevant Context")
            st.code(context)

            st.subheader("Chatbot Response")
            st.write(response)

# Footer
st.markdown("---")
st.caption("Built by Akhil Pratap Singh | GDGC RAG Project")