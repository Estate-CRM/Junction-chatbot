import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv("test.env")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)


try:
    df = pd.read_csv(r"C:\Users\MSI GAMER\Desktop\LangChain\properties.csv")
    docs = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
except FileNotFoundError:
    raise FileNotFoundError("⚠️ Properties CSV file not found at the specified path")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


faq_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=False
)


def classify_input(input_text, examples):
    examples_str = "\n".join(f'"{e["Input"]}" → {e["Intent"]}' for e in examples)
    prompt = f"""
Classify the following user input as either "FAQ" or "ACTION":

Examples:
{examples_str}

Input: "{input_text}"
Intent:
""".strip()
    response = llm.invoke(prompt)
    return response.content.strip()


def run_action_chat(initial_input):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(history=[])

    action_prompt = f"""
You are a helpful assistant whose job is to help users fill out a service request form.

Form fields:
- category
- description
- budget
- urgency
- location (longitude & latitude)

Rules:
- Ask one field at a time.
- Wait for the user to say "finish" before returning the form.
- Respond only with: DONE_JSON: {{...}} when finished.

User input:
"{initial_input}"
""".strip()

    print("Assistant:", chat.send_message(action_prompt).text)
    while True:
        user_input = input("User: ")
        response = chat.send_message(user_input).text
        print("Assistant:", response)

        if "DONE_JSON" in response:
            import re, json
            match = re.search(r'DONE_JSON\s*:\s*(\{.*\})', response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    print("✅ Final Collected Data:", data)
                    break
                except Exception as e:
                    print("⚠️ Error parsing JSON:", e)


def main():
    examples = [
        {"Input": "Where is your office?", "Intent": "FAQ"},
        {"Input": "What are your working hours?", "Intent": "FAQ"},
        {"Input": "Do you have properties near Hydra?", "Intent": "FAQ"},
        {"Input": "I want to rent an apartment under 5 million DA", "Intent": "ACTION"},
        {"Input": "Assign a plumber please", "Intent": "ACTION"},
        {"Input": "Schedule a visit", "Intent": "ACTION"},
    ]
    print("🔹 Welcome to the Real Estate Chatbot!")
    while True:
        input_text = input("\nUser: ")
        intent = classify_input(input_text, examples)
        print("🔎 Detected Intent:", intent)

        if intent == "FAQ":
            result = faq_chain.run(input_text)
            print("Bot:", result)
        elif intent == "ACTION":
            run_action_chat(input_text)
        else:
            print("🤖 Sorry, I didn't understand the intent of your request.")

if __name__ == "__main__":
    main()
