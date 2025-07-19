# Real Estate AI Chatbot with Gemini & LangChain

This project implements a smart real estate chatbot that can:
- Answer user questions from a property CSV (FAQ mode).
- Guide users to fill service or property request forms interactively (ACTION mode).

It uses Google’s **Gemini 1.5 Flash** via the LangChain framework, and FAISS for fast similarity-based document retrieval.

---

## Features

- Natural language understanding using Gemini LLM.
- Classifies queries as:
  - **FAQ** (e.g., “Do you have properties in Hydra?”)
  - **ACTION** (e.g., “I want to rent an apartment under 5 million DA.”)
- Loads real estate data from CSV and searches it using FAISS.
- Fills out multi-step forms using conversational memory and LLM prompts.
- Works as a command-line (terminal) chatbot.

---

## Technologies Used

- Python 3
- [LangChain](https://github.com/langchain-ai/langchain)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [FAISS](https://github.com/facebookresearch/faiss)
- Pandas
- dotenv (for loading API keys)



