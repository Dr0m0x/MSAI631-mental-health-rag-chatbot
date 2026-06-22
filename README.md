# MSAI631-mental-health-rag-chatbot
# Mental Health RAG Chatbot

MSAI-631 Human-Computer Interaction Group Project

## Team Members

- Chris Bellar
- Girmay Haile Gebreselassie
- Moriah Holland
- Sai Shanthan Rao Sagi
- Saikishore Chary
- Vidhi Tusharbhai Sheth
- Yamini Yaramachu

## Project Overview

This project aims to develop a Retrieval-Augmented Generation (RAG) Mental Health Support Chatbot that provides users with reliable mental wellness information through a conversational interface. The system will utilize trusted sources such as World Health Organization (WHO) guidelines and Cognitive Behavioral Therapy (CBT) resources to generate grounded responses while reducing the risk of misinformation.

## Technologies

- Python
- FAISS
- Sentence Transformers
- Gemini 2.5 Flash
- Llama 3.3 70B
- Flan-T5
- Gradio
- Hugging Face Spaces
- GitHub

  ## Source Code Components
The project source code consists of the following primary files:

•	app.py – Gradio-based user interface and application entry point.

•	rag.py – Retrieval-Augmented Generation workflow and response generation logic.

•	ingest.py – Document ingestion, chunking, embedding creation, and FAISS index generation.

•	requirements.txt – Project dependencies.

•	docs/ – Mental health source documents used to build the knowledge base.

•	index.faiss – Vector database used for similarity search.

•	chunks.pkl – Serialized document chunks used during retrieval.

<img width="468" height="346" alt="image" src="https://github.com/user-attachments/assets/4e6ee598-67b6-431c-8055-783ab8884ed9" />


## Sources and Reused Components
Gradio framework

FAISS vector database

Sentence Transformers

Groq API / Llama model

Hugging Face Spaces deployment

WHO and CBT source materials

## Challenges Encountered

Team GitHub access and permissions

Dependency management and package compatibility

Retrieval quality tuning

Deployment configuration

Knowledge base preparation and chunking

## Known Limitations
Educational use only

Not intended for crisis intervention

Limited by source document coverage

Response quality depends on retrieval accuracy

## Execution Status

The application executes successfully locally and on Hugging Face Spaces.

Public deployment available at: [link]

## Status

Project Proposal Phase
Summer 2026
University of the Cumberlands
MSAI-631: Artificial Intelligence for Human-Computer Interaction

## Public deployment available at: https://huggingface.co/spaces/ggebreselassie29144/mental-health-chatbot 
