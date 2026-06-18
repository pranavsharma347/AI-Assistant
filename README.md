# IntelliDocs – Multi-Source AI Q&A System

## Overview

IntelliDocs is an AI-powered Question Answering System that allows users to upload multiple PDF documents or provide URLs and ask questions in natural language.

The system processes documents, generates embeddings, performs semantic search, and returns contextual answers using Large Language Models (LLMs).

This project focuses on making document understanding faster and more interactive.

---

## Features

* Upload multiple PDF documents
* URL-based document ingestion
* AI-powered Question Answering
* Semantic Search using Vector Database
* Natural language interaction
* REST APIs for document and query processing
* Embedding generation and retrieval
* Cloud deployment support

---

## Tech Stack

### Backend

* Python
* Django
* Django REST Framework (DRF)

### AI / LLM

* LangChain
* Google Gemini
* Embeddings

### Vector Database

* FAISS
* ChromaDB

### Database

* SQLite

### Cloud / Deployment

* AWS EC2
* AWS Lambda
* AWS ECR
* AWS S3
* AWS Amplify

### Frontend

* React (Basic UI)

---

## System Architecture

User

↓

Upload PDFs / URLs

↓

Document Processing

↓

Text Extraction

↓

Embedding Generation

↓

Vector Database Storage

↓

Semantic Search

↓

LLM (Gemini)

↓

Answer Generation

↓

Response to User

---

## Workflow

### Step 1

User uploads documents or enters URLs.

### Step 2

System extracts text.

### Step 3

Embeddings are generated.

### Step 4

Embeddings are stored in Vector DB.

### Step 5

User asks questions.

### Step 6

Relevant chunks are retrieved.

### Step 7

Gemini generates contextual answers.

---

## Installation

Clone Repository

```bash
git clone <repository-url>
```

Move into project

```bash
cd IntelliDocs
```

Create virtual environment

```bash
virtualenv --python=3.11 venv
```

Activate environment

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create Migration Files

```bash
python3 manage.py makemigrations
```


Apply Database Migration

```bash
python3 manage.py migrate
```

Run server

```bash
python3 manage.py runserver
```

---

## Environment Variables

Create `.env`

```env
GOOGLE_API_KEY=

AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

S3_BUCKET_NAME=
```

## Future Improvements

* Multi-user workspace
* Streaming responses
* Chat history memory
* OCR support
* Hybrid Search

---

## Project Status

Completed – Ready for further enhancements.

---

