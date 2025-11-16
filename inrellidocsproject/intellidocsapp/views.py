import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

from django.shortcuts import render
from rest_framework.views import APIView
from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import FileQuestionSerializer,MultiFileUploadSerializer,MultiURLQuestionSerializer,AIGeneratorSerializer
from rest_framework.response import Response
from rest_framework import status
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader  # or from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQAWithSourcesChain,RetrievalQA
from langchain_classic.agents import initialize_agent,Tool
from langchain_community.tools import DuckDuckGoSearchRun
from rest_framework.parsers import JSONParser
import boto3
import json



# import langchain

load_dotenv()



# Create your views here.
class Docs(APIView):
    serializer_class = FileQuestionSerializer  # without this no file upload form and Question submit not show on browser
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = FileQuestionSerializer(data=request.data)
        print("seralizer",serializer)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['file_uploaded']
            user_question=serializer.validated_data['question']
        
            #step1 Now we first we upload a file
            pdf_reader = PdfReader(uploaded_file)
            #step2 create a splitter

            docs = ""
            for page in pdf_reader.pages:
                docs += page.extract_text() or ""
            
            doc_obj = [Document(page_content=docs)]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,    # ≈ 2000 tokens (safe)
                chunk_overlap=150,  # thoda overlap for context
                )
        
            #splits the documents into chunks
            splits = splitter.split_documents(doc_obj)#always take document object not a string



            # make emdding vector this chunks and store in chroma db
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

            # Step 3: Create Chroma vector store in memory
            vectorstore = Chroma.from_documents(
                documents=splits,
                  embedding=emb,
                  collection_name="my_collection"
                  )
            # Step 4: Convert vectorstore into a retriever and i want similar or top two results based on query
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
            #now we use LLM model
            llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables = ['context', 'question']
            )
            parser=StrOutputParser()

            # question= "What is data structure and explain its types?"
            retrieved_docs    = retriever.invoke(user_question)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chain=prompt|llm|parser
            answer = chain.invoke({"context": context_text,
                                    "question":  user_question
                                    })

            return Response({"answer": answer},status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    


class MultiDocs(APIView):
    serializer_class = MultiFileUploadSerializer
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):

        # 🧩 Step 1: Validate Input
        try:
            files_list = request.FILES.getlist('files_uploaded')
            data = {
                'files_uploaded': files_list,
                'question': request.data.get('question')
            }
            serializer = MultiFileUploadSerializer(data=data)
            serializer.is_valid(raise_exception=True)
            uploaded_files = serializer.validated_data['files_uploaded']
            user_question = serializer.validated_data['question']

        except Exception as e:
            print("⚠️ Serializer Error:", e)
            return Response(
                {
                    "error": "Please upload valid PDF files and enter your question.",
                    "error_type": "input_error"
                },
                status=400
            )

        # 📄 Step 2: Read & Extract Text from PDFs
        try:
            docs_text = ""
            for f in uploaded_files:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    docs_text += page.extract_text() or ""

            if not docs_text.strip():
                return Response(
                    {
                        "error": "Uploaded PDFs have no readable content. Try with different files.",
                        "error_type": "no_content"
                    },
                    status=400
                )

            doc_obj = [Document(page_content=docs_text)]

        except Exception as e:
            print("📄 PDF Read Error:", e)
            return Response(
                {
                    "error": "Unable to read the uploaded PDFs. Please upload valid documents.",
                    "error_type": "pdf_read_error"
                },
                status=500
            )

        # 🧠 Step 3: Create Embeddings + Vector Store
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=1000
            )
            splits = splitter.split_documents(doc_obj)

            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore = Chroma.from_documents(splits, emb, collection_name="multi_pdf_store")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        except Exception as e:
            print("🧩 Embedding Error:", e)
            return Response(
                {
                    "error": "System failed to process your documents. Please retry later.",
                    "error_type": "embedding_error"
                },
                status=500
            )

        # 🤖 Step 4: Initialize LLM
        try:
            llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say: "I don't know the answer based on uploaded files.Please try with different question".

                {context}
                Question: {question}
                """,
                input_variables=["context", "question"]
            )
            parser = StrOutputParser()

        except Exception as e:
            print("🤖 LLM Init Error:", e)
            return Response(
                {
                    "error": "AI system failed to initialize. Please try again after a few seconds.",
                    "error_type": "llm_init_error"
                },
                status=500
            )

        # 💬 Step 5: Retrieve Context + Generate Answer
        try:
            retrieved_docs = retriever.invoke(user_question)
            if not retrieved_docs:
                return Response(
                    {
                        "error": "No matching information found in uploaded PDFs.",
                        "error_type": "no_relevant_content"
                    },
                    status=404
                )

            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            chain = prompt | llm | parser

            answer = chain.invoke({
                "context": context_text,
                "question": user_question
            })

            return Response({"answer": answer}, status=200)

        except Exception as e:
            print("💥 Answer Generation Error:", e)
            return Response(
                {
                    "error": "Something went wrong while generating answer. Please try again.",
                    "error_type": "generation_error"
                },
                status=500
            )


class MultiUrls(APIView):
    def post(self, request):
        serializer = MultiURLQuestionSerializer(data=request.data)
        if not serializer.is_valid():
            return Response({'error': serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

        urls = serializer.validated_data["urls"]
        question = serializer.validated_data["question"]

        print("Incoming URLs:", urls)
        print("Question:", question)

        try:
            # Step 1: Load content from URLs
            try:
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()
            except Exception as e:
                print("❌ URL Loader Error:", e)
                return Response({"error": "Failed to load one or more URLs","error_type": "error in url"}, status=400)

            # Step 2: Split data
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(data)

            if not docs:
                return Response({"error": "Failed to handle urls","error_type": "Failed to handle urls"}, status=404)

            # Step 3: Embeddings + Vector Store
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            try:
                vector_store = FAISS.from_documents(docs, emb)
                vector_store.save_local("langchain_faiss_index")
                new_vs = FAISS.load_local("langchain_faiss_index", emb, allow_dangerous_deserialization=True)
            except Exception as e:
                print("❌ FAISS Error:", e)
                return Response({"error": "Error creating or loading FAISS index","error_type": "error in FAISS index"}, status=500)
             
        
            # Step 4: LLM + Chain
            llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

            template = """
                        You are a helpful assistant.
                        Answer ONLY from the provided context below.
                        If the context does not contain the answer, simply say: 
                        "Answer of this question is not available in the provided urls"

                        Context:
                        {context}

                        Question:
                        {question}

                        Answer:
                        """
            
            prompt = PromptTemplate(
                                    template=template,
                                    input_variables=["context", "question"]
                                 )
            
            # chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=new_vs.as_retriever(),chain_type_kwargs={"prompt": prompt})
            chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=new_vs.as_retriever(),
                            chain_type="stuff",   # default type
                            chain_type_kwargs={"prompt": prompt}
                            )
            
            answer=chain.invoke({"query": question})
            result = answer.get("result", "Answer of this question is not available in the provided urls.Please try with question realted to url context.")
            return Response({"answer": result}, status=200)

        except Exception as e:
            import traceback
            print("🔥 Unexpected Error:", e)
            print(traceback.format_exc())
            print("error",str(e))
            return Response({"answer": "Some thing goes wrong please try again "}, status=500)




class DocsAiAgent(APIView):
    serializer_class = FileQuestionSerializer  # without this no file upload form and Question submit not show on browser
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        serializer = FileQuestionSerializer(data=request.data)
        print("seralizer",serializer)
        if serializer.is_valid():
            print("yes")
            uploaded_file = serializer.validated_data['file_uploaded']
            user_question=serializer.validated_data['question']
        
            #step1 Now we first we upload a file
            pdf_reader = PdfReader(uploaded_file)
            #step2 create a splitter

            docs = ""
            for page in pdf_reader.pages:
                docs += page.extract_text() or ""
            
            doc_obj = [Document(page_content=docs)]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,    # ≈ 2000 tokens (safe)
                chunk_overlap=150,  # thoda overlap for context
                )
        
            #splits the documents into chunks
            splits = splitter.split_documents(doc_obj)#always take document object not a string



            # make emdding vector this chunks and store in chroma db
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

            # Step 3: Create Chroma vector store in memory
            vectorstore = Chroma.from_documents(
                documents=splits,
                  embedding=emb,
                  collection_name="my_collection"
                  )
            # Step 4: Convert vectorstore into a retriever and i want similar or top two results based on query
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            #now we use LLM model
            llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

            search_tool = DuckDuckGoSearchRun()

            tools = [
    Tool(name="PDF Retriever", func=retriever.invoke, description="Get info from PDF"),
    Tool(name="Web Search", func=search_tool.run, description="Search the web when document info is missing"),
]
            
            agent = initialize_agent(tools,llm,
                                     agent_type="zero-shot-react-description",  # simple agent type
                                     verbose=True  # show what the agent is doing
                                     )

            
            raw_output = agent.invoke({"input": user_question})
            # answer = parser.invoke(raw_output["output"])
            answer = raw_output["output"]


            return Response({"answer": answer},status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


        

class AIGenerator(APIView):
    serializer_class=AIGeneratorSerializer
    parser_classes = (FormParser,JSONParser)#form paresr for accept data from html

    def post(self,request):
        serializer=AIGeneratorSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        question=serializer.validated_data['question']
        try:
            llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            parser=StrOutputParser()
            chain=llm|parser
            output=chain.invoke(question)
            return Response({"result":output},status=status.HTTP_200_OK)
        except Exception as e:  
            print("💥 Answer Generation Error:", e)
            return Response(
                {
                    "error": "Something went wrong while generating answer. Please try again.",
                    "error_type": "generation_error"
                },
                status=500
            )


class TestLambdaPDF(APIView):
    def post(self, request):
        # 🧩 Step 1: Validate Input
        try:
            files_list = request.FILES.getlist('files_uploaded')
            data = {
                'files_uploaded': files_list,
                'question': request.data.get('question')
            }
            serializer = MultiFileUploadSerializer(data=data)
            serializer.is_valid(raise_exception=True)
            uploaded_files = serializer.validated_data['files_uploaded']
            user_question = serializer.validated_data['question']

        except Exception as e:
            print("⚠️ Serializer Error:", e)
            return Response(
                {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                    "error": "Please upload valid PDF files and enter your question.",
                    "error_type": "input_error"
                },
                status=400
            )
        
        try:
            s3 = boto3.client('s3', region_name='ap-south-1')
            bucket_name = 'geniehub-docs-bucket'

            uploaded_file_keys = []
            for file_obj in files_list:
                s3_key = f"geniehub-docs-bucket/pdf/{file_obj.name}"
                s3.upload_fileobj(file_obj, bucket_name, s3_key)
                uploaded_file_keys.append(s3_key)

            # ab Lambda call karo (file list + question ke sath)
            lambda_client = boto3.client('lambda', region_name='eu-north-1')

            payload = {
                "bucket": bucket_name,
                "files": uploaded_file_keys,
                "question": user_question
            }

            response = lambda_client.invoke(
                FunctionName='process_lambda_handler',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )

            result = json.load(response['Payload'])
            print("result from lambda",result)
            body = json.loads(result["body"])
            answer = body["answer"]
        except Exception as e:
            print("💥 Lambda Invocation Error:", e)
            return Response(
                {
                    "error": "Something went wrong while invoking Lambda. Please try again.",
                    "error_type": "lambda_invocation_error"
                },
                status=500
            )
        
        return Response({"message": "Files uploaded and Lambda invoked successfully.", "answer": answer}, status=200)


