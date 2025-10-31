import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

from django.shortcuts import render
from rest_framework.views import APIView
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter,TokenTextSplitter
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
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from rest_framework.parsers import JSONParser



import langchain

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

    serializer_class = MultiFileUploadSerializer  # without this no file upload form and Question submit not show on browser
    parser_classes = (MultiPartParser, FormParser)

    def post(self,request):
        # Step 1: Convert request.FILES into a list structure serializer expects
        files_list = request.FILES.getlist('files_uploaded')

                # Step 2: Manually prepare data for serializer
        data = {
            'files_uploaded': files_list,
            'question': request.data.get('question')
        }

        serializer=MultiFileUploadSerializer(data=data)
        if serializer.is_valid():
            uploaded_file = serializer.validated_data['files_uploaded']
            user_question=serializer.validated_data['question']

            docs = ""
            for f in uploaded_file:
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    docs += page.extract_text() or ""
            
            doc_obj = [Document(page_content=docs)]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,    # ≈ 2000 tokens (safe)
                chunk_overlap=1000,  # thoda overlap for context
                )
        
            #splits the documents into chunks
            splits = splitter.split_documents(doc_obj)#always take document object not a string

            # make emdding vector this chunks and store in chroma db
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

            # Step 3: Create Chroma vector store in memory
            vectorstore = Chroma.from_documents(
                documents=splits,
                  embedding=emb,
                  collection_name="multipdf_collection"
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

        return Response({"answer":"Question related answer is not found please try again"},status=status.HTTP_200_OK)
    


class MultiUrls(APIView):
    def post(self,request):
        serializer = MultiURLQuestionSerializer(data=request.data)
        if serializer.is_valid():
            urls = serializer.validated_data["urls"]
            question = serializer.validated_data["question"]


            print(urls)
            print(question)

            loader=UnstructuredURLLoader(urls=urls)
            data=loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000,    # ≈ 2000 tokens (safe)
                chunk_overlap=200)  # thoda overlap for context
            docs=splitter.split_documents(data)#here these documents now divide into chunks
            
            #now we create embedding of these chunks
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_store = FAISS.from_documents(docs,emb)#it only converts documents into embedding not store on the disk
            vector_store.save_local("langchain_faiss_index")#now it is locally stored on disk by creating folder
            new_vs = FAISS.load_local("langchain_faiss_index", emb, allow_dangerous_deserialization=True)#Now load embinng we already saved
            llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=new_vs.as_retriever())
            parser=StrOutputParser()
            answer=chain.invoke({"question":question})``
            return Response({"answer":answer}, status=status.HTTP_200_OK)
    
        return Response({"answer":"Question related answer is not found please try again"},status=status.HTTP_200_OK)





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
        if serializer.is_valid():
            question=serializer.validated_data['question']
            print("serialzer",serializer)
            llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
            parser=StrOutputParser()
            chain=llm|parser
            output=chain.invoke(question)
            return Response({"result":output},status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            