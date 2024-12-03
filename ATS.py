import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import google.generativeai as genai
import pandas as pd
import json
from io import BytesIO
import tempfile

# Load environment variables
load_dotenv()

# Set up embeddings
HF_TOKEN = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# API keys
api_key = os.getenv('API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GEMINI_API_KEY)

# Set up generative model
model = genai.GenerativeModel('gemini-pro')


def create_job_post(job_title, location, exp):
    prompt = (
        f"Create a job opening post for platforms like Internshala, LinkedIn, and Naukri.com. "
        f"The post should include the job title: {job_title}, location: {location}, and required experience: {exp}. "
        f"Make it attractive, include skills (if possible but the skills in boxes and highlight them) required, who can apply, benefits, and other necessary details. "
        f"The post should be 100-200 words."
    )

    try:
        # Replace 'model.generate_content' with the actual method to generate content
        response = model.generate_content(prompt)  # assuming 'model' is defined elsewhere
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"


# Streamlit app title
st.title("Recruitment AI")

# Job title, location, and experience input fields
job_title = st.text_input("Enter the job title you are looking for")
location = st.text_input("Enter the location you are looking for")
exp = st.text_input("Enter the experience you are looking for")

# Button to generate job post
if st.button("Create Job Post"):
    if job_title and location and exp:
        job_post = create_job_post(job_title, location, exp)
        st.write(job_post)
    else:
        st.warning("Please fill in all fields to generate the job post.")

job_post = create_job_post(job_title, location, exp)
# Resume scoring section

llm = ChatGroq(groq_api_key=api_key, model_name="gemma-7b-it")
llm_2 = ChatGroq(groq_api_key=api_key, model_name="gemma-7b-it")



jd=st.file_uploader("Upload Job Description", type="pdf")
def get_jd(jd):
    try:
        # Use a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(jd.getvalue())
            temp_pdf_path = temp_pdf.name

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

    finally:
        # Remove the temporary file after processing
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

    # Text splitting for embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    # Create FAISS vectorstore for retrieval
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # Define prompt and QA chain
    system_prompt = (
        f"Extract the Job description from the uploaded file in proper format."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{context}\n{input}"),
        ]
    )

    # Create question-answering chain
    question_answer_chain = create_stuff_documents_chain(llm_2, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    try:
        # Retrieve the job description using the chain
        response = rag_chain.invoke({
            "input": "Describe the job description in proper format"
        })
        job_description = response["answer"]

        return job_description

    except Exception as e:
        raise RuntimeError(f"Error retrieving job description: {e}")

if jd:
    job_description = get_jd(jd)



    # File uploader for PDF resumes
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)


# Function to process PDFs in batches of 4




def process_pdfs_in_batches(files):
    batch_size = 4
    num_batches = (len(files) // batch_size) + (1 if len(files) % batch_size != 0 else 0)
    all_json_data = []

    for i in range(num_batches):
        batch = files[i * batch_size: (i + 1) * batch_size]  # Select a batch of files
        documents = []  # List to hold all document contents

        for j, uploaded_file in enumerate(batch):
            try:
                # Use a temporary file to save the uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(uploaded_file.getvalue())
                    temp_pdf_path = temp_pdf.name

                # Load the PDF using PyPDFLoader
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                documents.extend(docs)

            finally:
                # Remove the temporary file after processing
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

        # Text splitting for embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Create FAISS vectorstore for retrieval
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()

        # Define prompt and QA chain
        system_prompt = (
            f"You are a smart AI agent tasked with evaluating resumes against the job description: "
            f"Job Title: {job_title}, Location: {location}, Experience: {exp}. "
            f"Your evaluation should provide a score (0-100) for each resume based on skills, experience, and other factors. "
            f"Extract the following details from each uploaded PDF: Name, Contact Number, Email,Address and the calculated Score. "
            "Output must be a JSON array of dictionaries, where each dictionary has the keys: 'Name', 'Contact Number', 'Email', 'Address','pdf link or name' and 'Score' "


        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{context}\n{input}"),
            ]
        )

        # Create question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        try:
            # Button for scoring resumes
            response = rag_chain.invoke({
                "input": "Evaluate these resumes and provide scores scores should be very accurate and strictly evaluated as it would be used by recruiter, as given by system prompt ,"
                         " just provide the json data only not anything else and make sure to be consistent with the output and generate text only. Output must be a JSON array of dictionaries in text format, "
                         "where each dictionary has the keys: 'Name', 'Contact Number', 'Email','Address and 'Score' .Just provide the json data nothing else."
                         "Also the generated data should be equal to uploaded resumes not more nor less"
            })



            json_data = json.loads(response["answer"])

            # Append the JSON data to the all_json_data list

            all_json_data.extend(json_data)

            #st.write(json_data)

        except Exception as e:
            st.error(f"Error: {e}")

    # Once all batches are processed, you can use all_json_data as needed
    # For example, converting it into a DataFrame and displaying
    df = pd.DataFrame(all_json_data)
    st.dataframe(
        df.style
        # Highlight min values
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
            # Table header style
            {'selector': 'tbody td', 'props': [('border', '1px solid #ddd'), ('padding', '8px')]}  # Table body style
        ])
    )


# Call the batch processing function if files are uploaded
if uploaded_files:
    process_pdfs_in_batches(uploaded_files)


