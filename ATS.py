import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


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


st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #a1c4fd, #c2e9fb);
        font-family: 'Roboto', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .stTextInput > label {
        font-weight: bold;
    }
    .stDataFrame {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        text-align: center;
        font-size: 52px;
        font-weight: bold;
        color: #333;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# CSS for background with reduced transparency


# Set up embeddings
HF_TOKEN = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# API keys
api_key = os.getenv('API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API')
os.environ["GOOGLE_API_KEY"] =GEMINI_API_KEY



st.markdown('<div class="title">Recruitment ֎🇦🇮</div>', unsafe_allow_html=True)

llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")
llm_2 = ChatGroq(groq_api_key=api_key, model_name="gemma-7b-it")
llm3 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash"    
)

def create_job_post(job_title, location, exp,salary_range,other_input):
    prompt = (
        f"Create a job opening post for platforms like Internshala, LinkedIn, and Naukri.com. "
        f"The post should include the job title: {job_title}, location: {location}, and required experience: {exp},salart range:{salary_range} and other input:{other_input}. "
        f"Make it attractive, include skills (if possible but the skills in boxes and highlight them) required, who can apply, benefits, and other necessary details. "
        f"The post should be 100-200 words."
    )

    try:
        # Replace 'model.generate_content' with the actual method to generate content
        response = llm_2.invoke(prompt)  # assuming 'model' is defined elsewhere
        return response.content
    except Exception as e:
        return f"Error generating response: {e}"


# Streamlit app title


with st.form("job_post_form"):
    st.write("📝 Fill in the details to create a job post ✍🏼 :")
   

    left, right = st.columns(2)
    with left:
        # Job title, location, and experience input fields
        job_title = st.text_input("Enter the job title you are looking for:")
        location = st.text_input("Enter the location you are looking for:")
    with right:
        exp = st.text_input("Enter the experience you are looking for:")
        salary_range = st.text_input("Enter the Salary range you are looking for:")

    other_input = st.text_input("Enter any other details you want to add in the job post:")

  

    # Form submission button
    submitted = st.form_submit_button("Create Job Post ✅")

    if submitted:
        if job_title and location and exp:
            job_post = create_job_post(job_title, location, exp, salary_range, other_input)
            st.write("### Generated Job Post:")
            st.write(job_post)
        else:
            st.warning("Please fill in all required fields to generate the job post.")

job_post = create_job_post(job_title, location, exp, salary_range,other_input)




# Resume scoring section




with st.form("JD form"):
    st.write("📝 Upload Job Description 📜")
    jd=st.file_uploader("Upload pdf file", type="pdf")
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
        f"Extract the Job description from the uploaded file in proper format. Also keep the value of the Job Location in mind as it will be required Later to match with resumes "
    )

        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{context}\n{input}"),
        ]
    )

    # Create question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm_2,qa_prompt)
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

    if st.form_submit_button("Submit"):
        job_description = get_jd(jd)
    else:
        job_description = None


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
            "Output must be a JSON array of dictionaries, where each dictionary has the keys: 'Name', 'Contact Number', 'Email', 'Address' and 'Score' "
            "Generate JSON data for, but present it as plain text in the response. Do not use a code block or any structured formatting like boxes."
            "Do not include extra line breaks or whitespace outside the JSON array. The JSON must start with [ and end with ]"
            f"if the Address of the candidate is too far (like more than 100 kms) away from {location} or the job location mentioned in the {job_description}, then the score should be 0."
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{context}\n{input}"),
            ]
        )

        # Create question-answering chain
        question_answer_chain = create_stuff_documents_chain(llm3, qa_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        try:
            # Button for scoring resumes
            response = rag_chain.invoke({
                "input": (
                    "You are a smart AI agent tasked with evaluating resumes based on a job description. "
                    "Strictly follow the instructions below to generate the output:\n\n"
                    "1. Extract the following details from each resume: Name, Contact Number, Email, Address, and Score.\n"
                    "2. Output the data as a JSON array of dictionaries. Each dictionary must contain these keys: "
                    "'Name', 'Contact Number', 'Email', 'Address', and 'Score'.\n"
                    "3. The output must be plain text JSON without any extra formatting, syntax highlighting, or visual artifacts. "
                    "Do not include explanations, metadata, or anything other than the JSON array.\n"
                    "4. Ensure the JSON is well-formed and matches the number of resumes exactly.\n\n"
                    "Generate only the JSON array in text format, ensuring it adheres to this structure."
                    "Generate JSON data for, but present it as plain text in the response. Do not use a code block or any structured formatting like boxes."
                    f"if the Address of the candidate is too far (like more than 100 kms) away from {location} or the job location mentioned in the {job_description}, then the score should be 0.but dont include comparison in the output"
                )
            })


            json_data = json.loads(response["answer"])

            # Append the JSON data to the all_json_data list

            all_json_data.extend(json_data)
            sorted_data = sorted(all_json_data, key=lambda x: x["Score"], reverse=True)

            #st.write(json_data)

        except Exception as e:
            st.error(f"Error: {e}")

    # Once all batches are processed, you can use all_json_data as needed
    # For example, converting it into a DataFrame and displaying
    df = pd.DataFrame(sorted_data)

    st.dataframe(
        df.style
        # Highlight min values
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]},
            # Table header style
            {'selector': 'tbody td', 'props': [('border', '1px solid #ddd'), ('padding', '8px')]}  # Table body style
        ])
    )

with st.form("resume_form"):
    st.write("📂 Upload the resumes to score them 📑:")

    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.form_submit_button("Score Resumes✔️"):
        if uploaded_files:
            process_pdfs_in_batches(uploaded_files)
        else:
            st.warning("Please upload files to score.")









