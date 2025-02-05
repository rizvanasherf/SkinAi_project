"""
SkinAI Application 
---------------------
This application utilizes Tensorflow and hugging Face model for image
classification and question answering task realted to skin conditions.

Feature:
- Load a trained model for image classification.
- Uses Faiss for vector store and search.
- Implements a custom prompt for handling user queries

Modules:
- Tensorflow for deep learning.
- Streamlit for UI.
- FAISS for effiective similarity search.
- Langchain for retrieval_base QA.

Usage:
- Upload an image for skin condition detection.
- Input a query for text-based response. 
"""

import os
import time
import boto3
import logging
import numpy as np
import tensorflow as tf
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from logging.handlers import RotatingFileHandler




#  logging configuration 
if 'logger_initialized' not in st.session_state:
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Remove any existing handlers from the root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Configure logging with both file and stream handlers
    logger = logging.getLogger('SkinAI')
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler (RotatingFileHandler)
    file_handler = RotatingFileHandler(
        'logs/skinAi_app.log',
        maxBytes=5*1024*1024,  # 5 MB
        backupCount=3,
        mode='a'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Stream Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent duplicate logging in Streamlit
    logger.propagate = False

    # Mark logger as initialized in session state
    st.session_state.logger_initialized = True
    
    # Log initial startup only once
    logger.info("SkinAI Application started.")
else:
    # Get existing logger
    logger = logging.getLogger('SkinAI')

# Function to get logger (use this in other parts of your code)
def get_logger():
    return logging.getLogger('SkinAI')



# Load environment veriables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_PATH = os.getenv("MODEL_PATH")
huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = os.getenv("DB_FAISS_PATH")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_MODEL_KEY = os.getenv("S3_MODEL_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY =os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")

# Validate environment variables
if not all([HF_TOKEN, MODEL_PATH, DB_FAISS_PATH]):
    logger.error("Missing required environment variables.")
    raise ValueError("Missing required environment variables.")

# CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: black !important;
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }
        .stApp {
            background-color: black !important;
        }
        .status-text {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #22662b;
        }
       .title {
            font-size: 5em;
            font-weight: bold;
            letter-spacing: 10px;
            background: linear-gradient(to right, #05a908, #000000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);  
            margin-bottom: 20px;
            animation: title-animation 3s infinite;  
        }
        .intro-text {
            font-size: 1.3em;
            line-height: 2;
            margin: 20px auto;
            padding: 0 15px;
            max-width: 850px;
            color:rgb(133, 133, 133);
            text-align: center;
        }
        .ask-skinai-title {
            font-size: 2em;  /* Smaller font size for ASK SkinAIbot */
            font-weight: bold;
            background: linear-gradient(to right, #05a908, #000000);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);  
            margin-bottom: 10px;
        }
        .intro_bot{
            font-size: 2.5em;
            font-weight: bold;
            color:rgb(133, 133, 133);
            text-align: center;
        }
        
        
        .stTextInput input {
            background-color:rgb(78, 76, 76);
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
            border-color: #cccccc !important;
            box-shadow: none !important;
        }
        .stWrite {
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
            background-color: #f9f9f9;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .footer {
            text-align: center;
            font-size: 1.1em;
            margin-top: 50px;
            color: #084c24;
            font-weight: bold;
        }

        .footer:hover {
            color: #F5F5F5;
            transition: color 0.4s ease-in-out;
        }

        .stButton > button {
            background: linear-gradient(75deg, #079537 0%, #03330b 100%);  
            color: #ccffcc;
            border: none;
            padding: 12px 30px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 50px;  
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: background 0.3s, transform 0.3s; 
            margin-top: 20px;
            width: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(75deg, #079537 0%, #03330b 100%);  
            transform: scale(1.05);  
            color: #ccffcc;
    
        }

        .stButton > button:focus {
            outline: none;
            box-shadow: 0 0 2px 2px rgba(255, 255, 255, .3);  
            color: inherit; 
        }

        .result-text {
            text-align: center;
            margin-top: 20px;
        }

        .uploaded-text {
            text-align: center;
            color:rgb(0, 0, 0);
            font-weight: bold;
        }
        
     
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize S3 client
s3 = boto3.client('s3')

def download_model_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_PATH):
    """
    Download the model from S3 to the local path.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_model_key (str): Key of the model in the S3 bucket.
        local_model_path (str): Local path to save the downloaded model.

    Raises:
        Exception: If the download fails.
    """
    try:
        logger.info(f"Downloading model from S3: {S3_MODEL_KEY}")
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_PATH)
        logger.info("Model downloaded from S3 successfully.")
    except Exception as e:
        logger.error(f"Failed to download model from S3: {str(e)}")
        raise e


@st.cache_resource(show_spinner=False) 
def load_model():
    """
    Load the Tensorflow model.
    
    This function loads a pre-trained TensorFlow model from the specified path 
    while displaying a progress bar and status updates in a Streamlit app.
    
    Return:
        Keras.Model: A compiled Tensorflow model loaded from the specification path.
        
    Raises:
        Exception: If the model fails to load. 
    """
    logger.info("Loading TensorFlow model...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check if the model exists locally
    LOCAL_MODEL_PATH = os.path.join(os.getcwd(), MODEL_PATH)
    if not os.path.exists(LOCAL_MODEL_PATH):
        logger.info("Model not found locally. Downloading from S3...")
        download_model_from_s3(S3_BUCKET_NAME, S3_MODEL_KEY, MODEL_PATH)
    
    # Display progress bar    
    for percent in range(0, 101, 10): 
        progress_bar.progress(percent)
        status_text.markdown(f"<p class='status-text'>Loading App... {percent}%</p>", unsafe_allow_html=True)
    try:    
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Tensorflow model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Tensorflow model: {str(e)}")
        raise e
    finally:
        progress_bar.empty()
        status_text.empty() 
    return model

@st.cache_resource(show_spinner=False) 
def get_vectorstores():
    """
    Load FAISS vector store.
    
    This fucntion initializes an embedding model using the 'all_miniLM-L6-v2'.
    
    Return:
        FAISS: An instance of the FAISS vector store.
    """
    logger.info("Loading FAISS vector store....")
    try:
        embedding_model =HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        logger.info("FAISS vector store loaded sucessfull.")
        return db
    except Exception as e:
        logger.error(f"Failde to load FAISS vectore store: {str(e)}")
        raise e

def set_custom_prompt(custom_prompt_template):
    """
    Set a custom prompt for the model.
    
    This function create a custome prompt template using the provided
    template string.
    
    Args:
        custom_prompt_template (str): The template string defining
                                        the prompt format.
                                        
    Returns:
        PromptTemplate: A formatted prompt template instance.
    """
    logger.info("Setting custome prompt")
    try:
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
        return prompt
    except Exception as e:
        logger.error(f"Failed to set custome prompt: {str(e)}")

def load_llm(huggingface_repo_id,HF_TOKEN):
    """
    Load the language model from Hugging face.
    
    This function load the Hugging Face llm model
    
    Args:
        huggingface_repo_id (str): The repository id of the Mistral-7B-Instruct-v0.3
        HF_TOKEN (str): Its the huggiface token id
    
    Return:
        HuggingFaceEndpoint: An instance of the loaded model
        
    """
    logger.info("Loading Hugging Face model.....")
    try:
        llm =HuggingFaceEndpoint(repo_id=huggingface_repo_id,temperature=0.5,model_kwargs={"token":HF_TOKEN,"max_length":512})
        logger.info("Hugging Face model loaded sucessfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to load Hugging Fcae model: {str(e)}")

def model_prediction(test_image):
    """
    Performs image classification using the loaded Tensorflow model.
    
    Args:
        test_image (str): Path to the test image.
    
    Returns:
        (predicted class index, Confidence score)
    """
    logger.info("Performing image classification.....")
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(299, 299))
        img_arr = tf.keras.preprocessing.image.img_to_array(image)
        img_arr = np.expand_dims(img_arr, axis=0) / 255.0  # Normalize
        prediction = model.predict(img_arr)
        index_pre = np.argmax(prediction)
        confidence = float(prediction[0][index_pre]) 
        logger.info(f"Prediction successful: Class index {index_pre}, Confidence {confidence}") 
        return index_pre, confidence
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None, None

# def handle_out_of_context():
#         return "Sorry, I couldn't find relevant information. Could you please ask a skin-related question?"
 

# Initialize session state to keep track of the current page
if "page" not in st.session_state:
    st.session_state.page = "home"

# Load models
model = load_model()  
   
 
# Main Page
def home_page():
    """
    Display the main page of the SkinAI application.
    
    The main page consist of:
    - An introduction section that explains the purpose of the app and its mission.
    - A button ("STRAT skin test"), which navigate to skin test page
    - A chat assistance,which allow users to Interact with the AI chat bot
    
    Session States:
    ---------------
    st.session_state.page: Stores the current page state and reroute to the skin testing page
    
    Exception Handling:
    ------------------
    - If vector store retrieval fails, an error message is displayed.
    - If an exception occurs during the response generation, the error is captured and shown.
    
    """
    
    logger.info("Rendering home page...")
    # Title Section
    st.markdown('<p class="title">SkinAI</p>', unsafe_allow_html=True)

    # Introduction Section
    st.markdown("""
    <div class="intro-text">
        SkinAI is an advanced application designed to detect and diagnose a select range of common skin diseases with high accuracy. While we focus on a smaller number of well-researched conditions, SkinAI ensures reliable and science-backed insights, allowing users to quickly identify and understand their skin health concerns. Whether it's a persistent rash, an unusual spot, or another common skin issue, SkinAI provides the guidance you need for informed decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="intro-text">
        Our mission at SkinAI is simple: to provide a reliable and user-friendly skin disease detection tool for a select group of conditions. By focusing on a smaller, manageable number of well-understood diseases, we ensure that every diagnosis is made with precision.
    </div>
    """, unsafe_allow_html=True)
    
    # SKin Test Button
    col1, col2, col3 = st.columns([2,6,2]) 
    with col2:
        if st.button("Start Skin Test"):
            logger.info("User navigated to the skin test page.")
            st.session_state.page = "skin_test"
            st.experimental_rerun()
                 
    st.markdown("""
    <div class="intro-text">
        SkinBot is your intelligent skin care companion, designed to provide expert advice, personalized recommendations, and insights tailored to your skin's unique needs. Whether you're looking for tips on skin care routines, product suggestions, or ways to address specific concerns like acne, dryness, or aging, GlowBot is here to help. With AI-driven analysis and real-time guidance, achieving healthy, glowing skin has never been easier.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="intro_bot">How can I assist you today?</p>', unsafe_allow_html=True)

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    # Input box for user message
    user_input = st.text_input("", placeholder='Message skinAi')

    # When the button is clicked
    if st.button("Send"):
        if user_input:
            logger.info(f"User query: {user_input}")
            # Append user input to conversation history
            st.session_state['conversation'].append(("You", user_input))

            # Custom prompt
            custom_prompt_template = """
                You are an AI assistant answering user questions based on the given context.

                Context:
                {context}

                Question: {question}

                If the context does not contain relevant information, respond with:
                "I'm sorry, but I don't have enough information to answer this question."
                """
            try:
                # Validate user input length
                if len(user_input) > 500:
                    st.warning("Your input is too long. Please keep it under 500 characters.")
                    return
                
                # Attempt to get vectorstores
                vectorstores = get_vectorstores()
                if vectorstores is None:
                    st.error('Failure  to load vector store')
                    
                if user_input:
                    # Initialize the QA chain    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(huggingface_repo_id,HF_TOKEN),
                        retriever=vectorstores.as_retriever(search_kwargs={'k': 2,'similarity_threshold': 0.8}),
                        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},
                         chain_type="stuff"  
                    )
                    
                    # Get the response from QA chain
                    response = qa_chain.invoke({"query": user_input})
                    
                    if "result" in response:
                        result = response['result']

                        # Check for the out-of-context default response
                        if len(result) < 10 or "sorry" in result.lower() or "do not have" in result.lower():
                            st.markdown("<div style=' font-size: 1.3em;padding:10px;border-radius:10px;margin-bottom:10px;': bold;'>I'm sorry,  I couldn't find a relevant answer to your question. Please try rephrasing or asking something else.</div>", unsafe_allow_html=True)
                        else:
                            # Display structured result
                            st.markdown("## SkinAI ")
                            st.markdown(f"<div style='font-size: 1.3em;padding:10px;border-radius:10px;margin-bottom:10px;'>{result}</div>", unsafe_allow_html=True)
                    
            except ValueError as e:
                st.error(f"Error: {str(e)}")
            except KeyError as e:
                st.error("Unexpected response format from the AI chain.")
            except Exception as e:
                logger.error(f"Error processing query: {str(e)}", exc_info=True)
                st.error(f"An unexpected error occurred: {str(e)}")

   
# Skin Test Page
def skin_test_page():
    """
    Skin Test Page: Allow uers to upload a skin image for AI-based diagnosis.
    
    - User upload an image (JPG, JPEG, PNG).
    - The AI model detects potential skin conditions.
    - If a conditio9n is detected, relevented information retrieved using vector store.
    - Provides a discilimer about the conuslting a dermatologist
    
    Return:
        None
    """
    
    logger.info("Rendering skin test page...")
    # Page Header
    st.markdown("<h3 style='text-align: center;'>Upload Your Skin Image for Diagnosis</h3>", unsafe_allow_html=True)
    
    # File Uplodeing Section
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    # Text after successful upload
    if uploaded_file is not None:
        logger.info(f"User uploaded an image: {uploaded_file.name}")
        st.markdown("<div class='uploaded-text'>Uploaded successfully.</div>", unsafe_allow_html=True)
    
    # Detect Button
    if st.button('Detect'):
        
        if uploaded_file is None:
            st.error("Please upload an image before proceeding.")
            return
        
        try:
            with st.spinner("Analyzing image... Please wait."):
                result_index, confidence = model_prediction(uploaded_file)
            
            # Skin Condition Categories
            class_name = [
                'Acne or Rosacea', 'Eczema', 'Fungal infections ', 
                'Seborrheic Keratoses or other Benign Tumors',
                'Warts Molluscum Or Other Viral Infections'
            ]

            if confidence >= 0.5:
                st.markdown(f"""
                    <div class="result-text">
                        <h4>It appears that you might have a condition associated with {class_name[result_index]}</h4>
                        <p>We recommend consulting a dermatologist for an accurate diagnosis and appropriate treatment.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Custom Prompt Template
                custom_prompt_template = """
                    Use the pieces of information provided in the context to give a detailed explanation about the predicted disease. 
                    Ensure the response is a natural, cohesive paragraph without any numbered lists, bullet points, or section headings. 
                    Do not include phrases like '1.', '2.', 'Point 1', or 'Section 1'. The explanation should flow smoothly, like a natural conversation or a medical description. 
                    Only focus on the disease’s description, causes, symptoms, and possible treatments. 
                    Do not provide anything outside of the given context.

                    Context: {context}
                    Question: {question}

                    Start the answer directly in a single, structured paragraph with no numbered points or section divisions.
                    """
                try:
                    vectorstores =get_vectorstores()
                    if vectorstores is None:
                        st.error('Failed to load vector store')
                        return
                    
                    #Load QA Chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(huggingface_repo_id,HF_TOKEN),
                        retriever=vectorstores.as_retriever(search_kwargs={'k': 3}),
                        chain_type_kwargs={'prompt': set_custom_prompt(custom_prompt_template)},
                        chain_type="stuff"  
                    )
                    
                    #Get AI Response
                    with st.spinner("Analyzing the condition... Please wait."):
                        response = qa_chain.invoke({"query": class_name[result_index]})
                        result = response['result'] 
                                         
                    if "sorry" in result.lower() or "do not have" in result.lower():
                        st.markdown("<p>No additional information available at this moment.</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-size: 1em;padding:10px;border-radius:10px;margin-bottom:10px;'>SkinAI: {result}</div>", unsafe_allow_html=True)
                        
                except Exception as e:
                    logger.error(f"Error during skin test: {str(e)}", exc_info=True)
                    st.error(f"Error loading AI response: {str(e)}") 
                    
            else:
                st.markdown(""" 
                    <div class="result-text">
                        <h4>Model couldn't identify the disease</h4>
                        <p>We recommend consulting a dermatologist for a better diagnosis and appropriate treatment.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown("<div style='text-align: center; color: red;'>An error occurred during Detecting. Please upload the image.</div>", unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown("""
            <div class="disclaimer-text" style="margin-top: 20px; font-size: 0.9em; color: #555;">
                <h5><strong>Disclaimer:</strong></h5>
                <p>This application provides an AI-based preliminary analysis of your skin condition. However, the results are not a definitive diagnosis and should not be used as a substitute for professional medical advice.</p>
                <p>We strongly recommend consulting a qualified healthcare professional or dermatologist for an accurate diagnosis and appropriate treatment options. The AI model is designed to assist but may not always be accurate. Always seek a doctor's advice for any concerns regarding your health.</p>
            </div>
        """, unsafe_allow_html=True)

    # Back to Home Button
    col1, col2 = st.columns([8, 3])  
    with col2:
        if st.button("Back to Home"):
            logger.info("User navigated to the home page.")
            st.session_state.page = "home"
            st.experimental_rerun()

# Page routing dictionary
pages = {
    "home": home_page,
    "skin_test": skin_test_page,
}

# Render the appropriate page
if st.session_state.page in pages:
    pages[st.session_state.page]()
else:
    logger.error("Page not found.")
    st.error("Page not found.")

# Footer Section
st.markdown("""
    <div class="footer">
    &copy; 2025 SkinAI. All Rights Reserved. <br>
    Designed with passion by the SkinAI Team.
    </div>
    """, unsafe_allow_html=True)

