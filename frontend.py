import requests
import streamlit as st
import logging  # Import logging module

st.set_page_config(page_title="Text Simplification App", layout="centered")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Ensure logs are sent to the terminal
)

logging.info("Initial")

st.markdown(
    """
    <style>
    .main {
        text-align: right;
        font-size: 20px;
        background-color: #f0f0f5;
        color: #333;
    }
    .stTextInput, .stTextArea {
        text-align: right;
        font-size: 18px;
    }
    .stButton button {
        font-size: 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("פישוט לשוני")

api_key = st.text_input("Enter your Azure OpenAI API Key", type="password")
text_input = st.text_area("הכניסו כאן את הטקסט המסובך")

if st.button("Simplify Text"):
    if api_key and text_input:
        logging.info("Simplify Text button clicked.")
        logging.debug(f"API Key provided: {'Yes' if api_key else 'No'}")
        logging.debug(f"Text input length: {len(text_input)} characters")

        data = {
            "text_input": text_input,
            "api_key": api_key  # Include the API key in the request payload
        }
        logging.debug(f"Payload being sent to backend: {data}")  # Debug log for payload
        try:
            logging.info("Sending request to backend...")
            response = requests.post(
                "https://pashutlinux.azurewebsites.net/api/endpoint",  # Updated to match the deployed backend URL
                json=data,
                timeout=10  # Add a timeout of 10 seconds
            )
            logging.debug(f"Response status code: {response.status_code}")
            logging.debug(f"Response content: {response.text}")  # Debug log for response content
            if response.status_code == 200:
                result = response.json().get("result", "No response from backend")
                logging.info("Received successful response from backend.")
                st.write("Simplified Text:")
                st.write(result)
            else:
                logging.error(f"Error from backend: {response.text}")
                st.error("Error from backend. Please try again later.")
        except requests.exceptions.ConnectionError:
            logging.error("Failed to connect to the backend. Is it running?")
            st.error("Failed to connect to the backend. Please ensure the backend is running.")
        except requests.exceptions.Timeout:
            logging.error("The request to the backend timed out.")
            st.error("The request to the backend timed out. Please try again later.")
        except Exception as e:
            logging.exception(f"An unexpected error occurred: {e}")
            st.error("An unexpected error occurred. Please try again later.")
    else:
        logging.warning("API key or text input missing.")
        st.error("Please provide both API key and text input")

if __name__ == "__main__":
    # Run Streamlit on a specific port
    st.run(port=8000)
