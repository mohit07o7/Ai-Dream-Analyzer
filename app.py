import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
import gtts
import tempfile

# Configure Gemini API
genai.configure(api_key="AIzaSyDk7XztYJ9YjkfNBlK89k7Wy5urk_RcbMw")
model = genai.GenerativeModel("gemini-1.5-pro")
chat_session = model.start_chat()

# Function to record voice and convert to text
def record_and_recognize():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand your speech."
    except sr.RequestError:
        return "Speech recognition service is unavailable."

# Function to convert text to speech
def text_to_speech(text):
    tts = gtts.gTTS(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Function to chat with Gemini
def chat_with_gemini(user_input):
    try:
        response = chat_session.send_message(user_input)
        return response.text if response and response.text else "⚠️ No response received from Gemini."
    except Exception as e:
        return f"🚨 Error: {str(e)}"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for dark theme and background image
st.markdown(
    """
    <style>
    .stApp {
    background-image: url("image.jpg");
    background-size: cover;
    background-position: center;
}

    .stTextArea textarea {
        background-color: rgba(45, 45, 45, 0.8) !important;
        color: #f0f0f0 !important;
    
        border: 1px solid #444 !important;
    }
    .stButton button {
        background-color: #4a90e2 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 5px !important;
        font-size: 16px !important;
    }
    .stButton button:hover {
        background-color: #357abd !important;
    }
    .stSuccess {
        background-color: rgba(46, 125, 50, 0.8) !important;
        color: white !important;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: rgba(211, 47, 47, 0.8) !important;
        color: white !important;
        padding: 10px;
        border-radius: 5px;
    }
    .stWarning {
        background-color: rgba(255, 160, 0, 0.8) !important;
        color: white !important;
        padding: 10px;
        border-radius: 5px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #f0f0f0 !important;
    }
    .stMarkdown strong {
        color: #4a90e2 !important;
    }
    .chat-history {
        background-color: rgba(45, 45, 45, 0.8);
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .chat-history p {
        margin: 0;
        padding: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit UI
st.title("AI Dream Analyzer")

# User input area
user_input = st.text_area(
    "Reveal Your Nightmares :",
    placeholder="Describe your dream or ask a question...",
    height=150,
)

# 🎙️ Voice input button
if st.button("🎙️ Speak Your Dream"):
    user_input = record_and_recognize()
    st.write("📝 You said:", user_input)

# Chat Button
if st.button("Chat here", key="chat_button") or user_input:
    user_input = user_input.strip()  # Remove extra spaces

    if user_input:  # Ensure input is not empty
        with st.spinner("🤖 Analyzing your dream... Please wait."):
            response = chat_with_gemini(user_input)

        if response.startswith(" Error"):
            st.error(response)  # Show error message
        else:
            st.success("Therapist's Response:")
            st.write(response)

            # 🔊 Convert response to speech
            audio_file = text_to_speech(response)
            st.audio(audio_file)

            # Store chat history
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Therapist", response))

    else:
        st.warning(" Please enter a message or try speaking again.")

# Display chat history
st.write("###  Chat History")
chat_history_container = st.container()
with chat_history_container:
    for role, text in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-history">
                <p><strong>{role}:</strong> {text}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


