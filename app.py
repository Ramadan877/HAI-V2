from flask import Flask, request, render_template, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import openai
import os
from gtts import gTTS
import whisper
import json

openai.api_key = "OPENAI-API-KEY"

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads/'
USER_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'user_audio')
AI_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'ai_audio')
CONCEPT_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'concept_audio')
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['USER_AUDIO_FOLDER'] = USER_AUDIO_FOLDER
app.config['AI_AUDIO_FOLDER'] = AI_AUDIO_FOLDER
app.config['CONCEPT_AUDIO_FOLDER'] = CONCEPT_AUDIO_FOLDER

os.makedirs(USER_AUDIO_FOLDER, exist_ok=True)
os.makedirs(AI_AUDIO_FOLDER, exist_ok=True)
os.makedirs(CONCEPT_AUDIO_FOLDER, exist_ok=True)

STATIC_FOLDER = 'static'
os.makedirs(STATIC_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

whisper_model = whisper.load_model("small")

def speech_to_text(audio_file_path):
    """Convert audio to text using OpenAI Whisper API or local fallback."""
    try:
        with open(audio_file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"Error using OpenAI Whisper API: {str(e)}")
        print("Falling back to local Whisper model...")
        
        try:
            result = whisper_model.transcribe(audio_file_path)
            return result["text"]
        except Exception as e2:
            print(f"Error using local Whisper model: {str(e2)}")
            return "Sorry, I couldn't understand the audio."

def generate_audio(text, file_path):
    """Generate speech (audio) from the provided text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(file_path)
        if os.path.exists(file_path):
            print(f"Audio file successfully saved: {file_path}")
            return True
        else:
            print(f"Failed to save audio file: {file_path}")
            return False
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/resources/<path:filename>')
def download_resource(filename):
    """Serve resources like PDF or video files from the resources folder."""
    return send_from_directory('resources', filename)

def load_concepts():
    """Load concepts from a JSON file or create if it doesn't exist."""
    try:
        with open("concepts.json", "r") as file:
            return json.load(file)["concepts"]
    except (FileNotFoundError, json.JSONDecodeError):
        concepts = [
            {
                "name": "Text Extraction and Cleanup",
                "description": "The process of extracting raw text from various file formats and cleaning it for NLP tasks.",
                "golden_answer": "Text extraction involves pulling raw text from different file formats like PDF, HTML, and Word. Cleanup involves removing unwanted characters, handling special symbols, and preparing the text for further processing."
            },
            {
                "name": "Character Encoding and Unicode Normalization",
                "description": "The process of handling different text encodings and normalizing Unicode representations.",
                "golden_answer": "Character encoding deals with how computers store and represent text characters. Unicode normalization ensures text is represented consistently by converting equivalent character sequences to a standardized form."
            },
            {
                "name": "Spelling Correction",
                "description": "The process of identifying and correcting spelling errors in text data.",
                "golden_answer": "Spelling correction identifies and fixes spelling errors using techniques like dictionary lookups, edit distance calculations, and context-aware algorithms to improve text quality."
            },
            {
                "name": "Tokenization",
                "description": "The process of breaking text into individual words, phrases, symbols, or other meaningful elements.",
                "golden_answer": "Tokenization splits text into smaller units called tokens, which can be words, phrases, or subwords. It's a fundamental step that prepares text for further processing in NLP pipelines."
            },
            {
                "name": "Stop Word Removal",
                "description": "The process of removing common words that add little meaning to text analysis.",
                "golden_answer": "Stop word removal filters out common words like 'the', 'is', and 'and' that occur frequently but carry minimal semantic value, helping to focus analysis on more meaningful content."
            },
            {
                "name": "Lowercasing, Punctuation, and Digit Removal",
                "description": "Text normalization techniques that convert text to lowercase and remove punctuation and numbers.",
                "golden_answer": "These normalization steps convert all text to lowercase for consistency, remove punctuation marks, and filter out digits when they're not relevant to the analysis, creating a cleaner text representation."
            },
            {
                "name": "Stemming and Lemmatization",
                "description": "Techniques to reduce words to their root or base forms.",
                "golden_answer": "Stemming is a simple process that cuts off word endings to get approximate root forms. Lemmatization is more sophisticated, using vocabulary and morphological analysis to return proper dictionary base forms of words."
            },
            {
                "name": "Text Normalization",
                "description": "The process of converting text into a standard canonical form.",
                "golden_answer": "Text normalization is a collection of techniques that transform text into a consistent, standardized format by handling case, punctuation, special characters, and word forms to ensure consistent processing."
            },
            {
                "name": "Part-of-Speech (POS) Tagging",
                "description": "The process of marking words with their grammatical categories.",
                "golden_answer": "POS tagging assigns grammatical labels (noun, verb, adjective, etc.) to each word in text based on its definition and context, enabling more advanced linguistic analysis and understanding."
            },
            {
                "name": "Named Entity Recognition (NER)",
                "description": "The process of locating and classifying named entities in text.",
                "golden_answer": "NER identifies and categorizes key elements in text such as names of people, organizations, locations, expressions of times, quantities, and more, helping extract structured information from unstructured text."
            },
            {
                "name": "Parsing and Syntactic Analysis",
                "description": "The process of analyzing the grammatical structure of sentences.",
                "golden_answer": "Parsing analyzes sentence structure according to grammar rules, creating parse trees or dependency graphs that show relationships between words and phrases, enabling deeper understanding of text meaning."
            },
            {
                "name": "Coreference Resolution",
                "description": "The task of determining when different expressions refer to the same entity.",
                "golden_answer": "Coreference resolution identifies when different words or phrases refer to the same entity (e.g., linking 'she' to 'Mary'), which is crucial for understanding complete documents and extracting relationships."
            }
        ]
        
        with open("concepts.json", "w") as file:
            json.dump({"concepts": concepts}, file, indent=4)
        
        return concepts

@app.route('/set_context', methods=['POST'])
def set_context():
    """Set the context for a specific concept from the provided material."""
    concept_name = request.form.get('concept_name')  
    concepts = load_concepts()
    
    selected_concept = next((c for c in concepts if c["name"] == concept_name), None)

    if not selected_concept:
        return jsonify({'error': 'Invalid concept selection'})

    session['concept_name'] = selected_concept["name"]
    session['description'] = selected_concept["description"]
    session['golden_answer'] = selected_concept["golden_answer"]
    session['attempt_count'] = 0

    return jsonify({'message': f'Context set for {selected_concept["name"]}.'})

@app.route('/get_intro_audio', methods=['GET'])
def get_intro_audio():
    """Generate the introductory audio message for the chatbot."""
    intro_text = "Hello, let us begin the self-explanation journey, just go through each concept of the following Natural Language Processing concepts, and then click on me to start explaining what you understood from each concept!"
    intro_audio_filename = 'intro_message.mp3'
    
    intro_audio_path = os.path.join(app.config['AI_AUDIO_FOLDER'], intro_audio_filename)

    generate_audio(intro_text, intro_audio_path)
    
    if os.path.exists(intro_audio_path):
        intro_audio_url = f"/uploads/ai_audio/{intro_audio_filename}"
        return jsonify({'intro_audio_url': intro_audio_url})
    else:
        return jsonify({'error': 'Failed to generate introduction audio'}), 500

@app.route('/get_concept_audio/<concept_name>', methods=['GET'])
def get_concept_audio(concept_name):
    """Generate concept introduction audio message."""
    safe_concept = secure_filename(concept_name)
    concept_audio_filename = f'{safe_concept}_intro.mp3'
    
    concept_audio_path = os.path.join(app.config['CONCEPT_AUDIO_FOLDER'], concept_audio_filename)
    
    concept_intro_text = f"Now go through this concept of {concept_name}, and try explaining what you understood from this concept in your own words!"
    
    generate_audio(concept_intro_text, concept_audio_path)
    
    return send_from_directory(app.config['CONCEPT_AUDIO_FOLDER'], concept_audio_filename)

@app.route('/submit_message', methods=['POST'])
def submit_message():
    """Handle the submission of user messages and generate AI responses."""
    user_message = request.form.get('message')
    audio_file = request.files.get('audio')
    concept_name = request.form.get('concept_name')  

    print(f"Received concept from frontend: {concept_name}")  

    if not user_message and not audio_file:
        print("Error: No message or audio received!")  
        return jsonify({'error': 'Message or audio is required.'})

    if not concept_name:
        print("Error: No concept detected!")  
        return jsonify({'error': 'Concept not detected.'})

    concepts = load_concepts()
    selected_concept = next((c for c in concepts if c["name"] == concept_name), None)

    if not selected_concept:
        print("Error: Concept not found in system!")  
        return jsonify({'error': 'Concept not found.'})

    print(f"Using concept: {selected_concept}")  

    if audio_file:
        audio_path = os.path.join(app.config['USER_AUDIO_FOLDER'], 'user_audio.wav')
        audio_file.save(audio_path)
        user_message = speech_to_text(audio_path)

    ai_response = generate_response(
        user_message,
        selected_concept["name"],
        selected_concept["description"],
        selected_concept["golden_answer"],
        session.get('attempt_count', 0)
    )

    if not ai_response:
        print("Error: AI response generation failed!")  
        return jsonify({'error': 'AI response generation failed.'})

    print(f"AI Response: {ai_response}")  

    ai_response_filename = "response_audio.mp3"
    audio_response_path = os.path.join(app.config['AI_AUDIO_FOLDER'], ai_response_filename)
    generate_audio(ai_response, audio_response_path)

    if not os.path.exists(audio_response_path):
        print("Error: AI audio file not created!")  
        return jsonify({'error': 'AI audio generation failed.'})

    ai_audio_url = f"/uploads/ai_audio/{ai_response_filename}"
    print(f"AI Response Audio URL: {ai_audio_url}")  

    return jsonify({
        'response': ai_response,
        'ai_audio_url': ai_audio_url
    })


def generate_response(user_message, original_text, resource_type, golden_answer, attempt_count):
    """Generate a response dynamically using OpenAI GPT."""

    if not golden_answer or not original_text:
        return "As your tutor, I'm not able to provide you with feedback without having context about your explanation. Please ensure the context is set."
    
    base_prompt = f"""
    Context: {original_text}
    Golden Answer: {golden_answer}
    User Explanation: {user_message}
    
     You are a friendly and encouraging tutor, helping a student refine their understanding in a supportive way. Your goal is to evaluate the student's explanation and provide warm, engaging feedback:
     - If the user's explanation is very accurate, celebrate their effort and reinforce their confidence.
     - If the explanation is partially correct, acknowledge their progress and gently guide them toward refining their answer.
     - If it's incorrect, provide constructive and positive feedback without discouraging them. Offer hints and encouragement.
     - Use a conversational tone, making the user feel comfortable and motivated to keep trying.
     - Offer increasingly specific hints or the correct answer after multiple attempts, always keeping a friendly and supportive attitude.
     """

    if attempt_count == 1:
        base_prompt += "\nProvide general feedback and a broad hint to guide the user."
    elif attempt_count == 2:
        base_prompt += "\nProvide more specific feedback and highlight key elements the user missed."
    elif attempt_count == 3:
        base_prompt += "\nProvide the correct explanation, as the user has made multiple attempts."
    else:
        base_prompt += "\nReset and encourage the user to try again with a fresh start."

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful tutor providing feedback to students."},
                {"role": "user", "content": base_prompt}
            ],
            max_tokens=200,
            temperature=0.7,
        )
        ai_response = response['choices'][0]['message']['content']
        return ai_response
    except Exception as e:
        return f"Error generating AI response: {str(e)}"

@app.route('/uploads/<folder>/<filename>')
def serve_audio(folder, filename):
    """Serve the audio files from the uploads folder."""
    print(f"Serving audio from folder: {folder}, file: {filename}")
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
    return send_from_directory(folder_path, filename)

@app.route('/pdf')
def serve_pdf():
    return send_from_directory('resources', '1_NLP_cleaning_and_preprocessing.pdf')

if __name__ == '__main__':
    app.run(debug=True)

