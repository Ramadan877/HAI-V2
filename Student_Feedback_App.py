from flask import Flask, request, render_template, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import openai
import os
from gtts import gTTS
import whisper  
from pydub import AudioSegment
import json
from tempfile import NamedTemporaryFile

openai.api_key = "OPENAI_API_KEY"

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads/'
USER_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'user_audio')
AI_AUDIO_FOLDER = os.path.join(UPLOAD_FOLDER, 'ai_audio')
ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg', 'gif', 'pdf', 'mp4', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['USER_AUDIO_FOLDER'] = USER_AUDIO_FOLDER
app.config['AI_AUDIO_FOLDER'] = AI_AUDIO_FOLDER

os.makedirs(USER_AUDIO_FOLDER, exist_ok=True)
os.makedirs(AI_AUDIO_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = whisper.load_model("small")

def speech_to_text(audio_file_path):
    """Convert audio to text using Whisper."""
    result = model.transcribe(audio_file_path)
    return result["text"]

def generate_audio(text, file_path):
    """Generate speech (audio) from the provided text using gTTS."""
    tts = gTTS(text=text, lang='en')
    tts.save(file_path)
    if os.path.exists(file_path):
        print(f"Audio file successfully saved: {file_path}")
    else:
        print(f"Failed to save audio file: {file_path}")

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/resources/<path:filename>')
def download_resource(filename):
    """Serve resources like PDF or video files from the resources folder."""
    return send_from_directory('resources', filename)

# Setting the context and the golden answer 
def load_concepts():
    """Load concepts from a JSON file."""
    with open("concepts.json", "r") as file:
        return json.load(file)["concepts"]

@app.route('/set_context', methods=['POST'])
def set_context():
    """Set the context for a specific concept from the provided material."""
    concept_name = request.form.get('concept_name')  # Get the selected concept name
    concepts = load_concepts()
    
    # Find the selected concept
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
    intro_text = "Hello my friend, let us begin the self-explanation journey, just go through each concept of the following Natural Language Processing concepts, and try explaining what you understood from each concept in your own words! and don't forget to go easy on yourself and take your time! have fun"
    intro_audio_filename = 'intro_message.mp3'
    
    intro_audio_path = os.path.join(app.config['AI_AUDIO_FOLDER'], intro_audio_filename)

    os.makedirs(app.config['AI_AUDIO_FOLDER'], exist_ok=True)

    if os.path.exists(intro_audio_path):
        os.remove(intro_audio_path)
        print("Old intro audio file removed.")

    generate_audio(intro_text, intro_audio_path)
    
    if os.path.exists(intro_audio_path):
        print(f"Intro audio generated at: {intro_audio_path}")
    else:
        print("Error: Intro audio not generated.")

    intro_audio_url = f"/uploads/ai_audio/{intro_audio_filename}"

    return jsonify({'intro_audio_url': intro_audio_url})



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
    Resource Type: {resource_type}
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
            model="gpt-4-turbo",
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

def compare_explanation(user_message, golden_answer):
    """Compare user explanation with the golden answer and provide qualitative feedback."""
    if user_message.lower() in golden_answer.lower():
        return "very accurate"
    elif some_partial_match(user_message, golden_answer): 
        return "partially correct"
    else:
        return "not accurate"

def some_partial_match(user_message, golden_answer):
    """Placeholder for partial matching logic."""
    return False

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


