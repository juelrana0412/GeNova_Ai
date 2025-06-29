from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
import mysql.connector
import os
from dotenv import load_dotenv
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr

from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import Pinecone as LangchainPinecone

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Flask setup
app = Flask(__name__)
app.secret_key = "supersecretkey"
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)

# MySQL DB connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="genova_ai"
    )

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class
class User(UserMixin):
    def __init__(self, id, full_name, email):
        self.id = str(id)
        self.full_name = full_name
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(id=user['id'], full_name=user['full_name'], email=user['email'])
    return None

# Pinecone & LangChain
pc = PineconeClient(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("genova")
pinecone_index_doctor = pc.Index("doctor")
pinecone_index_medicine = pc.Index("medicine")
embeddings = download_hugging_face_embeddings()
docsearch = LangchainPinecone(index=pinecone_index, embedding=embeddings, text_key="text")
docsearch_d = LangchainPinecone(index=pinecone_index_doctor, embedding=embeddings, text_key="text")
docsearch_m = LangchainPinecone(index=pinecone_index_medicine, embedding=embeddings, text_key="text")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n{context}"
)
PROMPT = PromptTemplate(template=system_prompt, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

llm = ChatGroq(model="llama3-8b-8192", temperature=0, max_tokens=1024)

#-----disease------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)
#----Doctor-----
qa_d = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch_d.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

#------Medicine----
qa_m = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch_m.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Chat log helper - save chat history in MySQL DB
def log_chat(user_id, question, answer):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, question, response, timestamp) VALUES (%s, %s, %s, NOW())",
        (user_id, question, answer)
    )
    conn.commit()
    cursor.close()
    conn.close()

#---------------Routes-------------------

#------------Resgister router-----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)", (full_name, email, password))
            conn.commit()
            return redirect(url_for("login"))
        except mysql.connector.IntegrityError:
            return "User already exists!"
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            user_obj = User(id=user["id"], full_name=user["full_name"], email=user["email"])
            login_user(user_obj)
            return redirect(url_for("index"))
        return "Invalid credentials", 401
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/")
@login_required
def index():
    return render_template("chat.html", full_name=current_user.full_name)

@app.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form["msg"]
    try:
        result = qa.invoke({"query": msg})
        response_text = result["result"]
        log_chat(current_user.id, msg, response_text)  # Use user_id for logging

        audio_path = os.path.join(STATIC_FOLDER, "response.mp3")
        gTTS(response_text).save(audio_path)
        return response_text + '<br><audio controls src="/static/response.mp3"></audio>'
    except Exception as e:
        return "Error: " + str(e)

@app.route("/voice", methods=["POST"])
@login_required
def voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=5)
    try:
        return recognizer.recognize_google(audio)
    except Exception as e:
        return "Voice Error: " + str(e)

@app.route("/doctor", methods=["GET", "POST"])
@login_required
def doctor():
    if request.method == "POST":
        query = request.form["doctor_query"]
        try:
            result = qa_d.invoke({"query": query})
            answer = result["result"]
            log_chat(current_user.id, f"[Doctor] {query}", answer)
            gTTS(answer).save(os.path.join(STATIC_FOLDER, "doctor_response.mp3"))
            return render_template("doctor.html", question=query, answer=answer, audio=True)
        except Exception as e:
            return render_template("doctor.html", error=str(e))
    return render_template("doctor.html")

@app.route("/medicine", methods=["GET", "POST"])
@login_required
def medicine():
    if request.method == "POST":
        query = request.form["symptom_query"]
        try:
            result = qa_m.invoke({"query": f"What medicine can be suggested for: {query}"})
            answer = result["result"]
            log_chat(current_user.id, f"[Medicine] {query}", answer)
            gTTS(answer).save(os.path.join(STATIC_FOLDER, "medicine_response.mp3"))
            return render_template("medicine.html", question=query, answer=answer, audio=True)
        except Exception as e:
            return render_template("medicine.html", error=str(e))
    return render_template("medicine.html")

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", full_name=current_user.full_name, email=current_user.email)

@app.route("/history")
@login_required
def history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT question, response, timestamp FROM chat_history WHERE user_id = %s ORDER BY timestamp DESC",
        (current_user.id,)
    )
    chats = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template("history.html", chats=chats)

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
