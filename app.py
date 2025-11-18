from flask import Flask, request, jsonify, render_template, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from flask_cors import CORS
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Flask App Initialization
# ------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

# ------------------------------
# Config
# ------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "supersecretkey")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=1)

# Allow local dev origin (adjust as needed)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://127.0.0.1:5000"]}})

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# ------------------------------
# User Model
# ------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

with app.app_context():
    db.create_all()

# ------------------------------
# Emotion model load (with fallback)
# ------------------------------
emotion_art = {
    "happy": "happy.png",
    "sadness": "sad.png",
    "anger": "angry.png",
    "fear": "fear.png",
    "love": "love.png",
    "surprise": "surprise.png",
    "disgust": "disgust.png",
    "neutral": "calm.png"
}

# Try to load HF pipeline; if it fails, fall back to None and use simple heuristic
try:
    from transformers import pipeline
    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base"
    )
    app.logger.info("Loaded HuggingFace emotion model.")
except Exception as e:
    emotion_model = None
    app.logger.warning("Could not load HF model. Using demo heuristic. Error: %s", e)

def demo_guess(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ["happy","joy","glad","excited","yay"]): return "happy"
    if any(w in t for w in ["sad","upset","tear","cry","depress"]): return "sadness"
    if any(w in t for w in ["angry","mad","furious","hate"]): return "anger"
    if any(w in t for w in ["fear","scared","afraid","nervous"]): return "fear"
    if any(w in t for w in ["love","romantic","heart"]): return "love"
    if any(w in t for w in ["surprise","wow","shocked","omg"]): return "surprise"
    if any(w in t for w in ["disgust","gross","yuck"]): return "disgust"
    return "neutral"

# ------------------------------
# Page routes (serve templates)
# ------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/auth.html")
def auth_page():
    return render_template("auth.html")

@app.route("/chat.html")
def chat_page():
    return render_template("chat.html")

# ------------------------------
# Auth API: Signup / Login / Me
# ------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email & password are required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already exists"}), 409
    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Account created successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email & password required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "User not found"}), 404
    if not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "Invalid password"}), 401
    token = create_access_token(identity=user.id)
    return jsonify({"token": token, "message": "Login successful"}), 200

@app.route("/me", methods=["GET"])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": user.id, "email": user.email}), 200

# ------------------------------
# Analyze: protected and demo
# ------------------------------
@app.route("/analyze_text", methods=["POST"])
@jwt_required()
def analyze_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400
    text = data.get("text")
    if not text or not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string"}), 400

    # run model safely
    try:
        if emotion_model:
            # emotion_model returns list of dicts
            result = emotion_model(text)[0]
            emotion = (result.get("label") or "").lower()
        else:
            emotion = demo_guess(text)
    except Exception as e:
        app.logger.exception("Emotion model error")
        return jsonify({"error": "Emotion model failed", "detail": str(e)}), 500

    art_file = emotion_art.get(emotion, emotion_art.get("neutral"))
    art_url = url_for("static", filename=f"art/{art_file}")
    return jsonify({"emotion": emotion, "art_url": art_url}), 200

# Unprotected demo endpoint for development/testing
@app.route("/analyze_demo", methods=["POST"])
def analyze_demo():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400
    emotion = demo_guess(text)
    art_file = emotion_art.get(emotion, emotion_art.get("neutral"))
    art_url = url_for("static", filename=f"art/{art_file}")
    return jsonify({"emotion": emotion, "art_url": art_url}), 200

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    os.makedirs("static/art", exist_ok=True)
    app.run(debug=True)
