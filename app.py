import calendar
import json
import os
import sqlite3
import base64
import hashlib
import hmac
import re
import secrets
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    from tensorflow.keras.models import load_model
    MODEL_BACKEND_ERROR = None
except Exception as tf_exc:
    try:
        from keras.models import load_model
        MODEL_BACKEND_ERROR = None
    except Exception as keras_exc:
        load_model = None
        MODEL_BACKEND_ERROR = (
            "Model backend not available. Install TensorFlow in this environment. "
            f"tensorflow import error: {tf_exc}; keras import error: {keras_exc}"
        )

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# -------------------------------------------------------------------
# Database setup
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "mood_tracker.db")
PASSWORD_ITERATIONS = 120_000
REMEMBER_COOKIE_NAME = "ftf_remember_token"
REMEMBER_SESSION_DAYS = 30
PH_TIMEZONE = timezone(timedelta(hours=8))
SQLITE_PH_OFFSET = "+8 hours"


def now_ph() -> datetime:
    return datetime.now(PH_TIMEZONE)


def ph_date_range(days: int):
    end_date = now_ph().date()
    start_date = end_date - timedelta(days=max(1, days) - 1)
    return start_date.isoformat(), end_date.isoformat()


def init_db():
    """Create data directory and database tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    users_exists = cur.fetchone() is not None
    if users_exists:
        cur.execute("PRAGMA table_info(users)")
        columns = {row[1] for row in cur.fetchall()}
        required_user_columns = {"id", "username", "nickname", "password_hash", "created_at"}
        if not required_user_columns.issubset(columns):
            # Reset legacy auth schema for a clean deployment auth rollout.
            cur.execute("DROP TABLE IF EXISTS reflections")
            cur.execute("DROP TABLE IF EXISTS mood_logs")
            cur.execute("DROP TABLE IF EXISTS users")
            conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            nickname TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            intensity INTEGER DEFAULT 3 CHECK (intensity >= 1 AND intensity <= 5),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reflections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            mood_log_id INTEGER NOT NULL,
            mood TEXT NOT NULL,
            answers TEXT,
            free_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (mood_log_id) REFERENCES mood_logs(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    cur.execute("PRAGMA table_info(mood_logs)")
    cols = [row[1] for row in cur.fetchall()]
    if "intensity" not in cols:
        cur.execute("ALTER TABLE mood_logs ADD COLUMN intensity INTEGER DEFAULT 3")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mood_logs_user ON mood_logs(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_mood_logs_created ON mood_logs(created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reflections_user ON reflections(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_reflections_mood_log ON reflections(mood_log_id)")
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_user ON auth_sessions(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires ON auth_sessions(expires_at)")
    cur.execute("PRAGMA table_info(reflections)")
    ref_cols = [row[1] for row in cur.fetchall()]
    if "improvement_rating" not in ref_cols:
        cur.execute("ALTER TABLE reflections ADD COLUMN improvement_rating INTEGER DEFAULT NULL")
    conn.commit()
    conn.close()


def get_conn():
    return sqlite3.connect(DB_PATH)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def validate_username(username: str) -> Optional[str]:
    if not username:
        return "Username is required."
    if not re.fullmatch(r"[a-z0-9_]{3,30}", username):
        return "Username must be 3-30 chars and use only letters, numbers, or underscores."
    return None


def validate_password(password: str) -> Optional[str]:
    if not password:
        return "Password is required."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(ch.isalpha() for ch in password) or not any(ch.isdigit() for ch in password):
        return "Password must include at least one letter and one number."
    return None


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return (
        f"{PASSWORD_ITERATIONS}$"
        f"{base64.b64encode(salt).decode('utf-8')}$"
        f"{base64.b64encode(digest).decode('utf-8')}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        iterations_str, salt_b64, digest_b64 = password_hash.split("$")
        iterations = int(iterations_str)
        salt = base64.b64decode(salt_b64.encode("utf-8"))
        expected_digest = base64.b64decode(digest_b64.encode("utf-8"))
    except (ValueError, TypeError):
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(actual_digest, expected_digest)


def create_user(username: str, nickname: str, password: str):
    """Create a user account. Returns (user_id, cleaned_nickname, error_message)."""
    clean_username = normalize_username(username)
    clean_nickname = nickname.strip()

    username_error = validate_username(clean_username)
    if username_error:
        return None, None, username_error
    if not clean_nickname:
        return None, None, "Nickname is required."
    password_error = validate_password(password)
    if password_error:
        return None, None, password_error

    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, nickname, password_hash) VALUES (?, ?, ?)",
            (clean_username, clean_nickname, hash_password(password)),
        )
        user_id = cur.lastrowid
        conn.commit()
        return user_id, clean_nickname, None
    except sqlite3.IntegrityError:
        return None, None, "That username is already taken."
    finally:
        conn.close()


def authenticate_user(username: str, password: str):
    """Authenticate by username/password. Returns (id, nickname, username) or None."""
    clean_username = normalize_username(username)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, nickname, password_hash FROM users WHERE username = ?",
        (clean_username,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    user_id, nickname, password_hash = row
    if not verify_password(password, password_hash):
        return None
    return user_id, nickname, clean_username


def _hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_auth_session(user_id: int, days_valid: int = REMEMBER_SESSION_DAYS) -> str:
    raw_token = secrets.token_urlsafe(48)
    token_hash = _hash_session_token(raw_token)
    expires_at = (datetime.utcnow() + timedelta(days=days_valid)).isoformat()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO auth_sessions (user_id, token_hash, expires_at) VALUES (?, ?, ?)",
        (user_id, token_hash, expires_at),
    )
    conn.commit()
    conn.close()
    return raw_token


def revoke_auth_session(raw_token: str) -> None:
    token_hash = _hash_session_token(raw_token)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
    conn.commit()
    conn.close()


def get_user_by_session_token(raw_token: str):
    token_hash = _hash_session_token(raw_token)
    now_iso = datetime.utcnow().isoformat()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT u.id, u.username, u.nickname
        FROM auth_sessions s
        JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ?
        """,
        (token_hash, now_iso),
    )
    row = cur.fetchone()
    conn.close()
    return row


def set_cookie_js(name: str, value: str, days_valid: int) -> str:
    return f"""
    <script>
        const expires = new Date(Date.now() + ({days_valid} * 24 * 60 * 60 * 1000)).toUTCString();
        document.cookie = "{name}=" + encodeURIComponent("{value}") + "; expires=" + expires + "; path=/; SameSite=Lax";
    </script>
    """


def clear_cookie_js(name: str) -> str:
    return f"""
    <script>
        document.cookie = "{name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax";
    </script>
    """


def log_mood(user_id: int, mood: str, intensity: int = 3) -> int:
    """Log a mood entry for the user with intensity 1-5. Returns mood_log_id."""
    intensity = max(1, min(5, intensity))
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO mood_logs (user_id, mood, intensity) VALUES (?, ?, ?)",
        (user_id, mood, intensity),
    )
    mood_log_id = cur.lastrowid
    conn.commit()
    conn.close()
    return mood_log_id


def get_daily_moods(user_id: int) -> List[tuple]:
    """Get mood counts by hour for today."""
    today_ph = now_ph().date().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, strftime('%H', datetime(created_at, ?)) AS hour, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND date(datetime(created_at, ?)) = ?
        GROUP BY hour, mood
        ORDER BY hour
    """, (SQLITE_PH_OFFSET, user_id, SQLITE_PH_OFFSET, today_ph))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_total_logs(user_id: int) -> int:
    """Get total mood log count for user (all time)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM mood_logs WHERE user_id = ?", (user_id,))
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_logs_today(user_id: int) -> int:
    """Get mood log count for today."""
    today_ph = now_ph().date().isoformat()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM mood_logs WHERE user_id = ? AND date(datetime(created_at, ?)) = ?",
        (user_id, SQLITE_PH_OFFSET, today_ph),
    )
    count = cur.fetchone()[0]
    conn.close()
    return count


def get_dominant_mood(user_id: int, days: int = 7) -> Optional[str]:
    """Get most frequent mood in last N days. Returns None if no data."""
    dist = get_mood_distribution(user_id, days)
    if not dist:
        return None
    return max(dist, key=lambda x: x[1])[0]


def get_weekly_moods(user_id: int, days: int = 7) -> List[tuple]:
    """Get mood counts by day for last N days."""
    start_date, end_date = ph_date_range(days)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, date(datetime(created_at, ?)) AS day, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND date(datetime(created_at, ?)) BETWEEN ? AND ?
        GROUP BY day, mood
        ORDER BY day
    """, (SQLITE_PH_OFFSET, user_id, SQLITE_PH_OFFSET, start_date, end_date))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_mood_distribution(user_id: int, days: int = 7) -> List[tuple]:
    """Get total mood counts over last N days for pie chart."""
    start_date, end_date = ph_date_range(days)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? AND date(datetime(created_at, ?)) BETWEEN ? AND ?
        GROUP BY mood
    """, (user_id, SQLITE_PH_OFFSET, start_date, end_date))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_moods_by_date(user_id: int, date_str: str) -> List[tuple]:
    """Get all mood entries for a specific date."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT mood, intensity, strftime('%H:%M', datetime(created_at, ?)) as time
        FROM mood_logs
        WHERE user_id = ? AND date(datetime(created_at, ?)) = ?
        ORDER BY created_at DESC
    """, (SQLITE_PH_OFFSET, user_id, SQLITE_PH_OFFSET, date_str))
    rows = cur.fetchall()
    conn.close()
    return rows


def get_dominant_moods_by_date(user_id: int, year: int, month: int) -> dict:
    """Get the dominant mood for each date in a month."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT date(datetime(created_at, ?)) as date, mood, COUNT(*) as cnt
        FROM mood_logs
        WHERE user_id = ? 
          AND strftime('%Y', datetime(created_at, ?)) = ?
          AND strftime('%m', datetime(created_at, ?)) = ?
        GROUP BY date(datetime(created_at, ?)), mood
        ORDER BY date, cnt DESC
    """, (
        SQLITE_PH_OFFSET,
        user_id,
        SQLITE_PH_OFFSET,
        str(year),
        SQLITE_PH_OFFSET,
        f"{month:02d}",
        SQLITE_PH_OFFSET,
    ))
    rows = cur.fetchall()
    conn.close()

    dominant_moods = {}
    for date_str, mood, _ in rows:
        if date_str not in dominant_moods:
            dominant_moods[date_str] = mood
    return dominant_moods


def save_reflection(user_id: int, mood_log_id: int, mood: str, answers: str, free_text: str) -> int:
    """Save a journal/reflection entry linked to a mood log. Returns reflection_id."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO reflections (user_id, mood_log_id, mood, answers, free_text)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, mood_log_id, mood, answers, free_text),
    )
    reflection_id = cur.lastrowid
    conn.commit()
    conn.close()
    return reflection_id


def update_reflection_rating(reflection_id: int, rating: int) -> bool:
    """Update the improvement_rating for a reflection."""
    rating = max(1, min(5, rating))
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE reflections SET improvement_rating = ? WHERE id = ?",
        (rating, reflection_id),
    )
    conn.commit()
    conn.close()
    return True


def get_user_reflections(user_id: int, limit: int = 50) -> List[tuple]:
    """Get all reflections for a user with mood log info."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT r.id, r.mood, r.answers, r.free_text, r.improvement_rating,
               r.created_at, m.intensity
        FROM reflections r
        LEFT JOIN mood_logs m ON r.mood_log_id = m.id
        WHERE r.user_id = ?
        ORDER BY r.created_at DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows


# -------------------------------------------------------------------
# Keras model + Haar Cascade face detection (7 emotions)
# -------------------------------------------------------------------
LABELS_DICT = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise",
}

EMOTION_EMOJIS = {
    "Angry": "", "Disgust": "", "Fear": "", "Happy": "",
    "Neutral": "", "Sad": "", "Surprise": "",
}

EMOTION_CV_COLORS = {
    "Angry": (0, 0, 255), "Disgust": (0, 128, 0), "Fear": (128, 0, 128),
    "Happy": (0, 200, 0), "Neutral": (200, 200, 0), "Sad": (255, 165, 0),
    "Surprise": (255, 255, 0),
}


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    if load_model is None:
        raise RuntimeError(MODEL_BACKEND_ERROR)
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_file_30epochs.h5")
    return load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_face_detector():
    cascade_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(cascade_path)


def detect_emotions(image_input, model, face_detector):
    """Run Keras CNN emotion detection on an image. Returns (annotated_rgb, results_list)."""
    if isinstance(image_input, Image.Image):
        frame = np.array(image_input)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame = image_input.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 3)

    results = []
    if len(faces) > 0:
        frame_h, frame_w = gray.shape[:2]
        image_cx = frame_w / 2.0
        image_cy = frame_h / 2.0

        # Keep only the face whose center is closest to image center.
        x, y, w, h = min(
            faces,
            key=lambda f: (
                ((f[0] + (f[2] / 2.0)) - image_cx) ** 2
                + ((f[1] + (f[3] / 2.0)) - image_cy) ** 2
            ),
        )
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped, verbose=0)
        label_idx = np.argmax(result, axis=1)[0]
        label = LABELS_DICT[label_idx]
        confidence = float(result[0][label_idx]) * 100
        all_probs = {LABELS_DICT[i]: float(result[0][i]) * 100 for i in range(7)}

        color = EMOTION_CV_COLORS.get(label, (50, 50, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), color, -1)
        cv2.putText(frame, f"{label} ({confidence:.1f}%)", (x + 5, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append(
            {
                "label": label,
                "confidence": confidence,
                "all_probabilities": all_probs,
                "bbox": (x, y, w, h),
            }
        )

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, results


def mirror_image(image_input):
    """Mirror image horizontally for camera-like preview behavior."""
    if isinstance(image_input, Image.Image):
        frame = np.array(image_input)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        mirrored = cv2.flip(frame, 1)
        return Image.fromarray(mirrored)
    return image_input


# -------------------------------------------------------------------
# Emotion-specific content (all 7 emotions)
# -------------------------------------------------------------------
JOURNAL_QUESTIONS = {
    "Happy": [
        "What made you feel happy today?",
        "Who contributed to this positive feeling?",
        "How can you carry this feeling into tomorrow?",
    ],
    "Sad": [
        "What's weighing on your mind right now?",
        "Is there something specific that triggered this feeling?",
        "What's one small thing that might help you feel better?",
    ],
    "Angry": [
        "What triggered your anger?",
        "What would help you feel calmer right now?",
        "Is this something within your control to change?",
    ],
    "Neutral": [
        "What's on your mind today?",
        "Is there something you're looking forward to?",
        "How would you describe your energy level right now?",
    ],
    "Disgust": [
        "What caused this feeling of disgust?",
        "Is this related to something you saw, heard, or experienced?",
        "What would help you feel more at ease right now?",
    ],
    "Fear": [
        "What is making you feel afraid or anxious?",
        "Is this fear tied to something specific or more general?",
        "What is one thing that could help you feel safer right now?",
    ],
    "Surprise": [
        "What surprised you?",
        "Was it a pleasant or unpleasant surprise?",
        "How has this surprise affected your mood overall?",
    ],
}

MOOD_ACTIVITIES = {
    "Happy": [
        ("Share it", "Tell someone about your good mood"),
        ("Journal", "Write down what made you happy"),
        ("Take a photo", "Capture the moment"),
        ("Go for a walk", "Enjoy the outdoors"),
    ],
    "Sad": [
        ("Listen to uplifting music", "Let music lift your spirits"),
        ("Talk to a friend", "Reach out for support"),
        ("Light exercise", "A short walk or stretch"),
        ("Self-care", "Take a relaxing bath or rest"),
    ],
    "Angry": [
        ("Deep breathing", "4-7-8 breath: inhale 4, hold 7, exhale 8"),
        ("Short walk", "Step away and cool down"),
        ("Journal", "Write down what you feel"),
        ("Cool-down exercise", "Stretching or gentle movement"),
    ],
    "Neutral": [
        ("Try something new", "Learn a new skill or hobby"),
        ("Short puzzle", "Crossword, Sudoku, or a game"),
        ("Listen to a podcast", "Explore an interesting topic"),
        ("Stretch", "A few minutes of gentle stretching"),
    ],
    "Disgust": [
        ("Step away", "Remove yourself from the source"),
        ("Fresh air", "Go outside and breathe deeply"),
        ("Mindful breathing", "Focus on slow, deep breaths"),
        ("Talk it out", "Share what bothered you with someone"),
    ],
    "Fear": [
        ("Grounding exercise", "Name 5 things you can see, 4 you can touch..."),
        ("Deep breathing", "Slow breaths to calm your nervous system"),
        ("Talk to someone", "Sharing your fears can lighten the load"),
        ("Positive affirmations", "Remind yourself of your strengths"),
    ],
    "Surprise": [
        ("Reflect on it", "Take a moment to process the surprise"),
        ("Journal", "Write down what happened and how you feel"),
        ("Share the news", "Tell someone about the unexpected event"),
        ("Go with the flow", "Embrace the unexpected and stay curious"),
    ],
}

INTENSITY_LABELS = {
    1: "1 - Very slight", 2: "2 - Slight", 3: "3 - Moderate",
    4: "4 - Strong", 5: "5 - Very strong",
}

MOOD_CARD_COLORS = {
    "Happy": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#92400e", "desc": "#b45309"},
    "Sad": {"bg": "#dbeafe", "border": "#3b82f6", "text": "#1e40af", "desc": "#1d4ed8"},
    "Angry": {"bg": "#fee2e2", "border": "#ef4444", "text": "#991b1b", "desc": "#b91c1c"},
    "Neutral": {"bg": "#f1f5f9", "border": "#64748b", "text": "#334155", "desc": "#475569"},
    "Disgust": {"bg": "#d1fae5", "border": "#10b981", "text": "#065f46", "desc": "#047857"},
    "Fear": {"bg": "#ede9fe", "border": "#8b5cf6", "text": "#5b21b6", "desc": "#6d28d9"},
    "Surprise": {"bg": "#fef9c3", "border": "#eab308", "text": "#854d0e", "desc": "#a16207"},
}

MOOD_CALENDAR_COLORS = {
    "Happy": {"bg": "#fbbf24", "border": "#d97706", "text": "#000000"},
    "Sad": {"bg": "#3b82f6", "border": "#1d4ed8", "text": "#ffffff"},
    "Angry": {"bg": "#ef4444", "border": "#b91c1c", "text": "#ffffff"},
    "Neutral": {"bg": "#9ca3af", "border": "#6b7280", "text": "#000000"},
    "Disgust": {"bg": "#34d399", "border": "#059669", "text": "#000000"},
    "Fear": {"bg": "#a78bfa", "border": "#7c3aed", "text": "#ffffff"},
    "Surprise": {"bg": "#facc15", "border": "#ca8a04", "text": "#000000"},
}

# -------------------------------------------------------------------
# Streamlit page config & DB init
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FACES TO FEELINGS",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "nickname" not in st.session_state:
    st.session_state.nickname = None
if "selected_calendar_date" not in st.session_state:
    st.session_state.selected_calendar_date = None
if "calendar_year" not in st.session_state:
    st.session_state.calendar_year = datetime.now().year
if "calendar_month" not in st.session_state:
    st.session_state.calendar_month = datetime.now().month
if "show_journal" not in st.session_state:
    st.session_state.show_journal = False
if "current_mood_log_id" not in st.session_state:
    st.session_state.current_mood_log_id = None
if "journal_mood" not in st.session_state:
    st.session_state.journal_mood = None
if "show_rating" not in st.session_state:
    st.session_state.show_rating = False
if "pending_reflection_id" not in st.session_state:
    st.session_state.pending_reflection_id = None
if "remember_token" not in st.session_state:
    st.session_state.remember_token = None
if "remember_cookie_to_set" not in st.session_state:
    st.session_state.remember_cookie_to_set = None
if "clear_remember_cookie" not in st.session_state:
    st.session_state.clear_remember_cookie = False
if "scroll_anchor_target" not in st.session_state:
    st.session_state.scroll_anchor_target = None
if "preferred_main_tab" not in st.session_state:
    st.session_state.preferred_main_tab = None


def rerun_to_anchor(anchor_id: str) -> None:
    st.session_state.scroll_anchor_target = anchor_id
    st.rerun()

if st.session_state.remember_cookie_to_set:
    components.html(
        set_cookie_js(
            REMEMBER_COOKIE_NAME,
            st.session_state.remember_cookie_to_set,
            REMEMBER_SESSION_DAYS,
        ),
        height=0,
        scrolling=False,
    )
    st.session_state.remember_cookie_to_set = None

if st.session_state.clear_remember_cookie:
    components.html(clear_cookie_js(REMEMBER_COOKIE_NAME), height=0, scrolling=False)
    st.session_state.clear_remember_cookie = False

if st.session_state.scroll_anchor_target:
    anchor_target = st.session_state.scroll_anchor_target
    components.html(
        f"""
        <script>
            setTimeout(() => {{
                const el = parent.document.getElementById("{anchor_target}");
                if (el) {{
                    el.scrollIntoView({{ behavior: "smooth", block: "start" }});
                }}
            }}, 180);
        </script>
        """,
        height=0,
        scrolling=False,
    )
    st.session_state.scroll_anchor_target = None

if st.session_state.user_id is None:
    remember_token = None
    try:
        remember_token = st.context.cookies.get(REMEMBER_COOKIE_NAME)
    except Exception:
        remember_token = None

    if remember_token:
        remembered_user = get_user_by_session_token(remember_token)
        if remembered_user:
            remembered_user_id, remembered_username, remembered_nickname = remembered_user
            st.session_state.user_id = remembered_user_id
            st.session_state.username = remembered_username
            st.session_state.nickname = remembered_nickname
            st.session_state.remember_token = remember_token
        else:
            st.session_state.clear_remember_cookie = True

# -------------------------------------------------------------------
# Nickname / Login screen
# -------------------------------------------------------------------
if st.session_state.user_id is None:
    quotes_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: transparent;
                overflow: hidden;
                pointer-events: none;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .quote {
                position: absolute;
                bottom: -80px;
                font-size: 1rem;
                font-weight: 500;
                white-space: normal;
                word-break: break-word;
                overflow-wrap: anywhere;
                max-width: min(78vw, 320px);
                padding: 14px 20px;
                background: white;
                border-radius: 25px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                animation: riseUp 14s linear forwards;
            }
            .quote::before, .quote::after {
                content: '';
                position: absolute;
                background: white;
                border-radius: 50%;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
            }
            .quote::before { width: 14px; height: 14px; bottom: -8px; left: 22px; }
            .quote::after { width: 9px; height: 9px; bottom: -16px; left: 18px; }
            @keyframes riseUp {
                0% { bottom: -80px; opacity: 0; }
                10% { opacity: 0.85; }
                60% { opacity: 0.85; }
                100% { bottom: 65%; opacity: 0; }
            }
            @media (max-width: 768px) {
                .quote {
                    font-size: 0.85rem;
                    padding: 10px 14px;
                    border-radius: 18px;
                    max-width: min(88vw, 260px);
                }
                .quote::before { width: 11px; height: 11px; bottom: -6px; left: 16px; }
                .quote::after { width: 7px; height: 7px; bottom: -12px; left: 13px; }
                @keyframes riseUp {
                    0% { bottom: -70px; opacity: 0; }
                    10% { opacity: 0.85; }
                    60% { opacity: 0.85; }
                    100% { bottom: 58%; opacity: 0; }
                }
            }
        </style>
    </head>
    <body>
        <div id="quotes-container"></div>
        <script>
            const quotes = [
                "Believe in yourself",
                "Every day is a fresh start",
                "Your feelings are valid",
                "Progress, not perfection",
                "You are stronger than you think",
                "One step at a time",
                "Be kind to yourself",
                "Today is full of possibilities",
                "You matter",
                "Breathe and let go",
                "Small steps lead to big changes",
                "You are enough",
                "This too shall pass",
                "Choose joy today",
                "Your story isn't over yet",
                "Embrace the journey",
                "You deserve happiness",
                "Keep going, you're doing great"
            ];
            const colors = [
                '#1f2937', '#3b82f6', '#10b981', '#f59e0b', '#ec4899',
                '#0ea5e9', '#14b8a6', '#f97316', '#64748b', '#06b6d4'
            ];
            const container = document.getElementById('quotes-container');
            let quoteIndex = 0;
            let colorIndex = 0;
            const isMobile = window.innerWidth <= 768;
            function createQuote() {
                const quote = document.createElement('div');
                quote.className = 'quote';
                quote.textContent = '"' + quotes[quoteIndex] + '"';
                quote.style.left = (Math.random() * (isMobile ? 58 : 65) + (isMobile ? 3 : 5)) + '%';
                quote.style.fontSize = (isMobile ? (0.82 + Math.random() * 0.18) : (1.0 + Math.random() * 0.3)) + 'rem';
                quote.style.color = colors[colorIndex];
                container.appendChild(quote);
                setTimeout(() => { quote.remove(); }, 14000);
                quoteIndex = (quoteIndex + 1) % quotes.length;
                colorIndex = (colorIndex + 1) % colors.length;
            }
            const initialBurst = isMobile ? 3 : 6;
            const intervalMs = isMobile ? 4200 : 3000;
            for (let i = 0; i < initialBurst; i++) { setTimeout(() => createQuote(), i * 1500); }
            setInterval(createQuote, intervalMs);
        </script>
    </body>
    </html>
    """

    components.html(quotes_html, height=0, scrolling=False)

    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background-color: #ffffff; overflow: hidden; }
        .stApp { background-color: #ffffff; }
        iframe {
            position: fixed !important; top: 0 !important; left: 0 !important;
            width: 100vw !important; height: 100vh !important;
            border: none !important; z-index: 0 !important; pointer-events: none !important;
        }
        .stApp > header, .main, .block-container { position: relative; z-index: 10; }
        [data-testid="stForm"] { position: relative; z-index: 15 !important; }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .main .block-container { animation: fadeInUp 0.6s ease-out; }
        .stTextInput input { background-color: #f8fafc; border: 2px solid #e2e8f0; color: #1e293b; }
        .stTextInput input:focus { box-shadow: 0 0 15px rgba(0,0,0,0.1); border-color: #1f2937; }
        .stTextInput label { color: #1f2937 !important; }
        .stButton > button {
            transition: all 0.3s ease; background: #ffffff;
            color: #000000; border: 2px solid #1f2937;
        }
        .stButton > button:hover {
            transform: scale(1.03); box-shadow: 0 6px 20px rgba(0,0,0,0.15); background: #f8fafc;
        }
        [data-testid="stForm"] {
            background-color: rgba(255,255,255,0.9); padding: 2rem;
            border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem; position: relative; z-index: 1;">
            <h1 style="margin-bottom: 0.5rem; color: #1e293b; font-weight: 700;">FACES TO FEELINGS</h1>
            <p style="color: #64748b; font-size: 1.1rem;">Let your emotions fill the pages today, and discover what's waiting for you..</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.caption("Create an account or log in to continue.")
        auth_mode = st.radio(
            "Choose mode",
            ["Login", "Register"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode_switch",
        )

        if auth_mode == "Login":
            with st.form("login_form"):
                login_username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    key="login_username",
                )
                login_password = st.text_input(
                    "Password",
                    type="password",
                    key="login_password",
                )
                remember_me = st.checkbox("Remember me", value=True)
                login_submitted = st.form_submit_button("Log in")
                if login_submitted:
                    if not login_username.strip() or not login_password:
                        st.error("Please enter both username and password.")
                    else:
                        auth_result = authenticate_user(login_username, login_password)
                        if auth_result:
                            uid, nickname, username = auth_result
                            st.session_state.user_id = uid
                            st.session_state.nickname = nickname
                            st.session_state.username = username
                            if remember_me:
                                token = create_auth_session(uid)
                                st.session_state.remember_token = token
                                st.session_state.remember_cookie_to_set = token
                            else:
                                st.session_state.remember_token = None
                                st.session_state.clear_remember_cookie = True
                            st.toast(f"Welcome back, {nickname}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")

        else:
            with st.form("register_form"):
                reg_username = st.text_input(
                    "Username",
                    placeholder="letters, numbers, underscores",
                    help="Username must be 3-30 characters.",
                    key="register_username",
                )
                reg_nickname = st.text_input(
                    "Nickname",
                    placeholder="How should we call you?",
                    key="register_nickname",
                )
                reg_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="At least 8 characters, include numbers and letters",
                    key="register_password",
                )
                reg_confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="At least 8 characters, include numbers and letters",
                    key="register_confirm_password",
                )
                register_submitted = st.form_submit_button("Create account")
                if register_submitted:
                    if reg_password != reg_confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        user_id, cleaned_nickname, error = create_user(
                            reg_username, reg_nickname, reg_password
                        )
                        if error:
                            st.error(error)
                        else:
                            st.session_state.user_id = user_id
                            st.session_state.nickname = cleaned_nickname
                            st.session_state.username = normalize_username(reg_username)
                            token = create_auth_session(user_id)
                            st.session_state.remember_token = token
                            st.session_state.remember_cookie_to_set = token
                            st.toast(f"Account created. Hello, {cleaned_nickname}!")
                            st.rerun()

    st.stop()

# -------------------------------------------------------------------
# Main app (authenticated)
# -------------------------------------------------------------------

# Global CSS
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #cbd5e1 100%);
        background-attachment: fixed;
    }
    .stApp h1, .stApp h2, .stApp h3 { color: #1e293b; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    .stApp p, .stApp span, .stApp label { color: #475569; }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main .block-container { animation: fadeIn 0.5s ease-out; }
    .stButton > button {
        transition: all 0.3s ease; background: #ffffff;
        color: #000000; border: 2px solid #1f2937;
    }
    .stButton > button:hover {
        transform: scale(1.02); box-shadow: 0 4px 20px rgba(0,0,0,0.15); background: #f8fafc;
    }
    .stButton > button:active { box-shadow: 0 0 25px rgba(0,0,0,0.2); }
    .activity-card {
        transition: all 0.3s ease; background: rgba(255,255,255,0.8); border: 1px solid #e2e8f0;
    }
    .activity-card:hover {
        transform: translateY(-4px); box-shadow: 0 8px 25px rgba(0,0,0,0.1); border-color: #1f2937;
    }
    .stTextInput input { background-color: #ffffff; border: 2px solid #e2e8f0; color: #000000; }
    .stTextInput input:focus { box-shadow: 0 0 10px rgba(0,0,0,0.1); border-color: #1f2937; }
    .stSlider [data-baseweb="slider"] div { transition: all 0.2s ease; }
    .stTabs [data-baseweb="tab"] { color: #475569; }
    .stTabs [data-baseweb="tab"]:hover { color: #000000; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #000000; }
    .stSidebar {
        background-color: rgba(248, 250, 252, 0.98);
        border-right: 1px solid #e2e8f0;
    }
    .stSidebar .stButton > button {
        background: #ffffff;
        color: #1e293b;
        border: 1px solid #cbd5e1;
    }
    .stSidebar .stButton > button:hover {
        background: #f1f5f9;
        color: #0f172a;
        border-color: #94a3b8;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar span, .stSidebar label {
        color: #1e293b !important;
    }
    [data-testid="stMetric"] {
        transition: all 0.3s ease; border-radius: 12px; padding: 12px;
        background: rgba(255,255,255,0.7); border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    [data-testid="stMetric"] label { color: #64748b !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #1e293b !important; }
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.8); padding: 1.5rem;
        border-radius: 12px; border: 1px solid #e2e8f0;
    }
    .streamlit-expanderHeader { color: #1e293b; }
    .emotion-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white;
        text-align: center; margin: 0.5rem 0;
    }
    .emotion-label { font-size: 1.4rem; font-weight: bold; }
    .confidence-text { font-size: 1rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Greeting (Philippine Time)
hour = now_ph().hour
if 5 <= hour < 12:
    time_greeting = "Good morning"
elif 12 <= hour < 17:
    time_greeting = "Good afternoon"
else:
    time_greeting = "Good evening"
nickname = st.session_state.get("nickname", "there")
st.title(f"Hello, {time_greeting}, {nickname}!")

# Sidebar
with st.sidebar:
    st.markdown("## FacesToFeelings")
    with st.expander("About"):
        st.caption(
            "The Mood Tracker System is designed to help users monitor, record, and reflect on their emotional states over time. It allows users to log their mood, rate its intensity, and provide personal reflections to better understand emotional patterns and triggers. The system securely stores mood data and provides a structured way for users to become more aware of their mental and emotional well-being. By combining technology and data-driven insights, the Mood Tracker aims to support self-awareness, emotional regulation, and overall mental wellness. "
        )
    if st.button("Logout"):
        if st.session_state.remember_token:
            revoke_auth_session(st.session_state.remember_token)

        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.clear_remember_cookie = True
        st.rerun()

# Load model
with st.spinner("Loading emotion model..."):
    model = load_emotion_model()
    face_detector = load_face_detector()

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_mood, tab_profile, tab_calendar, tab_journal = st.tabs(
    ["Mood Tracker", "Profile", "Mood Calendar", "My Journal"]
)

if st.session_state.preferred_main_tab:
    preferred_tab = st.session_state.preferred_main_tab
    components.html(
        f"""
        <script>
            function activatePreferredMainTab() {{
                const tabButtons = parent.document.querySelectorAll('[role="tab"]');
                for (const btn of tabButtons) {{
                    const label = (btn.textContent || '').trim();
                    if (label === "{preferred_tab}") {{
                        btn.click();
                        break;
                    }}
                }}
            }}
            setTimeout(activatePreferredMainTab, 40);
            setTimeout(activatePreferredMainTab, 160);
            setTimeout(activatePreferredMainTab, 320);
        </script>
        """,
        height=0,
        scrolling=False,
    )
    st.session_state.preferred_main_tab = None

# -------------------------------------------------------------------
# Mood Tracker tab
# -------------------------------------------------------------------
with tab_mood:
    st.subheader("Let's see what's written in your face.")

    with st.expander("How it works"):
        st.caption(
            """
            The Mood Tracker System allows users to create an account and securely log their daily emotional states. Users scan their current mood, rate its intensity, and optionally write reflections to describe their thoughts or experiences. Each entry is stored in the system and organized by date, allowing users to review their emotional history and observe patterns over time. This process helps users become more aware of their emotional changes and supports better self-reflection and mental well-being.

            The system is continuously being developed and improved. As part of its development, there may be occasional errors or inaccuracies in recording, displaying, or interpreting mood data. These mistakes are recognized as part of the learning and improvement process, and efforts are continuously made to enhance the system's accuracy, reliability, and overall user experience. User feedback and ongoing testing play an important role in refining the system to ensure it becomes more effective and dependable over time.
            """
        )

    # Two sub-tabs: Camera and Upload
    sub_camera, sub_upload = st.tabs(["Take Photo (Camera)", "Upload Image"])

    with sub_camera:
        st.subheader("Capture this moment")
        st.caption(
            "Position yourself in the frame and take your photo when you're ready. "
            "Let's see what your expression has to tell us about you today."
        )

        img_file = st.camera_input("Capture your face", label_visibility="collapsed")

        if img_file is not None:
            camera_image = Image.open(img_file)
            camera_image = mirror_image(camera_image)
            with st.spinner("Analyzing emotions..."):
                annotated_image, results = detect_emotions(camera_image, model, face_detector)

            if results:
                st.image(annotated_image, use_container_width=True)
                top_result = results[0]
                label = top_result["label"]
                st.success(f"Detected mood: {label} ({top_result['confidence']:.1f}%)")
                st.session_state.last_detected_mood = label

                if len(results) > 1:
                    st.info(f"**{len(results)} face(s) detected.** Using the first face for mood logging.")
            else:
                st.warning("Oops! We missed you that time. Let's try again - make sure you're front and center!")

    with sub_upload:
        st.subheader("Upload a photo")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Upload a photo with your face near the center for best results.",
        )
        if uploaded_file is not None:
            upload_image = Image.open(uploaded_file)
            col_img, col_res = st.columns([3, 2])

            with col_img:
                st.subheader("Detection Result")
                with st.spinner("Analyzing emotions..."):
                    annotated_image, results = detect_emotions(upload_image, model, face_detector)
                st.image(annotated_image, use_container_width=True)

            with col_res:
                st.subheader("Results")
                if len(results) == 0:
                    st.warning("No faces detected. Try a different photo with clearly visible faces.")
                else:
                    st.info("**1 face analyzed (center-most face selected).**")
                    top_result = results[0]
                    st.session_state.last_detected_mood = top_result["label"]

                    for i, res in enumerate(results):
                        st.markdown(
                            f'<div class="emotion-card">'
                            f'<div class="emotion-label">{res["label"]}</div>'
                            f'<div class="confidence-text">Confidence: {res["confidence"]:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # -------------------------------------------------------------------
    # Mood result, intensity, and activities (below face scanning)
    # -------------------------------------------------------------------
    st.markdown('<div id="mood-workflow-anchor"></div>', unsafe_allow_html=True)
    st.markdown("---")
    current_mood = st.session_state.get("last_detected_mood", None)

    if current_mood:
        mood_colors = MOOD_CARD_COLORS.get(current_mood, MOOD_CARD_COLORS["Neutral"])
        st.markdown(
            f"""
            <style>
            [data-testid="stFormSubmitButton"] > button,
            .stButton > button {{
                background-color: {mood_colors['bg']} !important;
                border: 2px solid {mood_colors['border']} !important;
                color: {mood_colors['text']} !important;
            }}
            [data-testid="stFormSubmitButton"] > button:hover,
            .stButton > button:hover {{
                filter: brightness(0.97);
                box-shadow: 0 4px 14px rgba(0,0,0,0.12);
            }}
            /* Mood-based slider styling */
            .stSlider [data-baseweb="slider"] [role="slider"],
            [data-testid="stSelectSlider"] [data-baseweb="slider"] [role="slider"] {{
                background-color: {mood_colors['border']} !important;
                border-color: {mood_colors['border']} !important;
            }}
            .stSlider [data-baseweb="slider"] > div > div > div,
            [data-testid="stSelectSlider"] [data-baseweb="slider"] > div > div > div {{
                background-color: {mood_colors['bg']} !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"## {current_mood}")
        st.caption("How much space is this mood taking up today? Choose from 1-5")
        with st.form("mood_form"):
            intensity = st.select_slider(
                "Mood intensity",
                options=[1, 2, 3, 4, 5],
                value=3,
                format_func=lambda x: INTENSITY_LABELS[x],
                key="mood_intensity",
                label_visibility="collapsed",
            )
            save_mood = st.form_submit_button(
                "Save mood",
                type="primary",
                use_container_width=False,
            )
        if save_mood:
            mood_log_id = log_mood(st.session_state.user_id, current_mood, intensity)
            st.session_state.show_journal = True
            st.session_state.current_mood_log_id = mood_log_id
            st.session_state.journal_mood = current_mood
            st.toast("Mood saved! Now let's reflect...")

        st.markdown("---")

        if st.session_state.show_journal and st.session_state.journal_mood:
            journal_mood = st.session_state.journal_mood
            st.markdown("### Reflection Journal")
            st.markdown(f"Take a moment to reflect on feeling **{journal_mood}**")

            st.markdown(
                """
                <style>
                .journal-container {
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    padding: 1.5rem; border-radius: 1rem;
                    border: 1px solid #e2e8f0; margin-bottom: 1rem;
                }
                .journal-question { color: #1e293b; font-weight: 500; margin-bottom: 0.5rem; }
                </style>
                """,
                unsafe_allow_html=True,
            )

            questions = JOURNAL_QUESTIONS.get(journal_mood, JOURNAL_QUESTIONS["Neutral"])

            with st.form("journal_form"):
                answers = {}
                for i, question in enumerate(questions):
                    st.markdown(f"**{i + 1}. {question}**")
                    answers[question] = st.text_area(
                        "Your answer",
                        key=f"journal_q_{i}",
                        height=80,
                        label_visibility="collapsed",
                        placeholder="Type your thoughts here...",
                    )

                st.markdown("**Anything else on your mind?** (Optional)")
                free_text = st.text_area(
                    "Free thoughts",
                    key="journal_free_text",
                    height=100,
                    label_visibility="collapsed",
                    placeholder="Write freely here...",
                )
                save_journal = st.form_submit_button(
                    "Save Journal",
                    type="primary",
                    use_container_width=True,
                )

            col_skip_left, col_skip_right = st.columns([1, 1])
            with col_skip_left:
                if st.button("Skip for now", use_container_width=True):
                    st.session_state.show_journal = False
                    st.session_state.current_mood_log_id = None
                    st.session_state.journal_mood = None
                    st.toast("No worries, you can reflect anytime!")
                    rerun_to_anchor("mood-workflow-anchor")

            if save_journal:
                answers_json = json.dumps(answers)
                reflection_id = save_reflection(
                    st.session_state.user_id,
                    st.session_state.current_mood_log_id,
                    journal_mood,
                    answers_json,
                    free_text,
                )
                st.session_state.show_journal = False
                st.session_state.show_rating = True
                st.session_state.pending_reflection_id = reflection_id
                st.toast("Journal saved! One more step...")
                rerun_to_anchor("mood-workflow-anchor")

        elif st.session_state.show_rating and st.session_state.pending_reflection_id:
            st.markdown("### How do you feel now?")
            st.markdown("After reflecting, do you feel any better?")

            RATING_LABELS = {
                1: "Much worse",
                2: "A bit worse",
                3: "About the same",
                4: "A bit better",
                5: "Much better",
            }

            with st.form("rating_form"):
                rating = st.select_slider(
                    "Rate your improvement",
                    options=[1, 2, 3, 4, 5],
                    value=3,
                    format_func=lambda x: RATING_LABELS[x],
                    key="improvement_rating_slider",
                )
                save_rating = st.form_submit_button(
                    "Save Rating",
                    type="primary",
                    use_container_width=True,
                )

            _, col_skip_rate = st.columns([1, 1])
            with col_skip_rate:
                if st.button("Skip rating", use_container_width=True):
                    st.session_state.show_rating = False
                    st.session_state.pending_reflection_id = None
                    st.session_state.current_mood_log_id = None
                    st.session_state.journal_mood = None
                    st.toast("That's okay! Your journal is saved.")
                    rerun_to_anchor("mood-workflow-anchor")

            if save_rating:
                update_reflection_rating(st.session_state.pending_reflection_id, rating)
                st.session_state.show_rating = False
                st.session_state.pending_reflection_id = None
                st.session_state.current_mood_log_id = None
                st.session_state.journal_mood = None
                st.toast("Thank you for sharing! Take care!")
                rerun_to_anchor("mood-workflow-anchor")

        else:
            st.markdown("#### Handpicked for how you feel")
            activities = MOOD_ACTIVITIES.get(current_mood, MOOD_ACTIVITIES["Neutral"])

            colors = MOOD_CARD_COLORS.get(current_mood, MOOD_CARD_COLORS["Neutral"])

            st.markdown(
                f"""
                <style>
                .activity-card {{
                    padding: 1rem; border-radius: 0.75rem;
                    border: 2px solid {colors['border']}; background: {colors['bg']};
                    margin-bottom: 0.75rem; transition: all 0.3s ease;
                }}
                .activity-card:hover {{
                    transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .activity-card h4 {{ margin: 0 0 0.25rem 0; color: {colors['text']}; }}
                .activity-card p {{ margin: 0; color: {colors['desc']}; font-size: 0.9rem; }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            for i in range(0, len(activities), 2):
                row = activities[i: i + 2]
                cols = st.columns(2)
                for j, (title, desc) in enumerate(row):
                    with cols[j]:
                        st.markdown(
                            f'<div class="activity-card"><h4>{title}</h4><p>{desc}</p></div>',
                            unsafe_allow_html=True,
                        )

# -------------------------------------------------------------------
# Profile tab
# -------------------------------------------------------------------
PLOTLY_DARK_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "responsive": True,
}


def _apply_plotly_dark(fig):
    """Apply dark template and layout for consistent styling."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


with tab_profile:
    st.subheader("Your mood statistics")
    user_id = st.session_state.user_id

    if st.button("Refresh charts", key="refresh_profile"):
        st.toast("Charts refreshed!")

    days_range = st.radio(
        "Date range",
        [7, 14, 30],
        format_func=lambda x: f"Last {x} days",
        horizontal=True,
        key="profile_days",
    )

    total_logs = get_total_logs(user_id)
    logs_today = get_logs_today(user_id)
    dominant = get_dominant_mood(user_id, days_range) or "N/A"

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total mood logs", total_logs)
    with m2:
        st.metric("Logs today", logs_today)
    with m3:
        st.metric("Dominant mood", dominant)

    st.divider()

    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for charts: `pip install plotly`")
    else:
        daily = get_daily_moods(user_id)
        weekly = get_weekly_moods(user_id, days_range)
        dist = get_mood_distribution(user_id, days_range)

        if not daily and not weekly and not dist:
            st.info("No mood data yet. Start tracking your mood in the Mood Tracker tab!")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Daily trends (today)")
                if daily:
                    # Guard against malformed timestamps so the chart never exceeds 0-23 hours.
                    df_daily = []
                    for m, h, c in daily:
                        try:
                            hour_val = int(h)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= hour_val <= 23:
                            df_daily.append({"Hour": hour_val, "Mood": m, "Count": c})

                    if not df_daily:
                        st.caption("No valid hourly data for today.")
                    else:
                        fig_daily = px.bar(
                            df_daily, x="Hour", y="Count", color="Mood",
                            title="Mood by hour today", barmode="stack",
                        )
                        _apply_plotly_dark(fig_daily)
                        fig_daily.update_layout(
                            height=350,
                            margin=dict(t=40, b=40, l=40, r=20),
                            xaxis=dict(range=[-0.5, 23.5], dtick=1),
                        )
                        st.plotly_chart(fig_daily, use_container_width=True, config=PLOTLY_DARK_CONFIG)
                else:
                    st.caption("No data for today yet.")

            with col2:
                st.markdown(f"#### Weekly trends (last {days_range} days)")
                if weekly:
                    # Strictly limit data points to selected trailing day window.
                    end_date = now_ph().date()
                    start_date = end_date - timedelta(days=days_range - 1)
                    df_weekly = []
                    for m, d, c in weekly:
                        try:
                            day_obj = datetime.strptime(d, "%Y-%m-%d").date()
                        except (TypeError, ValueError):
                            continue
                        if start_date <= day_obj <= end_date:
                            df_weekly.append(
                                {"Day": day_obj.strftime("%Y-%m-%d"), "Mood": m, "Count": c}
                            )

                    if not df_weekly:
                        st.caption(f"No valid daily data for the last {days_range} days yet.")
                    else:
                        fig_weekly = px.bar(
                            df_weekly, x="Day", y="Count", color="Mood",
                            title=f"Mood by day (last {days_range} days)", barmode="stack",
                        )
                        _apply_plotly_dark(fig_weekly)
                        fig_weekly.update_layout(height=350, margin=dict(t=40, b=40, l=40, r=20))
                        st.plotly_chart(fig_weekly, use_container_width=True, config=PLOTLY_DARK_CONFIG)
                else:
                    st.caption(f"No data for the last {days_range} days yet.")

            st.divider()

            st.markdown(f"#### Mood distribution (last {days_range} days)")
            if dist:
                df_dist = [{"Mood": m, "Count": c} for m, c in dist]
                fig_pie = px.pie(
                    df_dist, values="Count", names="Mood",
                    title="Overall mood distribution",
                )
                _apply_plotly_dark(fig_pie)
                fig_pie.update_layout(height=400, margin=dict(t=40, b=40, l=40, r=40))
                st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_DARK_CONFIG)
            else:
                st.caption("No mood distribution data yet.")

# -------------------------------------------------------------------
# Mood Calendar tab
# -------------------------------------------------------------------
with tab_calendar:
    st.subheader("Mood Calendar")
    st.write("Click on a date to view your mood entries for that day.")

    st.markdown(
        """
        <style>
        .calendar-header { text-align: center; font-size: 1.3rem; font-weight: 600; margin: 0.5rem 0; }
        .calendar-day-header { text-align: center; font-weight: 600; color: #94a3b8; padding: 0.5rem 0; }
        .mood-entry-card {
            padding: 1rem; border-radius: 0.5rem; border: 1px solid #e2e8f0;
            background: #ffffff; margin-bottom: 0.75rem;
        }
        .mood-entry-card .time { color: #64748b; font-size: 0.85rem; }
        .mood-entry-card .mood { font-size: 1.1rem; font-weight: 600; margin: 0.25rem 0; color: #000000; }
        .intensity-bar {
            height: 8px; border-radius: 4px;
            background: linear-gradient(90deg, #64748b, #1f2937); margin-top: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        if st.button("Previous", key="prev_month"):
            st.session_state.preferred_main_tab = "Mood Calendar"
            if st.session_state.calendar_month == 1:
                st.session_state.calendar_month = 12
                st.session_state.calendar_year -= 1
            else:
                st.session_state.calendar_month -= 1
            st.session_state.selected_calendar_date = None

    with nav_col2:
        month_name = calendar.month_name[st.session_state.calendar_month]
        st.markdown(
            f'<div class="calendar-header">{month_name} {st.session_state.calendar_year}</div>',
            unsafe_allow_html=True,
        )

    with nav_col3:
        if st.button("Next", key="next_month"):
            st.session_state.preferred_main_tab = "Mood Calendar"
            if st.session_state.calendar_month == 12:
                st.session_state.calendar_month = 1
                st.session_state.calendar_year += 1
            else:
                st.session_state.calendar_month += 1
            st.session_state.selected_calendar_date = None

    dominant_moods = get_dominant_moods_by_date(
        user_id,
        st.session_state.calendar_year,
        st.session_state.calendar_month,
    )
    today = now_ph().date()

    # Build JS color map for all 7 emotions
    js_mood_colors = ", ".join(
        f"'{m}': {{bg: '{c['bg']}', border: '{c['border']}', text: '{c['text']}'}}"
        for m, c in MOOD_CALENDAR_COLORS.items()
    )

    components.html(
        f"""
        <script>
        function colorMoodButtons() {{
            const moodColors = {{ {js_mood_colors} }};
            const buttons = parent.document.querySelectorAll('button');
            buttons.forEach(btn => {{
                const text = btn.textContent;
                for (const [mood, colors] of Object.entries(moodColors)) {{
                    if (text.includes(mood)) {{
                        btn.style.setProperty('background-color', colors.bg, 'important');
                        btn.style.setProperty('border-color', colors.border, 'important');
                        btn.style.setProperty('color', colors.text, 'important');
                        btn.style.setProperty('border-width', '2px', 'important');
                        break;
                    }}
                }}
            }});
        }}
        colorMoodButtons();
        setTimeout(colorMoodButtons, 50);
        setTimeout(colorMoodButtons, 150);
        setTimeout(colorMoodButtons, 300);
        setTimeout(colorMoodButtons, 600);
        setTimeout(colorMoodButtons, 1000);
        const observer = new MutationObserver(() => {{ colorMoodButtons(); }});
        if (parent.document.body) {{
            observer.observe(parent.document.body, {{ childList: true, subtree: true }});
        }}
        </script>
        """,
        height=0,
    )

    cal = calendar.Calendar(firstweekday=6)
    month_days = cal.monthdayscalendar(
        st.session_state.calendar_year,
        st.session_state.calendar_month,
    )

    day_headers = st.columns(7)
    day_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    for i, day_name in enumerate(day_names):
        with day_headers[i]:
            st.markdown(f'<div class="calendar-day-header">{day_name}</div>', unsafe_allow_html=True)

    for week in month_days:
        week_cols = st.columns(7)
        for i, day in enumerate(week):
            with week_cols[i]:
                if day == 0:
                    st.write("")
                else:
                    date_str = f"{st.session_state.calendar_year}-{st.session_state.calendar_month:02d}-{day:02d}"
                    current_date = datetime(
                        st.session_state.calendar_year,
                        st.session_state.calendar_month,
                        day,
                    ).date()

                    dominant_mood = dominant_moods.get(date_str)
                    is_today = current_date == today
                    is_selected = st.session_state.selected_calendar_date == date_str

                    label = str(day)

                    button_type = "primary" if is_selected else "secondary"

                    if st.button(label, key=f"cal_{date_str}", type=button_type, use_container_width=True):
                        st.session_state.preferred_main_tab = "Mood Calendar"
                        st.session_state.selected_calendar_date = date_str

    st.divider()

    if st.session_state.selected_calendar_date:
        selected_date = datetime.strptime(st.session_state.selected_calendar_date, "%Y-%m-%d")
        formatted_date = selected_date.strftime("%B %d, %Y")
        st.markdown(f"### Mood entries for {formatted_date}")

        entries = get_moods_by_date(user_id, st.session_state.selected_calendar_date)

        if entries:
            intensity_labels = {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}

            for mood, intensity, time in entries:
                intensity_pct = (intensity / 5) * 100
                intensity_label = intensity_labels.get(intensity, "Moderate")

                st.markdown(
                    f"""
                    <div class="mood-entry-card">
                        <div class="time">{time}</div>
                        <div class="mood">{mood}</div>
                        <div>Intensity: {intensity_label} ({intensity}/5)</div>
                        <div class="intensity-bar" style="width: {intensity_pct}%;"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No mood entries for this date.")
    else:
        st.info(
            "Select a date to view entries. Button colors show dominant mood: "
            "Happy, Sad, Angry, Neutral, Disgust, Fear, Surprise."
        )

# -------------------------------------------------------------------
# My Journal tab
# -------------------------------------------------------------------
with tab_journal:
    st.subheader("My Journal")
    st.write("Your reflection history - a record of your inner journey.")

    user_id = st.session_state.user_id

    if st.button("Refresh journal", key="refresh_journal"):
        st.toast("Journal refreshed!")

    reflections = get_user_reflections(user_id, limit=50)

    if reflections:
        rating_labels = {
            1: "Much worse",
            2: "A bit worse",
            3: "About the same",
            4: "A bit better",
            5: "Much better",
            None: "Not rated",
        }

        st.markdown(
            """
            <style>
            .journal-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                padding: 1.25rem; border-radius: 1rem;
                border: 1px solid #e2e8f0; margin-bottom: 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            }
            .journal-card-header {
                display: flex; justify-content: space-between; align-items: center;
                margin-bottom: 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid #f1f5f9;
            }
            .journal-date { color: #64748b; font-size: 0.85rem; }
            .journal-mood-badge {
                display: inline-block; padding: 0.25rem 0.75rem;
                border-radius: 1rem; font-size: 0.85rem; font-weight: 500;
            }
            .mood-Happy { background: #fef3c7; color: #92400e; }
            .mood-Sad { background: #dbeafe; color: #1e40af; }
            .mood-Angry { background: #fee2e2; color: #991b1b; }
            .mood-Neutral { background: #f3f4f6; color: #374151; }
            .mood-Disgust { background: #d1fae5; color: #065f46; }
            .mood-Fear { background: #ede9fe; color: #5b21b6; }
            .mood-Surprise { background: #fef9c3; color: #854d0e; }
            .journal-qa {
                margin: 0.75rem 0; padding: 0.75rem;
                background: #f8fafc; border-radius: 0.5rem;
            }
            .journal-question { color: #475569; font-size: 0.9rem; font-weight: 500; margin-bottom: 0.25rem; }
            .journal-answer { color: #1e293b; font-size: 0.95rem; }
            .journal-rating {
                margin-top: 0.75rem; padding: 0.5rem 0.75rem;
                background: #f0fdf4; border-radius: 0.5rem; color: #166534; font-size: 0.9rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        grouped = defaultdict(list)
        for r in reflections:
            ref_id, mood, answers_json, free_text, improvement_rating, created_at, intensity = r
            date_str = created_at.split(" ")[0] if " " in str(created_at) else str(created_at)[:10]
            grouped[date_str].append(r)

        for date_str, date_entries in grouped.items():
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%B %d, %Y")
            except Exception:
                formatted_date = date_str

            with st.expander(
                f"{formatted_date} ({len(date_entries)} entr{'y' if len(date_entries) == 1 else 'ies'})",
                expanded=False,
            ):
                for ref_id, mood, answers_json, free_text, improvement_rating, created_at, intensity in date_entries:
                    try:
                        time_str = created_at.split(" ")[1][:5] if " " in str(created_at) else ""
                    except Exception:
                        time_str = ""

                    try:
                        answers = json.loads(answers_json) if answers_json else {}
                    except Exception:
                        answers = {}

                    st.markdown(
                        f"""
                        <div class="journal-card">
                            <div class="journal-card-header">
                                <span class="journal-mood-badge mood-{mood}">{mood}</span>
                                <span class="journal-date">{time_str}</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    for question, answer in answers.items():
                        if answer and answer.strip():
                            st.markdown(
                                f"""
                                <div class="journal-qa">
                                    <div class="journal-question">Q: {question}</div>
                                    <div class="journal-answer">{answer}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    if free_text and free_text.strip():
                        st.markdown(
                            f"""
                            <div class="journal-qa">
                                <div class="journal-question">Additional thoughts:</div>
                                <div class="journal-answer">{free_text}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    rating_text = rating_labels.get(improvement_rating, "Not rated")
                    if improvement_rating:
                        st.markdown(
                            f"""
                            <div class="journal-rating">
                                After reflecting: {rating_text}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        total_entries = len(reflections)
        rated_entries = sum(1 for r in reflections if r[4] is not None)
        avg_rating = (
            sum(r[4] for r in reflections if r[4] is not None) / rated_entries
            if rated_entries > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Entries", total_entries)
        with col2:
            st.metric("Rated Entries", rated_entries)
        with col3:
            if avg_rating > 0:
                st.metric("Avg Improvement", f"{avg_rating:.1f}/5")
            else:
                st.metric("Avg Improvement", "N/A")
    else:
        st.info("No journal entries yet. Save a mood and complete the reflection journal to see your entries here!")
