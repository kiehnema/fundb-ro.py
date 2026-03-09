import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import sqlite3
import os
from datetime import datetime
import base64

# =============================
# SEITENEINSTELLUNGEN
# =============================
st.set_page_config(
    page_title="Digitales Fundbüro",
    layout="wide"
)

# =============================
# DESIGN / CSS
# =============================
st.markdown("""
<style>

.stApp {
    background-color: #f4f6f8;
}

h1 {
    text-align:center;
}

div.stButton > button {
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:10px;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom:25px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# LOGO ZENTRIERT
# =============================
def display_logo():
    with open("logo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="text-align:center;">
            <img src="data:image/png;base64,{data}" width="200">
        </div>
        """,
        unsafe_allow_html=True
    )

display_logo()
st.markdown("<h1>Digitales Fundbüro</h1>", unsafe_allow_html=True)
st.markdown("---")

# =============================
# UPLOAD ORDNER
# =============================
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# =============================
# DATENBANK
# =============================
conn = sqlite3.connect("fundbuero.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    category TEXT,
    confidence REAL,
    date TEXT
)
""")

c.execute("""
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER,
    email TEXT,
    date TEXT
)
""")

conn.commit()

# =============================
# MODELL LADEN
# =============================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# =============================
# NAVIGATION
# =============================
choice = st.radio(
    "Was möchtest du tun?",
    ["📸 Fund hochladen", "🔍 Fund suchen"],
    horizontal=True
)

# ==========================================================
# FUND HOCHLADEN
# ==========================================================
if choice == "📸 Fund hochladen":

    st.subheader("Neuen Fund registrieren")
    uploaded_file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image_resized)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array

        with st.spinner("🔍 Gegenstand wird analysiert..."):
            prediction = model.predict(data)

        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        st.success(f"Erkannt: {class_name}")
        st.write(f"Sicherheit: {round(confidence_score * 100, 2)} %")

        if st.button("Im Fundbüro speichern"):

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"uploads/{timestamp}_{uploaded_file.name}"
            image.save(filename)

            c.execute("""
                INSERT INTO items (filename, category, confidence, date)
                VALUES (?, ?, ?, ?)
            """, (filename, class_name, confidence_score, datetime.now()))

            conn.commit()
            st.success("✅ Fund erfolgreich gespeichert!")

# ==========================================================
# FUND SUCHEN
# ==========================================================
if choice == "🔍 Fund suchen":

    st.subheader("Gefundene Gegenstände durchsuchen")

    c.execute("SELECT DISTINCT category FROM items")
    db_categories = [row[0] for row in c.fetchall()]
    categories = ["Alle"] + db_categories
    selected_category = st.selectbox("Kategorie filtern", categories)

    if selected_category == "Alle":
        c.execute("SELECT * FROM items ORDER BY date DESC")
    else:
        c.execute("""
            SELECT * FROM items
            WHERE category LIKE ?
            ORDER BY date DESC
        """, (f"%{selected_category}%",))

    items = c.fetchall()

    if items:

        for item in items:

            item_id, filename, category, confidence, date = item

            st.markdown('<div class="card">', unsafe_allow_html=True)

            # Bild wieder feste Breite 250px
            st.image(Image.open(filename), width=250)

            st.write(f"**Kategorie:** {category}")
            date_str = datetime.fromisoformat(date).strftime("%d.%m.%Y %H:%M")
            st.write(f"**Datum:** {date_str}")
            st.write(f"**Sicherheit:** {round(confidence * 100, 2)} %")

            email = st.text_input(
                f"Deine E-Mail, falls es dir gehört (ID {item_id})",
                key=f"email_{item_id}"
            )

            if st.button(f"Anspruch senden für ID {item_id}"):

                if email:

                    c.execute("""
                        SELECT * FROM claims
                        WHERE item_id=? AND email=?
                    """, (item_id, email))

                    exists = c.fetchone()
                    if exists:
                        st.warning("Du hast bereits einen Anspruch gesendet.")
                    else:
                        c.execute("""
                            INSERT INTO claims (item_id, email, date)
                            VALUES (?, ?, ?)
                        """, (item_id, email, datetime.now()))
                        conn.commit()
                        st.success("📧 Dein Anspruch wurde registriert!")
                else:
                    st.error("Bitte eine E-Mail eingeben.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Derzeit keine Funde gespeichert.")
