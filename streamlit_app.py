import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import datetime
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Smart Attendance", layout="wide")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-5px);
}

/* Title */
.title {
    color: #000;
    font-size: 15px;
    font-weight: 500;
    opacity: 0.7;
}

/* Metric */
.metric {
    font-size: 30px;
    font-weight: bold;
}

/* Colors */
.green { color: #4CAF50; }
.blue { color: #007BFF; }
.orange { color: #FFA500; }

/* Icon */
.icon {
    font-size: 28px;
    float: right;
    opacity: 0.3;
}
</style>
""", unsafe_allow_html=True)

# -------------------- PATHS --------------------
CASCADE_PATH = "haarcascade_frontalface_default.xml"

if not os.path.exists(CASCADE_PATH):
    st.error("Haar Cascade file missing!")
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

os.makedirs("dataset", exist_ok=True)
os.makedirs("trainer", exist_ok=True)
os.makedirs("attendance", exist_ok=True)

ATT_FILE = "attendance/attendance.csv"
STUDENT_FILE = "students.csv"

# -------------------- LOGIN --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h2 style='text-align:center'>Smart Attendance Login</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")

        if st.button("Login"):
            if user == "admin" and pwd == "1234":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("Smart Attendance")

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>
    body { background-color: #0E1117; color: white; }
    .card { background: rgba(30,30,30,0.7); color: white; }
    .title { color: white; opacity: 0.8; }
    </style>
    """, unsafe_allow_html=True)

menu = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Register Student",
    "Train Model",
    "Take Attendance",
    "Download Data",
    "Logout"
])

# -------------------- REGISTER --------------------
def register():
    st.subheader("📸 Register Student")

    id = st.text_input("Student ID")
    name = st.text_input("Student Name")

    if st.button("Capture Face"):
        if not id.isdigit() or name == "":
            st.error("Enter valid details")
            return

        cam = cv2.VideoCapture(0)
        count = 0
        frame_placeholder = st.empty()

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x,y,w,h) in faces:
                count += 1
                cv2.imwrite(f"dataset/User.{id}.{count}.jpg", gray[y:y+h,x:x+w])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            frame_placeholder.image(frame, channels="BGR")

            if count >= 40:
                break

        cam.release()

        # Save student
        if os.path.exists(STUDENT_FILE):
            df = pd.read_csv(STUDENT_FILE)
        else:
            df = pd.DataFrame(columns=["ID","Name"])

        df = df[df["ID"] != int(id)]
        df.loc[len(df)] = [int(id), name]
        df.to_csv(STUDENT_FILE, index=False)

        st.success("Student Registered!")

# -------------------- TRAIN --------------------
def train():
    st.subheader("Train Model")

    if st.button("Train Model"):
        faces, ids = [], []

        for file in os.listdir("dataset"):
            path = os.path.join("dataset", file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            id = int(file.split(".")[1])

            detected = face_cascade.detectMultiScale(img)

            for (x,y,w,h) in detected:
                faces.append(img[y:y+h,x:x+w])
                ids.append(id)

        if len(faces) == 0:
            st.error("No dataset found")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save("trainer/trainer.yml")

        st.success("Model Trained!")

# -------------------- ATTENDANCE --------------------
def attendance():
    st.subheader("Take Attendance")

    if not os.path.exists("trainer/trainer.yml"):
        st.error("Train model first")
        return

    if st.button("Start Camera"):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer/trainer.yml")

        if not os.path.exists(ATT_FILE):
            pd.DataFrame(columns=["ID","Name","Date","Time"]).to_csv(ATT_FILE, index=False)

        students = pd.read_csv(STUDENT_FILE) if os.path.exists(STUDENT_FILE) else pd.DataFrame()

        cam = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in faces:
                id, conf = recognizer.predict(gray[y:y+h,x:x+w])

                if conf < 65:
                    name = students.loc[students["ID"]==id,"Name"].values
                    name = name[0] if len(name)>0 else "Unknown"

                    now = datetime.datetime.now()
                    date = now.strftime("%d-%m-%Y")
                    time = now.strftime("%H:%M:%S")

                    df = pd.read_csv(ATT_FILE)

                    if not ((df["ID"]==id)&(df["Date"]==date)).any():
                        df.loc[len(df)] = [id,name,date,time]
                        df.to_csv(ATT_FILE,index=False)

                    label = name
                else:
                    label = "Unknown"

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,label,(x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

            frame_placeholder.image(frame, channels="BGR")

# -------------------- DASHBOARD --------------------
def dashboard():
    st.subheader("Dashboard")

    if not os.path.exists(ATT_FILE):
        st.warning("No data yet")
        return

    df = pd.read_csv(ATT_FILE)
    today = datetime.datetime.now().strftime("%d-%m-%Y")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='title'>Total Records</div>
            <div class='metric green'>{len(df)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='card'>
            <div class='title'>Unique Students</div>
            <div class='metric blue'>{df['ID'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='card'>
            <div class='title'>Today's Attendance</div>
            <div class='metric orange'>{len(df[df['Date']==today])}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Attendance Trend")

    chart = df.groupby("Date").count()["ID"].reset_index()

    fig = px.area(chart, x="Date", y="ID", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Full Data")
    st.dataframe(df)

# -------------------- DOWNLOAD --------------------
def download():
    st.subheader("Download Data")

    if os.path.exists(ATT_FILE):
        df = pd.read_csv(ATT_FILE)
        st.download_button("Download CSV", df.to_csv(index=False), "attendance.csv")

# -------------------- LOGOUT --------------------
def logout():
    st.session_state.logged_in = False
    st.rerun()

# -------------------- ROUTING --------------------
if menu == "Dashboard":
    dashboard()
elif menu == "Register Student":
    register()
elif menu == "Train Model":
    train()
elif menu == "Take Attendance":
    attendance()
elif menu == "Download Data":
    download()
elif menu == "Logout":
    logout()