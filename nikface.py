import os
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Face Recognition Attendance System", page_icon="📊", layout="wide")

# Create necessary directories
for dir_path in ["data", "data/faces", "data/attendance"]:
    os.makedirs(dir_path, exist_ok=True)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, None
    
    # Extract the face with the largest area
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    # Extract the face region
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    
    return face, (x, y, w, h)

# Function to extract face features
def extract_features(face):
    hog = cv2.HOGDescriptor((100, 100), (20, 20), (10, 10), (10, 10), 9)
    return hog.compute(face)

# Function to register a new face
def register_face(name, img):
    face, rect = detect_face(img)
    if face is None:
        return False, "No face detected. Please try again."
    
    # Extract features and save
    features = extract_features(face)
    np.save(f"data/faces/{name}.npy", features)
    cv2.imwrite(f"data/faces/{name}.jpg", cv2.cvtColor(face, cv2.COLOR_GRAY2BGR))
    
    return True, "Face registered successfully!"

# Function to recognize a face
def recognize_face(img, threshold=0.7):
    face, rect = detect_face(img)
    if face is None:
        return None, None
    
    features = extract_features(face)
    
    # Load all registered faces
    registered_faces = {}
    for file in os.listdir("data/faces"):
        if file.endswith(".npy"):
            name = file.split(".")[0]
            registered_faces[name] = np.load(f"data/faces/{file}")
    
    if not registered_faces:
        return None, None
    
    # Compare with registered faces
    max_similarity = -1
    recognized_name = None
    
    for name, face_features in registered_faces.items():
        similarity = cosine_similarity(features.reshape(1, -1), face_features.reshape(1, -1))[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = name
    
    return (recognized_name, max_similarity) if max_similarity >= threshold else (None, None)

# Function to mark attendance
def mark_attendance(name):
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Create or load attendance file for today
    attendance_file = f"data/attendance/{date}.csv"
    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])
    
    # Check if already marked attendance
    if name in df["Name"].values:
        return False, "Attendance already marked for today."
    
    # Add new attendance record
    df = pd.concat([df, pd.DataFrame({"Name": [name], "Time": [time_str]})], ignore_index=True)
    df.to_csv(attendance_file, index=False)
    
    return True, f"Attendance marked for {name} at {time_str}"

# Streamlit UI
st.title("Face Recognition Attendance System")

# Sidebar
st.sidebar.title("Options")
option = st.sidebar.selectbox("Select Option", ["Register Face", "Take Attendance", "View Attendance"])

if option == "Register Face":
    st.header("Register New Face")
    
    name = st.text_input("Enter Name or PRN Number")
    
    col1, col2 = st.columns(2)
    
    with col1:
        img_file = st.camera_input("Take a photo")
    
    with col2:
        if img_file is not None:
            # Convert the file to an opencv image
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if st.button("Register") and name:
                success, message = register_face(name, img)
                st.write(message)
                
                if success:
                    st.success(f"Face registered for {name}")
                else:
                    st.error(message)

elif option == "Take Attendance":
    st.header("Mark Attendance")
    
    img_file = st.camera_input("Take a photo to mark attendance")
    
    if img_file is not None:
        # Convert the file to an opencv image
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Display a spinner while processing
        with st.spinner("Processing..."):
            name, confidence = recognize_face(img)
            
            if name:
                st.success(f"Recognized: {name} (Confidence: {confidence:.2f})")
                
                # Mark attendance
                success, message = mark_attendance(name)
                if success:
                    st.success(message)
                else:
                    st.info(message)
            else:
                st.error("Face not recognized. Please register first or try again.")

elif option == "View Attendance":
    st.header("View Attendance Records")
    
    # Get list of attendance files
    attendance_files = [f for f in os.listdir("data/attendance") if f.endswith(".csv")]
    attendance_files.sort(reverse=True)
    
    if not attendance_files:
        st.info("No attendance records found.")
    else:
        selected_date = st.selectbox("Select Date", [file.split(".")[0] for file in attendance_files])
        
        if selected_date:
            df = pd.read_csv(f"data/attendance/{selected_date}.csv")
            st.write(f"Attendance for {selected_date}:")
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_{selected_date}.csv",
                mime="text/csv"
            )