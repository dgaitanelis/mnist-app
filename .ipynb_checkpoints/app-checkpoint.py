import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
import torch.nn.functional as F
import psycopg2
from datetime import datetime
import os

# Model Path
model_path = "mnist_model.pth"

###################### PostgreSQL Setup ######################

# Database Connection Function
def connect_db():
    return psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME", "mnist_db"),
        user=os.getenv("DATABASE_USER", "postgres"),
        password=os.getenv("DATABASE_PASSWORD", "MLI_Project_Work"),
        host=os.getenv("DATABASE_HOST", "postgres_db"),  # Uses "db" inside Docker
        port=os.getenv("DATABASE_PORT", "5432")
    )

# Ensure Predictions Table Exists
def create_predictions_table():
    """Creates the predictions table if it does not exist."""
    conn = connect_db()
    cursor = conn.cursor()
    
    query = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_digit INT NOT NULL,
        correct_label INT,
        confidence NUMERIC(5,4)
    );
    """
    
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

# Log Predictions into PostgreSQL
def log_prediction(prediction, correct_label, confidence):
    """Inserts a prediction into PostgreSQL. If no feedback, correct_label = predicted digit."""
    conn = connect_db()
    cursor = conn.cursor()

    if correct_label is None:
        correct_label = prediction  

    query = """
    INSERT INTO predictions (timestamp, predicted_digit, correct_label, confidence)
    VALUES (%s, %s, %s, %s);
    """

    cursor.execute(query, (datetime.now(), prediction, correct_label, round(confidence, 4)))
    conn.commit()
    cursor.close()
    conn.close()

# Update Prediction Feedback in PostgreSQL
def update_feedback(correct_label):
    """Updates the last inserted prediction with user-provided feedback."""
    conn = connect_db()
    cursor = conn.cursor()

    query = """
    UPDATE predictions 
    SET correct_label = %s
    WHERE id = (
        SELECT id FROM predictions ORDER BY timestamp DESC LIMIT 1
    );
    """

    cursor.execute(query, (correct_label,))
    conn.commit()
    cursor.close()
    conn.close()

###################### Define Model Architecture ######################
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load Model
def load_model(model_path):
    model = MNISTModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model(model_path)

###################### Image Preprocessing ######################
transform = transforms.Compose([transforms.ToTensor()])

def predict(image):
    """Predicts the digit and its confidence."""
    image = image.convert('L')  
    image = image.resize((28, 28))
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)  
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    return prediction, confidence

###################### Streamlit UI ######################
st.title("MNIST Digit Recognizer")
st.write("Draw a digit below and the model will predict it!")

# Step 1: Ensure Table Exists
create_predictions_table()

# Step 2: Create Drawing Canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=20,
    stroke_color="white",
    background_color="black",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas"
)

###################### Process Image & Predict ######################
if st.button("Predict"):
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))  
        prediction, confidence = predict(img)

        # Store results in session state
        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        st.session_state["feedback_submitted"] = False  

        # Log prediction immediately
        log_prediction(prediction, None, confidence)  

###################### Display Prediction ######################
if "prediction" in st.session_state:
    st.write(f"**Prediction:** {st.session_state['prediction']}")
    if st.session_state['confidence'] is not None:
        st.write(f"**Confidence:** {st.session_state['confidence']:.2%}")
    else:
        st.write("**Confidence:** N/A")

    # Step 3: User Feedback
    correct_label = st.number_input("Enter the correct digit (if wrong)", min_value=0, max_value=9, step=1)

    # Step 4: Store Feedback and Update Database
    if st.button("Submit Feedback", key="feedback_button") and not st.session_state.get("feedback_submitted", False):
        update_feedback(correct_label)
        st.success("Feedback updated in PostgreSQL")
        st.session_state["feedback_submitted"] = True  

###################### Clear Canvas ######################
if st.button("Clear Canvas", key="clear_canvas"):
    st.session_state["prediction"] = None
    st.session_state["confidence"] = None
    st.session_state["feedback_submitted"] = False
    st.rerun()
