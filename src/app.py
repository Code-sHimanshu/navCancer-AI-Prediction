import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
from datetime import datetime
import config
from database import Database
import re
import requests
from scipy import ndimage

# Initialize database
db = Database()

# Page configuration
st.set_page_config(
    page_title="LungScan-AI",
    page_icon="🫁",
    layout="wide"
)

# Add this after the imports
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Add this before any other content
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    if st.button("🌓" if st.session_state.dark_mode else "🌞"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# Update the CSS to include dark mode
st.markdown(f"""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #FFFFFF;
    }
    [data-testid="stSidebar"] {
        background-color: #ffd166;
    }
    .upload-box {
        border: 2px dashed #FFFCFB;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-box {
        background-color: #ffd500;
        padding: 20px;
        border-radius: 10px;
        border-color: #ffd166;
        box-shadow: 4px 2px 4px rgba(0,3,4,0.7);
        margin: 20px 0;
    }
    .metric-box {
        background-color: #ffd500;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .user-form {
        background-color: #ffd500;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .logo-text {
        font-size: 3em;
        font-weight: 700;
        background: linear-gradient(45deg, #ffd166, #ffd500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0 0;
        font-family: 'Arial Black', sans-serif;
    }
    .logo-subtext {
        text-align: center;
        color: #b68b00;
        font-size: 4em;
        margin-bottom: 30px;
    }
    .consent-popup {
        background-color: #F4F754;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px auto;
        max-width: 800px;
        text-align: center;
    }
    .consent-text {
        font-size: 1.1em;
        line-height: 1.6;
        color: #ffffff;
        margin-bottom: 25px;
        text-align: justify;
    }
    .restricted-message {
        background-color: #ffd500;
        color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if 'user_consent_given' not in st.session_state:
    st.session_state.user_consent_given = None
if 'user_registered' not in st.session_state:
    st.session_state.user_registered = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'show_consent_popup' not in st.session_state:
    st.session_state.show_consent_popup = True

# Logo and Title
# st.markdown('<h1 class="logo-text">LungScan-AI</h1>', unsafe_allow_html=True)
st.markdown('<h1 class="logo-subtext">Advanced Lung Cancer Detection Using Artificial Intelligence</h1>', unsafe_allow_html=True)

def download_model():
    """Download the model file if it doesn't exist"""
    model_path = 'models/lung_cancer_detector.pth'
    if not os.path.exists(model_path):
        os.makedirs('models', exist_ok=True)
        # You need to replace this URL with your actual model file URL
        url = st.secrets.get("MODEL_URL", "")
        if not url:
            st.error("Model URL not configured. Please contact administrator.")
            st.stop()
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(response.content)
            st.success("Model downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading model: {str(e)}")
            st.stop()

@st.cache_resource
def load_model():
    """Load the trained model"""
    # Ensure model file exists
    download_model()
    
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)
    
    model.load_state_dict(torch.load('models/lung_cancer_detector.pth'))
    model = model.to(config.DEVICE)
    model.eval()
    return model

def is_valid_xray(image):
    """
    Validate if the uploaded image appears to be a chest X-ray
    Returns: (bool, str) - (is_valid, message)
    """
    try:
        # Convert to grayscale and get pixel values
        gray_img = image.convert('L')
        pixel_values = np.array(gray_img)
        
        # Check image characteristics typical of X-rays
        mean_intensity = np.mean(pixel_values)
        std_intensity = np.std(pixel_values)
        
        # Calculate histogram features
        hist, _ = np.histogram(pixel_values, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize histogram
        
        # X-ray specific checks:
        
        # 1. Check for bimodal distribution (typical in X-rays)
        peaks = []
        for i in range(1, 255):
            if hist[i-1] < hist[i] and hist[i] > hist[i+1]:
                peaks.append(i)
        if len(peaks) < 2:
            return False, "Image does not show typical X-ray intensity distribution."
            
        # 2. Check contrast and dynamic range
        if std_intensity < 45:  # Increased threshold for contrast
            return False, "Image has insufficient contrast for an X-ray."
            
        # 3. Check intensity range (X-rays have specific brightness characteristics)
        if mean_intensity < 100 or mean_intensity > 200:
            return False, "Image brightness is not in the typical range for X-rays."
            
        # 4. Check for color variation (X-rays are fundamentally grayscale)
        rgb_image = image.convert('RGB')
        r, g, b = rgb_image.split()
        r_arr = np.array(r)
        g_arr = np.array(g)
        b_arr = np.array(b)
        
        # Calculate color channel correlations
        rg_corr = np.corrcoef(r_arr.flat, g_arr.flat)[0,1]
        rb_corr = np.corrcoef(r_arr.flat, b_arr.flat)[0,1]
        gb_corr = np.corrcoef(g_arr.flat, b_arr.flat)[0,1]
        
        # X-rays should have very high correlation between channels (almost identical)
        if not (rg_corr > 0.98 and rb_corr > 0.98 and gb_corr > 0.98):
            return False, "Image contains significant color variation, which is not typical for X-rays."
            
        # 5. Check for typical X-ray edge characteristics
        edges = ndimage.sobel(pixel_values)
        edge_intensity = np.mean(np.abs(edges))
        if edge_intensity < 10:  # Threshold for edge detection
            return False, "Image lacks the characteristic edge patterns of chest X-rays."
            
        # 6. Check image size (X-rays typically have specific dimensions)
        width, height = image.size
        aspect_ratio = width / height
        if not (0.7 <= aspect_ratio <= 1.5):  # Typical chest X-ray aspect ratios
            return False, "Image dimensions are not typical for chest X-rays."
            
        return True, "Valid X-ray image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

def preprocess_image(image):
    """Preprocess the image for model input"""
    # First validate if it's an X-ray image
    is_valid, message = is_valid_xray(image)
    if not is_valid:
        st.error(message)
        return None
        
    transform = transforms.Compose([
        transforms.Resize((config.INPUT_SIZE, config.INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])
    
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

def predict_image(model, input_tensor):
    """Make prediction for a single image"""
    if input_tensor is None:
        return None, None, None
        
    with torch.no_grad():
        output = model(input_tensor.to(config.DEVICE))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    return predicted_class, confidence, probabilities

def plot_probabilities(probabilities, class_names):
    """Create probability plot"""
    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(class_names))
    
    # Plot horizontal bars
    bars = ax.barh(y_pos, probabilities.cpu().numpy())
    
    # Customize colors based on probabilities
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.RdYlGn(probabilities[i].item()))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    # Add percentage labels on bars
    for i, v in enumerate(probabilities):
        ax.text(v.item(), i, f'{v.item():.1%}', va='center')
    
    plt.tight_layout()
    return fig

def generate_detailed_report(predicted_class, confidence, probabilities, class_names):
    """Generate a detailed medical report"""
    diagnosis = class_names[predicted_class]
    medical_info = config.MEDICAL_INFO[diagnosis]
    
    report = f"""
LUNG CANCER DETECTION REPORT
{'=' * 50}

[WARNING] DISCLAIMER: This is an AI generated prediction. Please consult a specialist before taking any step further.
{'=' * 50}

Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DIAGNOSIS
--------
Primary Diagnosis: {diagnosis}
Confidence Level: {confidence:.1%}

DETAILED ANALYSIS
---------------
Description: {medical_info['description']}

Probability Distribution:
{'-' * 30}"""
    
    for i, (name, prob) in enumerate(zip(class_names, probabilities)):
        report += f"\n{name}: {prob.item():.1%}"

    if diagnosis != 'Normal':
        report += f"""

CHARACTERISTICS
-------------
The following characteristics are typical for {diagnosis}:
"""
        for char in medical_info['characteristics']:
            report += f"- {char}\n"

    report += f"""

RISK FACTORS
-----------
Key risk factors to consider:
"""
    for risk in medical_info['risk_factors']:
        report += f"- {risk}\n"

    report += f"""

RECOMMENDATIONS
-------------
Based on the analysis, the following actions are recommended:
"""
    for rec in medical_info['recommendations']:
        report += f"- {rec}\n"

    report += f"""

FOLLOW-UP
--------
{medical_info['follow_up']}

IMPORTANT NOTES
-------------
1. This analysis is based on AI interpretation and should be confirmed by a qualified healthcare professional.
2. The confidence level indicates the AI model's certainty but should not be used as the sole diagnostic criterion.
3. Additional imaging studies or tests may be required for definitive diagnosis.
4. Early detection and treatment are crucial for better outcomes.

DISCLAIMER
---------
This report is generated by an AI system and is intended to assist healthcare professionals. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions regarding a medical condition.

Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')}
"""
    return report

def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """Validate phone number format"""
    pattern = r'^\+?1?\d{9,15}$'
    return re.match(pattern, phone) is not None

def save_prediction(image, prediction_info):
    """Save the prediction results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = "predictions"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save image
    image_path = os.path.join(save_dir, f"prediction_{timestamp}.jpg")
    image.save(image_path)
    
    # Save prediction info
    info_path = os.path.join(save_dir, f"prediction_{timestamp}.txt")
    with open(info_path, "w") as f:
        f.write(prediction_info)
    
    return image_path, info_path

def main():
    # Show consent popup if not already handled
    if st.session_state.show_consent_popup:
        st.markdown("""
        <style>
            .consent-modal {
                background-color: #b68b00; #dark green
                border: 4px solid #c1121f; #red
                border-radius: 8px;
                padding: 2rem;
                margin: 1rem 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            .consent-title {
                color: #000000; #black
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 1.2rem;
            }
            .consent-content {
                color: #edf6f9;
                line-height: 1.6;
                margin-bottom: 1.5rem;
            }
            .consent-list {
                margin: 0.8rem 0;
                padding-left: 1.5rem;
            }
            .consent-button {
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
        </style>
        
        <div class="consent-modal">
            <div class="consent-title">Professional Review Service Authorization</div>
            <div class="consent-content">
                <p>To enhance your diagnostic experience, we offer:</p>
                <ul class="consent-list">
                    <li>Complementary secondary evaluation by domain experts</li>
                    <li>Detailed diagnostic report with actionable insights</li>
                    <li>Personalized decision-support framework</li>
                </ul>
                <p>By proceeding, you agree to provide basic demographic information necessary for report generation.</p>
                <div class="consent-footer">
                    All data is protected under HIPAA compliance standards and used solely for diagnostic purposes.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 2])
        with col2:
            if st.button("✓ I Agree", 
                        use_container_width=True,
                        key="agree_btn",
                        type="primary",
                        help="Provide consent for expert review and report generation"):
                st.session_state.user_consent_given = True
                st.session_state.show_consent_popup = False
                st.rerun()
        with col3:
            if st.button("✗ Decline", 
                        use_container_width=True,
                        key="disagree_btn",
                        type="secondary",
                        help="Continue without expert review and report features"):
                st.session_state.user_consent_given = False
                st.session_state.show_consent_popup = False
                st.rerun()
        
        return
    # Header
    st.title("🫁 Lung Cancer Detection System")
    st.markdown("---")
    
    # Show registration form only if user agreed
    if st.session_state.user_consent_given and not st.session_state.user_registered:
        st.markdown("### User Information")
        with st.form("user_form"):
            name = st.text_input("Full Name*")
            email = st.text_input("Email Address*")
            phone = st.text_input("Phone Number*")
            address = st.text_area("Address*")
            
            if st.form_submit_button("Submit"):
                if not all([name, email, phone, address]):
                    st.error("All fields are required!")
                elif not validate_email(email):
                    st.error("Please enter a valid email address!")
                elif not validate_phone(phone):
                    st.error("Please enter a valid phone number!")
                else:
                    user_id = db.add_user(name, email, phone, address)
                    if user_id:
                        st.session_state.user_registered = True
                        st.session_state.user_id = user_id
                        st.success("Information submitted successfully!")
                        st.rerun()
    
    # Main content - show to all users but with restrictions for those who disagreed
    if st.session_state.user_registered or not st.session_state.user_consent_given:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload X-ray Image")
            uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Validate and make prediction
                model = load_model()
                input_tensor = preprocess_image(image)
                
                if input_tensor is not None:
                    predicted_class, confidence, probabilities = predict_image(model, input_tensor)
                    class_names = list(config.CLASS_NAMES.values())
                    
                    # Display prediction
                    prediction_box = f"""
                    <div class='prediction-box'>
                        <h3>Diagnosis</h3>
                        <p style='font-size: 24px; color: {'red' if predicted_class != 0 else 'green'};'>
                            {class_names[predicted_class]}
                        </p>
                        <p style='font-size: 18px;'>Confidence: {confidence:.1%}</p>
                    </div>
                    """
                    st.markdown(prediction_box, unsafe_allow_html=True)
                    
                    # Display disclaimer
                    st.warning("⚠️ This is an AI generated prediction. Please consult a specialist before taking any step further.")
                    
                    # Display probability plot
                    st.markdown("<h3 style='color: #720026;'>Probability Distribution</h3>", unsafe_allow_html=True)
                    prob_fig = plot_probabilities(probabilities, class_names)
                    st.pyplot(prob_fig)
                    
                    # Generate detailed report
                    detailed_report = generate_detailed_report(
                        predicted_class, 
                        confidence, 
                        probabilities, 
                        class_names
                    )
                    
                    # Save results
                    image_path, info_path = save_prediction(image, detailed_report)
                    
                    # Save prediction to database
                    db.add_prediction(
                        st.session_state.user_id,
                        image_path,
                        class_names[predicted_class],
                        confidence,
                        info_path
                    )
                    
                    # Display key findings
                    st.markdown("### Key Findings")
                    with st.expander("View Detailed Analysis", expanded=True):
                        diagnosis = class_names[predicted_class]
                        medical_info = config.MEDICAL_INFO[diagnosis]
                        
                        st.markdown(f"**Description:**")
                        st.write(medical_info['description'])
                        
                        if diagnosis != 'Normal':
                            st.markdown(f"**Characteristics:**")
                            for char in medical_info['characteristics']:
                                st.write(f"- {char}")
                        
                        st.markdown(f"**Recommendations:**")
                        for rec in medical_info['recommendations']:
                            st.write(f"- {rec}")
                    
                    # Download buttons
                    st.markdown("### Download Results")
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        with open(info_path, 'r') as f:
                            st.download_button(
                                label="📄 Download Detailed Report",
                                data=f.read(),
                                file_name=f"lung_cancer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    with col_btn2:
                        with open(image_path, 'rb') as f:
                            st.download_button(
                                label="🖼️ Download Image",
                                data=f.read(),
                                file_name=f"xray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                                mime="image/jpeg"
                            )
                else:
                    st.markdown("""
                    <div class="restricted-message" style="text-align: left;">
                        ⚠️ Fill in your details to see the complete report and download options.
                        <br>This includes :
                        <br>
                        <br>• Detailed probability distribution
                        <br>• Key findings and characteristics
                        <br>• Downloadable detailed report
                        <br>• Expert review of your case
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("Provide Details Now"):
                        st.session_state.user_consent_given = True
                        st.session_state.show_consent_popup = True
                        st.rerun()

if __name__ == "__main__":
    main()