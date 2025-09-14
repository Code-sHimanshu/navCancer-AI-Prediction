# LungScan-AI

🔬 **Revolutionizing Lung Cancer Detection with AI**

LungScan-AI is a state-of-the-art medical imaging platform that harnesses the power of deep learning to detect lung cancer with 97.03% accuracy. By analyzing chest X-rays in seconds, it provides healthcare professionals with rapid, reliable insights for early cancer detection and improved patient outcomes.

![LungScan-AI Logo](https://img.shields.io/badge/LungScan-AI-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3ZnPg==)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)


### ✨ Key Highlights
- **High Accuracy**: 97.03% accurate in detecting lung cancer variants
- **Instant Analysis**: Get comprehensive results in seconds
- **Smart Reporting**: Detailed medical reports with actionable insights
- **User-Friendly**: Intuitive interface for healthcare professionals

### ✨ Key Highlights
- **High Accuracy**: 97.03% accurate in detecting lung cancer variants
- **Instant Analysis**: Get comprehensive results in seconds
- **Smart Reporting**: Detailed medical reports with actionable insights
- **User-Friendly**: Intuitive interface for healthcare professionals

🔗 [Try the Live Demo](https://lungscan-ai.streamlit.app/)

## UI Preview

Here's a preview of the LungScan-AI application interface:

### User Registration
![User Registration](docs/ui_images/registration.png)
*User registration interface for secure access*

### Analysis Interface
![Analysis Interface](docs/ui_images/analysis.png)
*X-ray upload and analysis interface showing probability distribution*

### Detailed Results
![Detailed Results](docs/ui_images/results.png)
*Detailed analysis results with key findings and recommendations*

## Features

- Upload and analyze chest X-ray images
- Detect three types of conditions:
  - Normal
  - Squamous Cell Carcinoma
  - Adenocarcinoma
- Detailed medical report generation
- Interactive visualization of results
- User registration and history tracking

## Technical Stack

- Python 3.11+
- Streamlit for web interface
- PyTorch for deep learning (ResNet18)
- SQLite for database

## Deployment

### Local Deployment
1. Follow the installation steps above
2. Run the application using `streamlit run src/app.py`
3. Access the application at `http://localhost:8501`

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select src/app.py as the main file
5. Deploy

## Project Structure

```
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── config.py           # Configuration settings
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── train.py            # Model training script
│   └── database.py         # Database operations
├── models/
│   └── lung_cancer_detector.pth  # Trained model
├── data/
│   └── train/              # Training data directory
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Model Performance

- Accuracy: 97.03%
- Supports detection of three classes
- Based on ResNet18 architecture

## Security and Privacy

- User data is stored securely in a local database
- X-ray images are processed locally
- No data is shared with external services

## Security Notes

- Never commit `.streamlit/secrets.toml` to version control
- Use environment variables or secrets management for sensitive data
- For Streamlit Cloud deployment, configure secrets in the Streamlit Cloud dashboard
- Regularly rotate passwords and access credentials
- Monitor access logs for suspicious activity

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2024 LungScan-AI

<<<<<<< HEAD
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
