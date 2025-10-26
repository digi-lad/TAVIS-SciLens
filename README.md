# TAVIS.SciLens

A Flask web application designed to assist visually impaired students in studying science by converting textbook images into detailed spoken descriptions using the Gemini API.

## Features

- **ArUco Marker Detection**: Automatically captures images when 4 markers are detected
- **Blur Detection**: Ensures only sharp images are processed
- **Lighting Enhancement**: Automatically improves poor lighting conditions
- **Multi-language Support**: Vietnamese, English, Chinese, Hindi, and Spanish
- **TalkBack Compatibility**: Screen reader support for visually impaired users
- **Text-to-Speech**: Automatic audio playback for non-TalkBack users
- **High Contrast UI**: Black background with yellow text for maximum visibility

## Setup

### Prerequisites

- Python 3.11+
- Gemini API Key

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python app.py
```

5. Open your browser to `http://localhost:5000`

## Deployment to Render

### Steps:

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Create Render Account**: Sign up at [render.com](https://render.com)
3. **New Web Service**: 
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
4. **Configure Settings**:
   - **Name**: `tavis-scilens` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Add Environment Variable**:
     - Key: `GEMINI_API_KEY`
     - Value: (your API key)
   - **Add Environment Variable**:
     - Key: `PORT`
     - Value: `10000` (Render's default)
5. **Deploy**: Click "Create Web Service"

### Note

The `async_mode='threading'` in `app.py` is already configured for production use with Render. The app will be accessible at `https://your-app-name.onrender.com`

## Architecture

- **Backend**: Flask with Socket.IO for real-time communication
- **Image Processing**: OpenCV for ArUco detection and image enhancement
- **AI Analysis**: Google Gemini API for generating scientific descriptions
- **Frontend**: Vanilla JavaScript with HTML5 MediaDevices API for camera access

## Usage

1. Select your language
2. Choose whether you use TalkBack
3. Tap anywhere to start camera
4. Point camera at textbook page with ArUco markers
5. App will automatically capture and analyze
6. Listen to or read the scientific description

## Testing

Run the test script to verify your setup:
```bash
python test_setup.py
```

## Generate ArUco Markers

Open `generate_aruco_markers.html` in your browser to print the required ArUco markers (IDs 0-3).
