import os
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import base64
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# ArUco detector setup
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Blur detection threshold (higher = stricter)
# Typical values: 100 (very lenient), 200 (moderate), 500+ (strict)
BLUR_THRESHOLD = 200
# Lighting thresholds
MIN_BRIGHTNESS = 60  # Minimum average brightness (0-255)
MIN_CONTRAST = 30     # Minimum standard deviation (contrast)
REQUIRED_MARKER_IDS = [0, 1, 2, 3]

# Store last captured image for debugging
last_captured_image = None
last_detection_score = None

# Track active captures per client session to prevent duplicate successes
active_captures = set()

# System prompt for Gemini (unified, English base)
SYSTEM_PROMPT = """Context: You are a science teacher with the ability to transform any scientific image (chemistry, physics, biology) into a coherent and easy-to-understand spoken lecture for visually impaired students.

Task: Analyze the image and generate a spoken-word lecture that combines visual description with in-depth scientific explanation, ensuring the output is screen-reader friendly.

Thought Process (These are your internal thinking steps, not a rigid outline for the output):
1. Overview: Look at the image as a whole and identify its main topic. (e.g., "This is a diagram of the water cycle in nature," or "This is an illustration of Newton's Third Law of Motion.")
2. Identify Key Elements: Determine the important scientific components, objects, and processes in the image. DO NOT describe text labels, visual symbols, or presentation elements.
3. Analyze Relationships or Sequence:
- For a process (like an experiment or a biological cycle): Describe it step-by-step, from beginning to end, focusing only on scientific components and their relationships.
- For a static diagram (like atomic structure or a force diagram): Explain the logical, structural, or interactive relationships between the elements. NEVER describe visual elements like "a label says", "an arrow points", "a line connects", or "the image contains text". Instead, directly state the scientific relationship. For example, instead of saying "A label indicates that the sun provides energy to the plant," say directly "The sun provides energy to the plant through photosynthesis."
4. Explain the Science: This is the most crucial step. Based on what you have described, explain the core scientific principle. Naturally weave in answers to the question "Why does this happen?" and connect it to broader laws or concepts.

IMPORTANT - Avoid describing visual elements:
- NEVER mention "label", "text says", "caption", "text on the image", "name printed", or any description of text in the image.
- NEVER describe presentation elements like "arrow", "connecting line", "text box", "textbox", or how things are visually represented.
- ONLY focus on scientific content: processes, components, relationships, and scientific principles.

Output and Style Requirements (MANDATORY):
1. Opening: Always begin with the single, concise overview sentence identified in step 1.
2. Flowing Spoken Language: Do not present the information as a report or a list. Weave your thinking steps into a narrative lecture with a natural, guided flow.
3. Teacher's Tone: Be confident, certain, and clear. Avoid speculative words like "it seems," "it appears to be," or "perhaps."
4. Chemical Notation Formatting (Crucial for Screen Readers):
- Formulas: When writing chemical formulas, spell out subscript numbers linearly. Absolutely DO NOT use subscript formatting. For example: write H₂O as 'H two O', write CH₃COOH as 'C H three C O O H', write Al₂(SO₄)₃ as 'A l two S O four taken three times'.
- Physical States: Write out the full state name instead of using abbreviations in parentheses. Use 'solid' for (s), 'liquid' for (l), 'gas' for (g), and 'aqueous solution' for (aq). For example: H₂O(l) should be written as 'H two O liquid', NaCl(aq) should be written as 'N a C l aqueous solution'.
- Decisive Conclusion: End immediately after the explanation is complete. Do not ask open-ended questions or provide conversational summaries."""

# Language instruction mapping - appended to base prompt
LANGUAGE_INSTRUCTIONS = {
    'vi': 'Please respond in Vietnamese.',
    'en': 'Please respond in English.',
    'zh': 'Please respond in Chinese.',
    'hi': 'Please respond in Hindi.',
    'es': 'Please respond in Spanish.'
}


def detect_blur(image):
    """Detect if image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > BLUR_THRESHOLD


def check_lighting_quality(image):
    """Check if image has good lighting (brightness and contrast)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate average brightness
    brightness = np.mean(gray)
    
    # Calculate contrast (standard deviation)
    contrast = np.std(gray)
    
    return brightness >= MIN_BRIGHTNESS and contrast >= MIN_CONTRAST, brightness, contrast


def enhance_image(image):
    """Enhance image with poor lighting using CLAHE and brightness adjustment."""
    # Convert to LAB color space to preserve color
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Reconstruct LAB image
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    
    # Convert back to BGR
    bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Apply brightness adjustment if needed
    gray = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    if brightness < MIN_BRIGHTNESS:
        # Increase brightness
        alpha = MIN_BRIGHTNESS / brightness
        bgr_enhanced = cv2.convertScaleAbs(bgr_enhanced, alpha=alpha, beta=0)
    
    return bgr_enhanced


def validate_corner_positions(corners):
    """
    Validate that corners form proper page layout:
    - corners[0] (top-left) is left of corners[1] (top-right)
    - corners[1] (top-right) is above corners[2] (bottom-right)
    - corners[2] (bottom-right) is right of corners[3] (bottom-left)
    - corners[3] (bottom-left) is below corners[0] (top-left)
    - corners[3] (bottom-left) is left of corners[2] (bottom-right)
    - corners[0] (top-left) is above corners[3] (bottom-left)
    """
    tl = corners[0]  # top-left
    tr = corners[1]  # top-right
    br = corners[2]  # bottom-right
    bl = corners[3]  # bottom-left
    
    # Check relative positions (accounting for perspective)
    # TL is left of TR
    if tl[0] >= tr[0]:
        return False
    
    # TR is right of TL and above BR
    if tr[1] >= br[1]:
        return False
    
    # BR is right of BL
    if br[0] <= bl[0]:
        return False
    
    # BL is left of BR and below TL
    if bl[1] <= tl[1]:
        return False
    
    return True


def calculate_quad_score(corners):
    """
    Calculate a score for how well the 4 corners form a valid page boundary.
    Higher score = better combination.
    """
    if corners is None or len(corners) != 4:
        return 0.0
    
    try:
        # First validate positional constraints
        if not validate_corner_positions(corners):
            return 0.0
        
        # Check area (reasonable for a page)
        area = cv2.contourArea(corners)
        if area < 10000 or area > 10000000:  # Too small or too large
            return 0.0
        
        # Check aspect ratio (reasonable for page: ~0.7 to 1.4)
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[2] - corners[1])
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return 0.0
        
        # Check if convex quadrilateral (no self-intersection)
        hull = cv2.convexHull(corners)
        if len(hull) != 4:
            return 0.0
        
        # Check if quadrilateral is roughly rectangular
        # Compare perimeter of convex hull vs actual perimeter
        perimeter = cv2.arcLength(corners, True)
        hull_perimeter = cv2.arcLength(hull, True)
        rectangular_score = perimeter / hull_perimeter if hull_perimeter > 0 else 0
        
        # Combined score (area normalized + aspect ratio good + rectangular)
        score = (rectangular_score * 0.5) + (1.0 / aspect_ratio) * 0.5
        return score
        
    except Exception as e:
        print(f"Error calculating quad score: {e}")
        return 0.0


def extract_corners_from_markers(marker_instances):
    """
    Extract page corners from marker instances dictionary.
    Returns ordered_corners array.
    """
    ordered_corners = np.zeros((4, 2), dtype=np.float32)
    
    # Extract inner corner from each marker
    for marker_id, instance in marker_instances.items():
        marker_corners = instance['corners']  # Shape: (4, 2)
            
        # Select the inner corner based on marker position
        if marker_id == 0:  # Top-left page corner
            inner_corner = marker_corners[2]  # Marker's bottom-right
        elif marker_id == 1:  # Top-right page corner
            inner_corner = marker_corners[3]  # Marker's bottom-left
        elif marker_id == 2:  # Bottom-right page corner
            inner_corner = marker_corners[0]  # Marker's top-left
        elif marker_id == 3:  # Bottom-left page corner
            inner_corner = marker_corners[1]  # Marker's top-right
        
        # Store in correct position
        ordered_corners[marker_id] = inner_corner
    
    return ordered_corners


def try_all_combinations(marker_instances_by_id):
    """
    Try all combinations of marker instances and return the best one.
    
    Args:
        marker_instances_by_id: Dict of {id: [instances]} where each instance has 'corners'
    
    Returns:
        Best combination as dict {id: instance} or None if no valid combination
    """
    if not all(id in marker_instances_by_id and len(marker_instances_by_id[id]) > 0 
               for id in REQUIRED_MARKER_IDS):
        return None
    
    from itertools import product
    
    # Get all combinations
    combinations = list(product(
        marker_instances_by_id[0],
        marker_instances_by_id[1],
        marker_instances_by_id[2],
        marker_instances_by_id[3]
    ))
    
    best_combination = None
    best_score = 0.0
    min_valid_score = 0.1  # Require a minimum score to be considered valid
    
    for combo in combinations:
        # Create dict of {id: instance}
        marker_instances = {
            0: combo[0],
            1: combo[1],
            2: combo[2],
            3: combo[3]
        }
        
        # Extract corners
        try:
            ordered_corners = extract_corners_from_markers(marker_instances)
            
            # Score this combination
            score = calculate_quad_score(ordered_corners)
            
            if score > best_score:
                best_score = score
                best_combination = marker_instances
                
        except Exception as e:
            print(f"Error processing combination: {e}")
            continue
    
    # Only return if we found a combination with valid score
    if best_score >= min_valid_score:
        return best_combination, best_score
    else:
        return None, 0.0


def detect_aruco_markers(frame):
    """Detect ArUco markers in frame and return extracted corners and score."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) < 4:
        return False, None, 0.0
    
    # Group all detected instances by marker ID
    marker_instances_by_id = {0: [], 1: [], 2: [], 3: []}
    
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in REQUIRED_MARKER_IDS:
            marker_instances_by_id[marker_id].append({
                'corners': corners[i][0],  # Store the marker corners
                'index': i
            })
    
    # Check if we have at least one instance of each required marker
    if not all(len(instances) > 0 for instances in marker_instances_by_id.values()):
        return False, None, 0.0
    
    # If we have multiple instances, try all combinations
    if any(len(instances) > 1 for instances in marker_instances_by_id.values()):
        # Multiple instances detected - find best combination
        best_combination, score = try_all_combinations(marker_instances_by_id)
        if best_combination is None:
            return False, None, 0.0
        marker_instances = best_combination
    else:
        # Only one instance of each marker - use them
        marker_instances = {
            id: instances[0] 
            for id, instances in marker_instances_by_id.items()
        }
    
    # Extract corners from the selected combination
    ordered_corners = extract_corners_from_markers(marker_instances)
    
    # Calculate score for validation
    score = calculate_quad_score(ordered_corners)
    
    # Validate score (same threshold as try_all_combinations)
    min_valid_score = 0.1
    if score >= min_valid_score:
        return True, ordered_corners, score
    else:
        return False, None, 0.0


def extract_image(frame, corners):
    """Extract and warp the image region defined by ArUco corners."""
    # corners has shape (4, 2) - 4 corner points with 2 coordinates each
    # Convert to required format for getPerspectiveTransform
    src_pts = corners.reshape(1, 4, 2).astype(np.float32)
    
    # Calculate the actual width and height from the bounding box
    # to maintain the aspect ratio
    top_left = corners[0]
    top_right = corners[1]
    bottom_right = corners[2]
    bottom_left = corners[3]
    
    # Calculate the actual dimensions from the corners
    width1 = np.linalg.norm(top_right - top_left)
    width2 = np.linalg.norm(bottom_right - bottom_left)
    height1 = np.linalg.norm(bottom_left - top_left)
    height2 = np.linalg.norm(bottom_right - top_right)
    
    # Use average to get more accurate dimensions
    width = int((width1 + width2) / 2)
    height = int((height1 + height2) / 2)
    
    # Define destination points
    dst_pts = np.array([
        [[0, 0]],           # top-left
        [[width, 0]],       # top-right
        [[width, height]], # bottom-right
        [[0, height]]       # bottom-left
    ], dtype=np.float32)
    
    # Compute perspective transform
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (width, height))
    
    return warped


def split_into_sentences(text):
    """Split text into sentences for screen reader compatibility."""
    # Split by sentence-ending punctuation
    sentences = re.split(r'([.!?]\s+)', text)
    result = []
    
    # Clean up and pair punctuation with sentences
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            sentence = sentence.strip()
            if sentence:
                result.append(sentence)
    
    return result if result else [text]


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('process_frame')
def handle_frame(data):
    """Process incoming frame for ArUco detection (TalkBack) or direct capture (non-TalkBack)."""
    global last_captured_image, last_detection_score
    
    try:
        # Prevent duplicate processing after a successful capture for this client
        sid = request.sid
        if sid in active_captures:
            return
        # Decode base64 image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get TalkBack flag (default True for backward compatibility)
        uses_talkback = data.get('talkback', True)
        
        if uses_talkback:
            # TalkBack mode: Full ArUco detection and processing
            detected, corners, score = detect_aruco_markers(frame)
            
            if detected:
                # Extract the image region first
                extracted_image = extract_image(frame, corners)
                
                # Check and enhance lighting if needed
                has_good_lighting, brightness, contrast = check_lighting_quality(extracted_image)
                if not has_good_lighting:
                    print(f"Poor lighting detected - Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
                    extracted_image = enhance_image(extracted_image)
                    print("Image enhanced")
                
                # Check if extracted image is sharp enough
                is_sharp = detect_blur(extracted_image)
                
                if is_sharp:
                    # Store for debugging
                    last_captured_image = extracted_image.copy()
                    last_detection_score = score
                    print(f"Detection score: {score:.3f}")
                    
                    # Also save to file
                    cv2.imwrite('static/last_captured.jpg', extracted_image)
                    
                    # Encode extracted image for saving
                    _, buffer = cv2.imencode('.jpg', extracted_image)
                    image_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Mark this session as having an active capture
                    active_captures.add(sid)
                    emit('capture_success', {'success': True, 'image': image_base64})
                else:
                    emit('capture_failed', {'reason': 'blurry'})
            else:
                # Markers not found, continue processing
                pass
        else:
            # Non-TalkBack mode: Skip ArUco detection, use image directly
            # Just perform basic blur check
            is_sharp = detect_blur(frame)
            
            if is_sharp:
                # Store for debugging
                last_captured_image = frame.copy()
                last_detection_score = None
                
                # Save to file
                cv2.imwrite('static/last_captured.jpg', frame)
                
                # Encode original frame
                _, buffer = cv2.imencode('.jpg', frame)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Mark this session as having an active capture
                active_captures.add(sid)
                emit('capture_success', {'success': True, 'image': image_base64})
            else:
                emit('capture_failed', {'reason': 'blurry'})
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})


@socketio.on('reset_capture')
def reset_capture():
    """Allow the client to start a new capture by clearing the guard."""
    try:
        sid = request.sid
        if sid in active_captures:
            active_captures.discard(sid)
    except Exception as e:
        print(f"Error resetting capture state: {e}")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/last_captured')
def last_captured():
    """Debug endpoint to view the last captured image with score."""
    try:
        from flask import send_file
        from flask import make_response
        import os
        
        # Check if image exists
        if not os.path.exists('static/last_captured.jpg'):
            return "No image captured yet", 404
        
        # If score exists, add it to the image
        global last_detection_score
        if last_detection_score is not None:
            # Read the image
            img = cv2.imread('static/last_captured.jpg')
            if img is not None:
                # Add score text to image
                score_text = f"Detection Score: {last_detection_score:.3f}"
                cv2.putText(img, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (0, 255, 0), 2)
                cv2.imwrite('static/last_captured_debug.jpg', img)
                return send_file('static/last_captured_debug.jpg', mimetype='image/jpeg')
        
        return send_file('static/last_captured.jpg', mimetype='image/jpeg')
    except Exception as e:
        return f"Error: {str(e)}", 500


@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze extracted image with Gemini API."""
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        language = data.get('language', 'vi')  # Default to Vietnamese
        
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        
        # Create image part for Gemini (using the correct API)
        import PIL.Image as PILImage
        from io import BytesIO
        
        image = PILImage.open(BytesIO(image_bytes))
        
        # Use unified English prompt and append language instruction
        language_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS['en'])
        prompt = SYSTEM_PROMPT + '\n\n' + language_instruction
        
        # Generate content with Gemini
        response = model.generate_content([prompt, image])
        
        # Parse response into sentences
        description_text = response.text
        sentences = split_into_sentences(description_text)
        
        return jsonify({'sentences': sentences})
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # For local development
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    socketio.run(app, debug=debug, host='0.0.0.0', port=port)
else:
    # For production (gunicorn/other WSGI servers)
    application = app

