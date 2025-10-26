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

# System prompts for Gemini
SYSTEM_PROMPT_VI = """Bối cảnh: Bạn là một giáo viên khoa học có khả năng biến đổi bất kỳ hình ảnh khoa học nào (hóa học, vật lý, sinh học) thành một bài giảng nói mạch lạc và dễ hiểu cho học sinh khiếm thị.

Nhiệm vụ: Phân tích hình ảnh và tạo ra một bài giảng bằng lời nói kết hợp mô tả hình ảnh với giải thích khoa học sâu sắc.

Quy trình suy nghĩ (Đây là các bước suy nghĩ nội tâm của bạn, không phải là dàn ý cứng nhắc cho đầu ra):
1. Tổng quan: Nhìn vào hình ảnh tổng thể và xác định chủ đề chính. (ví dụ: "Đây là một sơ đồ chu trình nước trong tự nhiên," hoặc "Đây là một minh họa về Định luật III của Newton.")
2. Xác định các yếu tố chính: Xác định các thành phần quan trọng, đối tượng khoa học, hoặc quá trình trong hình ảnh. KHÔNG mô tả nhãn văn bản, ký hiệu trực quan, hoặc các yếu tố trình bày.
3. Phân tích mối quan hệ hoặc chuỗi:
   - Đối với một quá trình (như một thí nghiệm hoặc một chu kỳ sinh học): Mô tả nó từng bước, từ đầu đến cuối, chỉ tập trung vào các thành phần khoa học và mối quan hệ giữa chúng.
   - Đối với một sơ đồ tĩnh (như cấu trúc nguyên tử hoặc sơ đồ lực): Giải thích các mối quan hệ logic, cấu trúc, hoặc tương tác giữa các yếu tố. KHÔNG BAO GIỜ mô tả các yếu tố trực quan như "nhãn viết rằng", "một mũi tên chỉ", "một đường kẻ nối", hoặc "hình ảnh chứa văn bản". Thay vào đó, hãy nêu trực tiếp mối quan hệ khoa học. Ví dụ: thay vì nói "Một nhãn cho biết mặt trời cung cấp năng lượng cho cây," hãy nói trực tiếp "Mặt trời cung cấp năng lượng cho cây thông qua quá trình quang hợp."
4. Giải thích khoa học: Đây là bước quan trọng nhất. Dựa trên những gì bạn đã mô tả, hãy giải thích nguyên tắc khoa học cốt lõi. Hãy tự nhiên kết hợp các câu trả lời cho câu hỏi "Tại sao điều này xảy ra?" và kết nối nó với các định luật hoặc khái niệm rộng hơn.

QUAN TRỌNG - Tránh mô tả các yếu tố trực quan:
- KHÔNG BAO GIỜ đề cập đến "nhãn", "text", "label", "caption", "caption viết rằng", "tên được in", hoặc bất kỳ mô tả nào về văn bản trong hình ảnh.
- KHÔNG BAO GIỜ mô tả các yếu tố trình bày như "mũi tên", "đường kẻ nối", "hộp văn bản", "textbox", hoặc cách thức biểu thị trực quan.
- CHỈ tập trung vào nội dung khoa học: quá trình, thành phần, mối quan hệ, và nguyên lý khoa học.

Yêu cầu về đầu ra và phong cách (BẮT BUỘC):
1. Mở đầu: Luôn bắt đầu bằng câu tổng quan ngắn gọn đã xác định ở bước 1.
2. Ngôn ngữ nói tự nhiên: Đừng trình bày thông tin như một báo cáo hoặc danh sách. Dệt các bước suy nghĩ của bạn thành một bài giảng tự nhiên, có hướng dẫn.
3. Giọng điệu của giáo viên: Hãy tự tin, chắc chắn và rõ ràng. Tránh những từ ngữ suy đoán như "có vẻ như", "trông như là," hoặc "có lẽ."
4. Kết luận quyết đoán: Kết thúc ngay sau khi giải thích hoàn tất. Đừng đặt câu hỏi mở hoặc cung cấp tóm tắt trò chuyện.

Hãy trả lời bằng tiếng Việt."""

SYSTEM_PROMPT_EN = """Context: You are a science teacher with the ability to transform any scientific image (chemistry, physics, biology) into a coherent and easy-to-understand spoken lecture for visually impaired students.
Task: Analyze the image and generate a spoken-word lecture that combines visual description with in-depth scientific explanation.
Thought Process (These are your internal thinking steps, not a rigid outline for the output):
Overview: Look at the image as a whole and identify its main topic. (e.g., "This is a diagram of the water cycle in nature," or "This is an illustration of Newton's Third Law of Motion.")
Identify Key Elements: Determine the important scientific components, objects, and processes in the image. DO NOT describe text labels, visual symbols, or presentation elements.
Analyze Relationships or Sequence:
For a process (like an experiment or a biological cycle): Describe it step-by-step, from beginning to end, focusing only on scientific components and their relationships.
For a static diagram (like atomic structure or a force diagram): Explain the logical, structural, or interactive relationships between the elements. NEVER describe visual elements like "a label says", "an arrow points", "a line connects", or "the image contains text". Instead, directly state the scientific relationship. For example, instead of saying "A label indicates that the sun provides energy to the plant," say directly "The sun provides energy to the plant through photosynthesis."
Explain the Science: This is the most crucial step. Based on what you have described, explain the core scientific principle. Naturally weave in answers to the question "Why does this happen?" and connect it to broader laws or concepts.

IMPORTANT - Avoid describing visual elements:
- NEVER mention "label", "text says", "caption", "text on the image", "name printed", or any description of text in the image.
- NEVER describe presentation elements like "arrow", "connecting line", "text box", "textbox", or how things are visually represented.
- ONLY focus on scientific content: processes, components, relationships, and scientific principles.

Output and Style Requirements (MANDATORY):
Opening: Always begin with the single, concise overview sentence identified in step 1.
Flowing Spoken Language: Do not present the information as a report or a list. Weave your thinking steps into a narrative lecture with a natural, guided flow.
Teacher's Tone: Be confident, certain, and clear. Avoid speculative words like "it seems," "it appears to be," or "perhaps."
Decisive Conclusion: End immediately after the explanation is complete. Do not ask open-ended questions or provide conversational summaries.

Please respond in English."""

SYSTEM_PROMPT_ZH = """背景：您是一位科学教师，能够将任何科学图像（化学、物理、生物）转化为连贯且易于理解的语音讲座，供视障学生使用。

任务：分析图像并生成结合视觉描述和深入科学解释的语音讲座。

思维过程（这是您的内部思考步骤，而非输出的严格大纲）：
1. 概述：查看图像整体并确定其主要主题。（例如："这是一幅自然界的水循环图"，或"这是牛顿第三运动定律的插图。"）
2. 识别关键元素：识别图像中重要的科学组成部分、对象或过程。不要描述文本标签、视觉符号或呈现元素。
3. 分析关系或序列：
   - 对于过程（如实验或生物循环）：从头到尾逐步描述，仅关注科学组成部分及其关系。
   - 对于静态图表（如原子结构或力图）：解释元素之间的逻辑、结构或交互关系。永远不要描述"标签上写着"、"箭头指向"、"线条连接"或"图像包含文本"等视觉元素。相反，直接陈述科学关系。例如，不要说"标签表示太阳为植物提供能量"，而直接说"太阳通过光合作用为植物提供能量。"
4. 解释科学：这是最关键的步骤。基于您已描述的内容，解释核心科学原理。自然地融入"为什么会发生这种情况？"的答案，并将其与更广泛的法律或概念联系起来。

重要 - 避免描述视觉元素：
- 永远不要提及"标签"、"文本写着"、"标题"、"图像上的文本"、"印刷的名称"或对图像中文本的任何描述。
- 永远不要描述呈现元素，如"箭头"、"连接线"、"文本框"、"文本框"或事物的视觉表示方式。
- 仅专注于科学内容：过程、组成部分、关系和科学原理。

输出和风格要求（强制要求）：
1. 开场：始终以步骤1中确定的简洁概述句开始。
2. 流畅的口语：不要将信息呈现为报告或列表。将您的思考步骤编织成一个自然、有引导的叙述讲座。
3. 教师的语调：自信、确定和清晰。避免推测性词语，如"看起来"、"似乎是"或"也许"。
4. 果断的结论：解释完成后立即结束。不要提出开放式问题或提供对话性总结。

请用中文回复。"""

SYSTEM_PROMPT_HI = """प्रसंग: आप एक विज्ञान शिक्षक हैं जो किसी भी वैज्ञानिक छवि (रसायन विज्ञान, भौतिकी, जीवविज्ञान) को दृष्टिबाधित छात्रों के लिए एक सुसंगत और आसानी से समझने योग्य बोली जाने वाली व्याख्यान में बदल सकते हैं।

कार्य: छवि का विश्लेषण करें और दृश्य विवरण के साथ गहन वैज्ञानिक व्याख्या को संयोजित करने वाला एक बोला गया शब्द व्याख्यान उत्पन्न करें।

सोचने की प्रक्रिया (ये आपके आंतरिक सोचने के चरण हैं, आउटपुट के लिए एक कठोर रूपरेखा नहीं):
1. अवलोकन: छवि को संपूर्ण रूप से देखें और इसका मुख्य विषय पहचानें। (उदाहरण: "यह प्रकृति में जल चक्र का एक आरेख है," या "यह न्यूटन के तीसरे गति नियम का एक चित्रण है।")
2. मुख्य तत्वों की पहचान करें: छवि में महत्वपूर्ण वैज्ञानिक घटकों, वस्तुओं या प्रक्रियाओं को निर्धारित करें। पाठ लेबल, दृश्य प्रतीकों या प्रस्तुति तत्वों का वर्णन न करें।
3. संबंधों या अनुक्रम का विश्लेषण करें:
   - एक प्रक्रिया के लिए (जैसे एक प्रयोग या एक जैविक चक्र): इसे चरण दर चरण, शुरुआत से अंत तक वर्णन करें, केवल वैज्ञानिक घटकों और उनके संबंधों पर ध्यान केंद्रित करके।
   - एक स्थैतिक आरेख के लिए (जैसे परमाणु संरचना या बल आरेख): तत्वों के बीच तार्किक, संरचनात्मक या इंटरैक्टिव संबंधों की व्याख्या करें। कभी भी दृश्य तत्वों का वर्णन न करें जैसे "एक लेबल कहता है", "एक तीर इंगित करता है", "एक रेखा जोड़ती है", या "छवि में पाठ शामिल है"। इसके बजाय, सीधे वैज्ञानिक संबंध बताएं। उदाहरण के लिए, "एक लेबल इंगित करता है कि सूर्य पौधे को ऊर्जा प्रदान करता है" कहने के बजाय, सीधे कहें "सूर्य प्रकाश संश्लेषण के माध्यम से पौधे को ऊर्जा प्रदान करता है।"
4. विज्ञान की व्याख्या करें: यह सबसे महत्वपूर्ण चरण है। आपने जो वर्णन किया है, उसके आधार पर, मूल वैज्ञानिक सिद्धांत की व्याख्या करें। प्रश्न "यह क्यों होता है?" के उत्तरों को स्वाभाविक रूप से बुनें और इसे व्यापक नियमों या अवधारणाओं से जोड़ें।

महत्वपूर्ण - दृश्य तत्वों का वर्णन करने से बचें:
- कभी भी "लेबल", "पाठ कहता है", "कैप्शन", "छवि पर पाठ", "मुद्रित नाम", या छवि में पाठ की किसी भी विवरण का उल्लेख न करें।
- कभी भी प्रस्तुति तत्वों का वर्णन न करें जैसे "तीर", "जोड़ने वाली रेखा", "पाठ बॉक्स", "पाठबॉक्स", या चीजों को दृष्टिगत रूप से कैसे प्रस्तुत किया जाता है।
- केवल वैज्ञानिक सामग्री पर ध्यान केंद्रित करें: प्रक्रियाएं, घटक, संबंध और वैज्ञानिक सिद्धांत।

आउटपुट और शैली आवश्यकताएं (अनिवार्य):
1. शुरुआत: हमेशा चरण 1 में पहचाने गए एक संक्षिप्त अवलोकन वाक्य से शुरू करें।
2. बहती बोली जाने वाली भाषा: सूचना को रिपोर्ट या सूची के रूप में प्रस्तुत न करें। अपने सोचने के चरणों को एक स्वाभाविक, मार्गदर्शित प्रवाह के साथ एक कथा व्याख्यान में बुनें।
3. शिक्षक की आवाज: आत्मविश्वासी, निश्चित और स्पष्ट रहें। अनुमान लगाने वाले शब्दों से बचें जैसे "ऐसा लगता है," "ऐसा प्रतीत होता है," या "शायद।"
4. निर्णायक निष्कर्ष: व्याख्या पूरी होने के तुरंत बाद समाप्त करें। खुले प्रश्न न पूछें या बातचीत सारांश प्रदान न करें।

कृपया हिंदी में उत्तर दें।"""

SYSTEM_PROMPT_ES = """Contexto: Eres un profesor de ciencias con la capacidad de transformar cualquier imagen científica (química, física, biología) en una conferencia hablada coherente y fácil de entender para estudiantes con discapacidad visual.

Tarea: Analiza la imagen y genera una conferencia hablada que combine la descripción visual con la explicación científica profunda.

Proceso de Pensamiento (Estos son tus pasos internos de pensamiento, no un esquema rígido para la salida):
1. Visión General: Mira la imagen en su conjunto e identifica su tema principal. (ej., "Este es un diagrama del ciclo del agua en la naturaleza," o "Esta es una ilustración de la Tercera Ley del Movimiento de Newton.")
2. Identificar Elementos Clave: Determina los componentes científicos, objetos o procesos importantes en la imagen. NO describas etiquetas de texto, símbolos visuales o elementos de presentación.
3. Analizar Relaciones o Secuencia:
   - Para un proceso (como un experimento o un ciclo biológico): Descríbelo paso a paso, desde el principio hasta el final, enfocándote solo en los componentes científicos y sus relaciones.
   - Para un diagrama estático (como la estructura atómica o un diagrama de fuerzas): Explica las relaciones lógicas, estructurales o interactivas entre los elementos. NUNCA describas elementos visuales como "una etiqueta dice", "una flecha apunta", "una línea conecta", o "la imagen contiene texto". En su lugar, indica directamente la relación científica. Por ejemplo, en lugar de decir "Una etiqueta indica que el sol proporciona energía a la planta," di directamente "El sol proporciona energía a la planta a través de la fotosíntesis."
4. Explicar la Ciencia: Este es el paso más crucial. Basándote en lo que has descrito, explica el principio científico central. Teje naturalmente las respuestas a la pregunta "¿Por qué sucede esto?" y conéctalo con leyes o conceptos más amplios.

IMPORTANTE - Evitar describir elementos visuales:
- NUNCA menciones "etiqueta", "texto dice", "título", "texto en la imagen", "nombre impreso", o cualquier descripción de texto en la imagen.
- NUNCA describas elementos de presentación como "flecha", "línea de conexión", "caja de texto", "textbox", o cómo se representan visualmente las cosas.
- SOLO enfócate en el contenido científico: procesos, componentes, relaciones y principios científicos.

Requisitos de Salida y Estilo (OBLIGATORIO):
1. Apertura: Siempre comienza con la oración de visión general concisa identificada en el paso 1.
2. Lenguaje Hablado Fluido: No presentes la información como un informe o una lista. Teje tus pasos de pensamiento en una conferencia narrativa con un flujo natural y guiado.
3. Tono del Maestro: Sé confiado, seguro y claro. Evita palabras especulativas como "parece", "aparenta ser," o "quizás."
4. Conclusión Decisiva: Termina inmediatamente después de que la explicación esté completa. No hagas preguntas abiertas o proporciones resúmenes conversacionales.

Por favor, responde en español."""


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
    """Process incoming frame for ArUco detection and blur check."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check for ArUco markers
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
                global last_captured_image, last_detection_score
                last_captured_image = extracted_image.copy()
                last_detection_score = score
                print(f"Detection score: {score:.3f}")
                
                # Also save to file
                cv2.imwrite('static/last_captured.jpg', extracted_image)
                
                # Encode extracted image for saving
                _, buffer = cv2.imencode('.jpg', extracted_image)
                image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                emit('capture_success', {'success': True, 'image': image_base64})
            else:
                emit('capture_failed', {'reason': 'blurry'})
        else:
            # Debug: log when markers are not found (only occasionally to avoid spam)
            pass  # Remove emit to avoid too much logging
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': str(e)})


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
        
        # Select the appropriate prompt based on language
        prompt_map = {
            'vi': SYSTEM_PROMPT_VI,
            'en': SYSTEM_PROMPT_EN,
            'zh': SYSTEM_PROMPT_ZH,
            'hi': SYSTEM_PROMPT_HI,
            'es': SYSTEM_PROMPT_ES
        }
        prompt = prompt_map.get(language, SYSTEM_PROMPT_EN)  # Default to English
        
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

