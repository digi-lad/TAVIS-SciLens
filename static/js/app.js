// Socket.IO client
const socket = io();

// State management
const app = document.getElementById('app');
let currentScreen = 'language';
let stream = null;
let videoElement = null;
let captureInterval = null;
let processingInterval = null;
let currentLanguage = 'vi'; // 'vi', 'en', 'zh', 'hi', 'es'
let isProcessing = false; // Flag to prevent multiple concurrent API calls
let abortController = null; // To cancel ongoing fetch requests

// Translations
const translations = {
    vi: {
        talkbackQuestion: 'BẠN CÓ DÙNG TALKBACK?',
        talkbackYes: 'CÓ',
        talkbackNo: 'KHÔNG',
        cancel: 'HỦY BỎ',
        tapToCapture: 'CHẠM ĐỂ CHỤP ẢNH',
        processing: 'ĐANG XỬ LÝ ẢNH',
        scienceDescription: 'MÔ TẢ KHOA HỌC',
        tryAgain: 'CHỤP ẢNH KHÁC',
        error: 'LỖI',
        cameraError: 'Không thể truy cập camera. Vui lòng cho phép truy cập camera.',
        serverError: 'Mất kết nối với máy chủ. Vui lòng thử lại.',
        errorPrefix: 'Lỗi kỹ thuật: ',
        cameraScreen: 'Giao diện camera',
        cameraPreview: 'Xem trước camera',
        imageCaptured: 'Ảnh đã được chụp thành công',
        resultsReady: 'Kết quả đã sẵn sàng'
    },
    en: {
        talkbackQuestion: 'DO YOU USE TALKBACK?',
        talkbackYes: 'YES',
        talkbackNo: 'NO',
        cancel: 'CANCEL',
        tapToCapture: 'TAP TO CAPTURE IMAGE',
        processing: 'PROCESSING IMAGE',
        scienceDescription: 'SCIENCE DESCRIPTION',
        tryAgain: 'CAPTURE ANOTHER IMAGE',
        error: 'ERROR',
        cameraError: 'Cannot access camera. Please allow camera access.',
        serverError: 'Connection lost to server. Please try again.',
        errorPrefix: 'Technical error: ',
        cameraScreen: 'Camera view',
        cameraPreview: 'Camera preview',
        imageCaptured: 'Image captured successfully',
        resultsReady: 'Results ready'
    },
    zh: {
        talkbackQuestion: '您使用TALKBACK吗？',
        talkbackYes: '是',
        talkbackNo: '否',
        cancel: '取消',
        tapToCapture: '点击拍摄图像',
        processing: '正在处理图像',
        scienceDescription: '科学描述',
        tryAgain: '拍摄另一张图像',
        error: '错误',
        cameraError: '无法访问相机。请允许相机访问。',
        serverError: '与服务器断开连接。请重试。',
        errorPrefix: '技术错误：',
        cameraScreen: '相机界面',
        cameraPreview: '相机预览',
        imageCaptured: '图像已成功捕获',
        resultsReady: '结果已准备就绪'
    },
    hi: {
        talkbackQuestion: 'क्या आप TALKBACK का उपयोग करते हैं?',
        talkbackYes: 'हाँ',
        talkbackNo: 'नहीं',
        cancel: 'रद्द करें',
        tapToCapture: 'छवि कैप्चर करने के लिए टैप करें',
        processing: 'छवि प्रसंस्करण',
        scienceDescription: 'विज्ञान विवरण',
        tryAgain: 'अन्य छवि कैप्चर करें',
        error: 'त्रुटि',
        cameraError: 'कैमरा तक नहीं पहुंच सकते। कृपया कैमरा एक्सेस की अनुमति दें।',
        serverError: 'सर्वर से कनेक्शन टूट गया। कृपया पुनः प्रयास करें।',
        errorPrefix: 'तकनीकी त्रुटि: ',
        cameraScreen: 'कैमरा दृश्य',
        cameraPreview: 'कैमरा पूर्वावलोकन',
        imageCaptured: 'छवि सफलतापूर्वक कैप्चर की गई',
        resultsReady: 'परिणाम तैयार हैं'
    },
    es: {
        talkbackQuestion: '¿USA TALKBACK?',
        talkbackYes: 'SÍ',
        talkbackNo: 'NO',
        cancel: 'CANCELAR',
        tapToCapture: 'TOCAR PARA CAPTURAR IMAGEN',
        processing: 'PROCESANDO IMAGEN',
        scienceDescription: 'DESCRIPCIÓN CIENTÍFICA',
        tryAgain: 'CAPTURAR OTRA IMAGEN',
        error: 'ERROR',
        cameraError: 'No se puede acceder a la cámara. Por favor permita el acceso a la cámara.',
        serverError: 'Conexión perdida con el servidor. Por favor intente de nuevo.',
        errorPrefix: 'Error técnico: ',
        cameraScreen: 'Vista de cámara',
        cameraPreview: 'Vista previa de cámara',
        imageCaptured: 'Imagen capturada exitosamente',
        resultsReady: 'Resultados listos'
    }
};

// Screen elements
let screens = {};

// Audio elements
let beepSound;
let processingSound;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize screen elements after DOM is ready
    screens = {
        language: document.getElementById('language-selection'),
        talkback: document.getElementById('talkback-question'),
        initial: document.getElementById('initial-state'),
        camera: document.getElementById('camera-state'),
        processing: document.getElementById('processing-state'),
        results: document.getElementById('results-state'),
        error: document.getElementById('error-state')
    };
    
    // Initialize audio elements
    beepSound = document.getElementById('beep-sound');
    processingSound = document.getElementById('processing-sound');
    
    // Debug: Check if native camera input exists
    const nativeInput = document.getElementById('native-camera-input');
    console.log('Native camera input found:', nativeInput !== null);
    
    setupEventListeners();
    setupSocketListeners();
    
    // Check if language is already selected
    const savedLang = localStorage.getItem('language');
    if (savedLang && (savedLang === 'vi' || savedLang === 'en' || savedLang === 'zh' || savedLang === 'hi' || savedLang === 'es')) {
        currentLanguage = savedLang;
        updateLanguage();
        // Check if TalkBack preference is saved
        const talkbackAnswer = localStorage.getItem('talkback');
        if (talkbackAnswer !== null && talkbackAnswer !== '') {
            showScreen('initial');
        } else {
            showScreen('talkback');
        }
    } else {
        showScreen('language');
    }
});

function setupEventListeners() {
    // Language selection buttons
    const langViBtn = document.getElementById('lang-vi');
    const langEnBtn = document.getElementById('lang-en');
    const langZhBtn = document.getElementById('lang-zh');
    const langHiBtn = document.getElementById('lang-hi');
    const langEsBtn = document.getElementById('lang-es');
    
    langViBtn.addEventListener('click', () => selectLanguage('vi'));
    langEnBtn.addEventListener('click', () => selectLanguage('en'));
    langZhBtn.addEventListener('click', () => selectLanguage('zh'));
    langHiBtn.addEventListener('click', () => selectLanguage('hi'));
    langEsBtn.addEventListener('click', () => selectLanguage('es'));
    
    // TalkBack question buttons
    const talkbackYesBtn = document.getElementById('talkback-yes');
    const talkbackNoBtn = document.getElementById('talkback-no');
    talkbackYesBtn.addEventListener('click', () => selectTalkback(true));
    talkbackNoBtn.addEventListener('click', () => selectTalkback(false));
    
    // Initial screen - tap anywhere to capture
    const initialScreen = document.getElementById('initial-state');
    initialScreen.addEventListener('click', startCamera);
    initialScreen.addEventListener('touchstart', startCamera);
    
    // Processing cancel button
    const processingCancelBtn = document.getElementById('processing-cancel-btn');
    processingCancelBtn.addEventListener('click', (e) => {
        // Prevent event bubbling and add a small delay check
        e.preventDefault();
        e.stopPropagation();
        
        // Use setTimeout to ensure this happens after any capture event completes
        setTimeout(() => {
            // Abort ongoing fetch request if exists
            if (abortController) {
                abortController.abort();
                abortController = null;
            }
            
            clearInterval(processingInterval);
            processingInterval = null;
            isProcessing = false;
            showScreen('initial');
        }, 50);
    });
    
    // Try again buttons
    const tryAgainBtn = document.getElementById('try-again-btn');
    tryAgainBtn.addEventListener('click', resetToInitial);
    
    const errorRetryBtn = document.getElementById('error-retry-btn');
    errorRetryBtn.addEventListener('click', resetToInitial);
    
    // Native camera file input handler
    const nativeCameraInput = document.getElementById('native-camera-input');
    if (nativeCameraInput) {
        nativeCameraInput.addEventListener('change', handleNativeCameraCapture);
    }
}

function selectLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    updateLanguage();
    showScreen('talkback');
}

function selectTalkback(usesTalkback) {
    // Store user's TalkBack preference as boolean string
    localStorage.setItem('talkback', usesTalkback.toString());
    
    // For now, just proceed to capture screen regardless of answer
    // TTS will be enabled automatically if they selected "No TalkBack"
    showScreen('initial');
}

function wrapKeywords(text) {
    // Define keywords for each language
    const keywords = {
        'vi': 'CHẠM',
        'en': 'TAP',
        'zh': '点击',
        'hi': 'टैप',
        'es': 'TOCAR'
    };
    
    const keyword = keywords[currentLanguage];
    
    if (!keyword || !text.includes(keyword)) {
        return text;
    }
    
    // Wrap the keyword in a span and add line break after it
    const regex = new RegExp(`(${keyword})\\s*`, 'gi');
    return text.replace(regex, '<span class="keyword">$1</span><br>');
}

function updateLanguage() {
    const t = translations[currentLanguage];
    const usesTalkback = JSON.parse(localStorage.getItem('talkback') || 'false');
    
    // Update talkback question
    document.getElementById('talkback-title').textContent = t.talkbackQuestion;
    document.getElementById('talkback-yes').querySelector('.talkback-answer').textContent = t.talkbackYes;
    document.getElementById('talkback-no').querySelector('.talkback-answer').textContent = t.talkbackNo;
    
    // Update talkback button aria-labels
    const talkbackYesBtn = document.getElementById('talkback-yes');
    const talkbackNoBtn = document.getElementById('talkback-no');
    if (talkbackYesBtn) {
        talkbackYesBtn.setAttribute('aria-label', `${t.talkbackYes}, ${t.talkbackQuestion}`);
    }
    if (talkbackNoBtn) {
        talkbackNoBtn.setAttribute('aria-label', `${t.talkbackNo}, ${t.talkbackQuestion}`);
    }
    
    // Update initial screen with keyword highlighting
    const tapTextWithKeywords = wrapKeywords(t.tapToCapture);
    document.getElementById('initial-title').innerHTML = tapTextWithKeywords;
    
    // Update processing screen
    document.querySelector('#processing-state h2').textContent = t.processing;
    const processingCancelBtn = document.getElementById('processing-cancel-btn');
    if (processingCancelBtn) {
        processingCancelBtn.textContent = t.cancel;
        processingCancelBtn.setAttribute('aria-label', t.cancel);
    }
    
    // Update results screen
    document.querySelector('.results-header h2').textContent = t.scienceDescription;
    
    // Update screen aria-labels dynamically
    const cameraState = document.getElementById('camera-state');
    if (cameraState) {
        cameraState.setAttribute('aria-label', t.cameraScreen);
    }
    
    const processingState = document.getElementById('processing-state');
    if (processingState) {
        processingState.setAttribute('aria-label', t.processing);
    }
    
    const resultsState = document.getElementById('results-state');
    if (resultsState) {
        resultsState.setAttribute('aria-label', t.scienceDescription);
    }
    
    const videoElement = document.getElementById('video-preview');
    if (videoElement) {
        videoElement.setAttribute('aria-label', t.cameraPreview);
    }
    
    // Update button aria-labels
    const tryAgainBtn = document.getElementById('try-again-btn');
    if (tryAgainBtn) {
        tryAgainBtn.textContent = t.tryAgain;
        tryAgainBtn.setAttribute('aria-label', t.tryAgain);
    }
    
    const errorRetryBtn = document.getElementById('error-retry-btn');
    if (errorRetryBtn) {
        errorRetryBtn.textContent = t.tryAgain;
        errorRetryBtn.setAttribute('aria-label', t.tryAgain);
    }
    
    const errorStateH2 = document.querySelector('#error-state h2');
    if (errorStateH2) {
        errorStateH2.textContent = t.error;
    }
}

function setupSocketListeners() {
    socket.on('capture_success', (data) => {
        handleCaptureSuccess(data);
    });
    
    socket.on('capture_failed', (data) => {
        console.log('Image too blurry, retaking...');
        // Will continue trying automatically
    });
    
    socket.on('markers_not_found', () => {
        // Will continue trying automatically
    });
    
    socket.on('error', (data) => {
        const t = translations[currentLanguage];
        showError(t.errorPrefix + data.message);
    });
}

function showScreen(screenName) {
    // Hide all screens
    Object.values(screens).forEach(screen => {
        screen.classList.remove('active');
    });
    
    // Manage video element for TalkBack users
    const usesTalkback = JSON.parse(localStorage.getItem('talkback') || 'false');
    const videoElement = document.getElementById('video-preview');
    const t = translations[currentLanguage];
    
    if (screenName === 'camera' && videoElement) {
        // Camera screen is active, update aria attributes
        if (usesTalkback) {
            videoElement.setAttribute('aria-label', t.cameraPreview || 'Camera preview');
            videoElement.setAttribute('aria-hidden', 'false');
        }
    } else if (videoElement) {
        // Camera screen is not active, hide from screen readers
        videoElement.setAttribute('aria-hidden', 'true');
    }
    
    // Show target screen
    if (screens[screenName]) {
        screens[screenName].classList.add('active');
        currentScreen = screenName;
    }
}

async function startCamera() {
    // Get TalkBack preference
    const usesTalkback = JSON.parse(localStorage.getItem('talkback') || 'false');
    
    if (usesTalkback) {
        // TalkBack mode: Use video stream with ArUco detection
        try {
            showScreen('camera');
            
            // Request camera access
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment', // Use back camera on mobile
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            videoElement = document.getElementById('video-preview');
            videoElement.srcObject = stream;
            
            // Start processing frames
            startFrameProcessing();
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            showError(translations[currentLanguage].cameraError);
        }
    } else {
        // Non-TalkBack mode: Open camera, wait for tap to capture
        try {
            showScreen('camera');
            
            // Request camera access
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment', // Use back camera on mobile
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            
            videoElement = document.getElementById('video-preview');
            videoElement.srcObject = stream;
            
            // Don't start continuous processing - wait for user tap
            // Add tap listener to camera screen
            const cameraScreen = document.getElementById('camera-state');
            let tapHandler;
            tapHandler = (e) => {
                // Prevent event from propagating
                e.preventDefault();
                e.stopPropagation();
                
                // Immediately remove listeners to prevent double-tap
                cameraScreen.removeEventListener('click', tapHandler);
                cameraScreen.removeEventListener('touchstart', tapHandler);
                
                // Capture photo
                captureManualPhoto();
            };
            cameraScreen.addEventListener('click', tapHandler);
            cameraScreen.addEventListener('touchstart', tapHandler);
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            showError(translations[currentLanguage].cameraError);
        }
    }
}

function startFrameProcessing() {
    if (!videoElement || !stream) return;
    
    // Process frames at ~3 fps
    captureInterval = setInterval(() => {
        captureFrame();
    }, 333); // ~3 fps
}

function captureFrame() {
    // Don't send frames if already processing
    if (isProcessing) {
        return;
    }
    
    if (!videoElement || videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
        return;
    }
    
    // Create canvas to capture frame
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    
    // Get TalkBack preference
    const usesTalkback = JSON.parse(localStorage.getItem('talkback') || 'false');
    
    // Send to server for processing with TalkBack flag
    socket.emit('process_frame', { frame: imageData, talkback: usesTalkback });
}

function captureManualPhoto() {
    // Check if already processing
    if (isProcessing) {
        return;
    }
    
    if (!videoElement || videoElement.readyState !== videoElement.HAVE_ENOUGH_DATA) {
        console.error('Video not ready');
        return;
    }
    
    // Set processing flag
    isProcessing = true;
    
    // IMPORTANT: Capture frame BEFORE stopping camera
    // Create canvas to capture current frame
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    const base64Data = imageData.split(',')[1];
    
    // Validate we got the data
    if (!base64Data || base64Data.length === 0) {
        console.error('Failed to capture image data');
        isProcessing = false;
        const t = translations[currentLanguage];
        showError(t.errorPrefix + 'Failed to capture image.');
        return;
    }
    
    console.log('Captured image, base64 length:', base64Data.length);
    
    // Now stop video stream
    stopCamera();
    
    // Play beep sound
    beepSound.play().catch(e => console.log('Could not play beep:', e));
    
    // Add a small delay before showing processing screen to prevent tap from triggering cancel button
    // This ensures the tap event has fully completed
    setTimeout(() => {
        // Show processing screen
        showScreen('processing');
        
        // Send directly to Gemini for analysis (bypass ArUco detection)
        analyzeWithGemini(base64Data);
    }, 100);
}

function handleNativeCameraCapture(event) {
    const file = event.target.files[0];
    if (!file) {
        // User cancelled
        return;
    }
    
    // Check if already processing
    if (isProcessing) {
        event.target.value = '';
        return;
    }
    
    // Set processing flag
    isProcessing = true;
    
    // Play beep sound
    beepSound.play().catch(e => console.log('Could not play beep:', e));
    
    // Show processing screen
    showScreen('processing');
    
    // Read file as data URL
    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        // Extract base64 part (after comma)
        const base64Data = dataUrl.split(',')[1];
        
        // Send directly to Gemini for analysis (bypass ArUco detection)
        analyzeWithGemini(base64Data);
    };
    
    reader.onerror = (error) => {
        console.error('Error reading file:', error);
        isProcessing = false;
        const t = translations[currentLanguage];
        showError(t.errorPrefix + 'Failed to read image file.');
    };
    
    reader.readAsDataURL(file);
    
    // Reset file input
    event.target.value = '';
}

function handleCaptureSuccess(data) {
    // Check if already processing
    if (isProcessing) {
        console.log('Already processing an image, ignoring new capture');
        return;
    }
    
    // Set processing flag
    isProcessing = true;
    
    // Stop frame processing
    clearInterval(captureInterval);
    captureInterval = null;
    
    // Stop video stream
    stopCamera();
    
    // Play beep sound
    beepSound.play().catch(e => console.log('Could not play beep:', e));
    
    // Show processing screen
    showScreen('processing');
    
    // Play processing sound periodically
    let soundCount = 0;
    processingInterval = setInterval(() => {
        soundCount++;
        
        // Play processing sound every 2 seconds (6 intervals at ~3 fps)
        if (soundCount % 6 === 0) {
            processingSound.play().catch(e => console.log('Could not play processing sound:', e));
        }
    }, 333);
    
    // Send to Gemini for analysis
    analyzeWithGemini(data.image);
}

async function analyzeWithGemini(imageData) {
    // Create abort controller for this request
    abortController = new AbortController();
    const signal = abortController.signal;
    
    // Set a timeout for the API call (30 seconds)
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => {
            if (!signal.aborted) {
                reject(new Error('Request timeout. The server took too long to respond.'));
            }
        }, 30000);
    });
    
    try {
        const response = await Promise.race([
            fetch('/analyze_image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    image: imageData,
                    language: currentLanguage 
                }),
                signal: signal // Add abort signal
            }),
            timeoutPromise
        ]);
        
        // Check if request was cancelled
        if (signal.aborted) {
            console.log('Request was cancelled');
            return;
        }
        
        const result = await response.json();
        
        // Check again if cancelled before processing result
        if (signal.aborted) {
            console.log('Request was cancelled before processing result');
            return;
        }
        
        if (response.ok && result.sentences) {
            // Stop processing indicator
            clearInterval(processingInterval);
            processingInterval = null;
            
            // Reset processing flag
            isProcessing = false;
            abortController = null; // Clear abort controller
            
            // Play completion beep
            beepSound.play().catch(e => console.log('Could not play completion beep:', e));
            
            // Display results
            displayResults(result.sentences);
        } else {
            throw new Error(result.error || 'Image processing error');
        }
        
    } catch (error) {
        // Check if error is due to abort
        if (error.name === 'AbortError' || signal.aborted) {
            console.log('Request was cancelled');
            abortController = null;
            return;
        }
        
        console.error('Error analyzing image:', error);
        clearInterval(processingInterval);
        processingInterval = null;
        
        // Reset processing flag on error
        isProcessing = false;
        abortController = null;
        
        // Only show error if not cancelled
        if (!signal.aborted) {
            const t = translations[currentLanguage];
            showError(t.errorPrefix + error.message);
        }
    }
}

function displayResults(sentences) {
    const resultsContent = document.getElementById('results-content');
    resultsContent.innerHTML = ''; // Clear previous results
    
    const usesTalkback = JSON.parse(localStorage.getItem('talkback') || 'false');
    const t = translations[currentLanguage];
    
    // For TalkBack users, hide the wrapper's ARIA role to avoid duplication
    if (usesTalkback) {
        resultsContent.removeAttribute('role');
        resultsContent.removeAttribute('aria-label');
        
        // Add a simple announcement for TalkBack users
        const announcement = document.createElement('div');
        announcement.className = 'sr-only';
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('role', 'status');
        const sentenceCount = sentences.length;
        // Simple announcement in current language
        announcement.textContent = `${t.resultsReady || 'Results ready'}. ${sentenceCount} ${sentenceCount === 1 ? 'sentence' : 'sentences'}.`;
        resultsContent.appendChild(announcement);
        
        // Clear announcement after a short delay to avoid repetition
        setTimeout(() => {
            if (announcement.parentNode) {
                announcement.parentNode.removeChild(announcement);
            }
        }, 1000);
    }
    
    // Create a sentence element for each sentence
    sentences.forEach((sentence, index) => {
        const sentenceElement = document.createElement('p');
        sentenceElement.className = 'sentence';
        sentenceElement.textContent = sentence;
        sentenceElement.setAttribute('tabindex', '0');
        sentenceElement.setAttribute('data-sentence-index', index);
        
        // For TalkBack users, add proper ARIA attributes
        if (usesTalkback) {
            sentenceElement.setAttribute('role', 'text');
            // Don't add aria-label - the textContent is enough, aria-label would be redundant
        } else {
            // For non-TalkBack users with TTS
            sentenceElement.style.cursor = 'pointer';
            sentenceElement.addEventListener('click', () => {
                speakFromIndex(index, sentences);
            });
        }
        
        resultsContent.appendChild(sentenceElement);
    });
    
    // Show results screen
    showScreen('results');
    
    // Auto-start TTS if user doesn't use TalkBack
    if (!usesTalkback) {
        speakFromIndex(0, sentences);
    }
}

let currentUtterance = null;
let currentSentenceIndex = 0;
let allSentences = []; // Store all sentences for replay

function speakFromIndex(index, sentences) {
    // Cancel any ongoing speech
    speechSynthesis.cancel();
    
    // Store sentences for later replay
    allSentences = sentences;
    currentSentenceIndex = index;
    
    // Find and focus the current sentence
    const sentenceElements = document.querySelectorAll('.sentence');
    if (sentenceElements[index]) {
        sentenceElements[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
        sentenceElements[index].focus();
    }
    
    // Speak current sentence
    function speakSentence(i) {
        if (i >= sentences.length) {
            console.log('All sentences spoken');
            return;
        }
        
        const sentence = sentences[i];
        const utterance = new SpeechSynthesisUtterance(sentence);
        
        // Set language based on current language
        const langMap = {
            'vi': 'vi-VN',
            'en': 'en-US',
            'zh': 'zh-CN',
            'hi': 'hi-IN',
            'es': 'es-ES'
        };
        utterance.lang = langMap[currentLanguage] || 'en-US';
        utterance.rate = 1.0; // Normal speed
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        currentUtterance = utterance;
        
        // Highlight current sentence
        highlightSentence(i);
        
        // When finished, move to next sentence
        utterance.onend = () => {
            currentUtterance = null;
            speakSentence(i + 1);
        };
        
        utterance.onerror = (event) => {
            console.error('Speech synthesis error:', event);
        };
        
        speechSynthesis.speak(utterance);
    }
    
    speakSentence(index);
}

function highlightSentence(index) {
    const sentenceElements = document.querySelectorAll('.sentence');
    sentenceElements.forEach((el, i) => {
        if (i === index) {
            el.style.backgroundColor = '#333333';
            el.style.borderLeftColor = '#FFD700';
            el.style.borderLeftWidth = '8px';
        } else {
            el.style.backgroundColor = '#000000';
            el.style.borderLeftWidth = '6px';
        }
    });
}

function stopCamera() {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    
    if (videoElement) {
        videoElement.srcObject = null;
    }
}

function showError(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    showScreen('error');
}

function resetToInitial() {
    stopCamera();
    
    // Stop any ongoing speech synthesis
    speechSynthesis.cancel();
    
    // Abort any ongoing fetch requests
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    
    // Clear any intervals
    if (captureInterval) clearInterval(captureInterval);
    if (processingInterval) clearInterval(processingInterval);
    
    // Reset processing flag
    isProcessing = false;
    
    showScreen('initial');
}

// Socket connection indicators
socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('disconnect', () => {
    console.log('Disconnected from server');
    showError(translations[currentLanguage].serverError);
});

