document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements
    const chatWidget = document.getElementById('chatWidget');
    const toggleButton = document.getElementById('toggleChat');
    const toggleIcon = toggleButton.querySelector('i');
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');

    // Generate UUID with fallback for mobile browsers
    // Creates a UUID in format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    // Used when crypto.randomUUID() is not available (e.g. older browsers)
    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Initialize session ID from localStorage or create new one
    // Uses crypto.randomUUID() with fallback to generateUUID()
    let sessionId = localStorage.getItem('chatSessionId');
    if (!sessionId) {
        try {
            sessionId = crypto.randomUUID();
        } catch (e) {
            sessionId = generateUUID();
        }
        localStorage.setItem('chatSessionId', sessionId);
    }

    // Create typing indicator element with animation dots
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';

    // Show typing indicator animation while waiting for response
    function showTypingIndicator() {
        if (!chatMessages.contains(typingIndicator)) {
            chatMessages.appendChild(typingIndicator);
        }
        // Use setTimeout to ensure the transition works
        setTimeout(() => typingIndicator.classList.add('active'), 10);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Hide typing indicator after response received
    function hideTypingIndicator() {
        typingIndicator.classList.remove('active');
        // Remove the element after the fade-out animation
        setTimeout(() => {
            if (chatMessages.contains(typingIndicator)) {
                chatMessages.removeChild(typingIndicator);
            }
        }, 300);
    }

    // Check if device is mobile based on screen width
    function isMobile() {
        return window.innerWidth <= 480;
    }

    // Handle chat widget size on mobile devices
    function handleResize() {
        if (chatWidget.classList.contains('active')) {
            if (isMobile()) {
                const viewportHeight = window.innerHeight;
                chatWidget.style.maxHeight = `${viewportHeight * 0.8}px`;
            } else {
                chatWidget.style.maxHeight = '';
            }
        }
    }

    // Toggle chat widget and icon visibility
    toggleButton.addEventListener('click', () => {
        chatWidget.classList.toggle('active');
        // Toggle between chat and close icons
        toggleIcon.classList.toggle('fa-comments');
        toggleIcon.classList.toggle('fa-times');

        if (chatWidget.classList.contains('active')) {
            userInput.focus();
            if (isMobile()) {
                handleResize();
                setTimeout(() => {
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }, 100);
            }
        }
    });

    // Handle window resize for mobile responsiveness
    window.addEventListener('resize', handleResize);

    // Handle mobile keyboard showing/hiding
    userInput.addEventListener('focus', () => {
        if (isMobile()) {
            setTimeout(() => {
                chatWidget.scrollIntoView({ behavior: 'smooth' });
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 300);
        }
    });

    // Auto-resize textarea based on content
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });

    // Send message on Enter (but create new line on Shift+Enter)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Button click event listeners
    sendButton.addEventListener('click', sendMessage);
    clearButton.addEventListener('click', clearConversation);

    // Append a new message to the chat window
    function appendMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-widget-message ${isUser ? 'user-message' : 'assistant-message'}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    // Send message to server and handle response
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Disable input while processing
        userInput.disabled = true;
        sendButton.disabled = true;

        // Show user message and typing indicator
        appendMessage(message, true);
        showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    session_id: sessionId // Include session ID with each message
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Display bot response
            const data = await response.json();
            hideTypingIndicator();
            appendMessage(data.response, false);
        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
            appendMessage('Omlouvám se, došlo k chybě. Zkuste to prosím znovu.', false);
        } finally {
            // Reset input state
            userInput.value = '';
            userInput.style.height = 'auto';
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    // Clear conversation history and generate new session
    async function clearConversation() {
        try {
            const response = await fetch('/clear', {
                method: 'POST',
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Clear messages and generate new session ID
            chatMessages.innerHTML = '';
            try {
                sessionId = crypto.randomUUID();
            } catch (e) {
                sessionId = generateUUID();
            }
            localStorage.setItem('chatSessionId', sessionId);
        } catch (error) {
            console.error('Error:', error);
            appendMessage('Nepodařilo se vymazat konverzaci. Zkuste to prosím znovu.', false);
        }
    }
});