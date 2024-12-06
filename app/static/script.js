document.addEventListener('DOMContentLoaded', () => {
    const chatWidget = document.getElementById('chatWidget');
    const toggleButton = document.getElementById('toggleChat');
    const toggleIcon = toggleButton.querySelector('i');
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');

    function isMobile() {
        return window.innerWidth <= 480;
    }

    function handleMobileChat(isActive) {
        if (isMobile()) {
            document.body.style.overflow = isActive ? 'hidden' : '';
            if (isActive) {
                chatWidget.style.height = '100vh';
                chatWidget.style.width = '100vw';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
    }

    // Toggle chat widget
    toggleButton.addEventListener('click', () => {
        const willBeActive = !chatWidget.classList.contains('active');
        chatWidget.classList.toggle('active');
        toggleIcon.classList.toggle('fa-comments');
        toggleIcon.classList.toggle('fa-times');

        handleMobileChat(willBeActive);

        if (willBeActive) {
            userInput.focus();
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    });

    // Handle resize
    window.addEventListener('resize', () => {
        if (chatWidget.classList.contains('active')) {
            handleMobileChat(true);
        }
    });

    // Auto-resize textarea
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 100) + 'px';
    });

    // Handle mobile keyboard
    userInput.addEventListener('focus', () => {
        if (isMobile()) {
            setTimeout(() => {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 300);
        }
    });

    // Send message on Enter (but create new line on Shift+Enter)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Button click handlers
    sendButton.addEventListener('click', sendMessage);
    clearButton.addEventListener('click', clearConversation);

    function appendMessage(content, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-widget-message ${isUser ? 'user-message' : 'assistant-message'}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        userInput.disabled = true;
        sendButton.disabled = true;

        appendMessage(message, true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            appendMessage(data.response, false);
        } catch (error) {
            console.error('Error:', error);
            appendMessage('Omlouvám se, došlo k chybě. Zkuste to prosím znovu.', false);
        } finally {
            userInput.value = '';
            userInput.style.height = 'auto';
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }

    async function clearConversation() {
        try {
            const response = await fetch('/clear', {
                method: 'POST',
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            chatMessages.innerHTML = '';
        } catch (error) {
            console.error('Error:', error);
            appendMessage('Nepodařilo se vymazat konverzaci. Zkuste to prosím znovu.', false);
        }
    }
});