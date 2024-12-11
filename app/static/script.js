document.addEventListener('DOMContentLoaded', () => {
    const chatWidget = document.getElementById('chatWidget');
    const toggleButton = document.getElementById('toggleChat');
    const toggleIcon = toggleButton.querySelector('i');
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearButton');

    // Create typing indicator element
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';

    function showTypingIndicator() {
        if (!chatMessages.contains(typingIndicator)) {
            chatMessages.appendChild(typingIndicator);
        }
        // Use setTimeout to ensure the transition works
        setTimeout(() => typingIndicator.classList.add('active'), 10);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function hideTypingIndicator() {
        typingIndicator.classList.remove('active');
        // Remove the element after the fade-out animation
        setTimeout(() => {
            if (chatMessages.contains(typingIndicator)) {
                chatMessages.removeChild(typingIndicator);
            }
        }, 300);
    }

    function isMobile() {
        return window.innerWidth <= 480;
    }

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

    // Toggle chat widget and icon
    toggleButton.addEventListener('click', () => {
        chatWidget.classList.toggle('active');
        // Toggle the icon class
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

    // Handle window resize
    window.addEventListener('resize', handleResize);

    // Handle mobile keyboard
    userInput.addEventListener('focus', () => {
        if (isMobile()) {
            setTimeout(() => {
                chatWidget.scrollIntoView({ behavior: 'smooth' });
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }, 300);
        }
    });

    // Auto-resize textarea
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
        showTypingIndicator();

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
            hideTypingIndicator();
            appendMessage(data.response, false);
        } catch (error) {
            console.error('Error:', error);
            hideTypingIndicator();
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