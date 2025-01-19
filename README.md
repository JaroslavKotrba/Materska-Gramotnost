# 👶 Czech Maternity Advisory Chatbot 🤱

A specialized chatbot designed to provide support and advice for mothers with babies in the Czech Republic. Built with FastAPI, OpenAI GPT-4, and FAISS vector storage.

## 🌐 Live Demo

Visit <a href="http://www.mami-bot.cz" target="_blank">www.mami-bot.cz</a> to try the chatbot!

## ✨ Features

- 🗣️ Natural language understanding and generation in Czech
- 💭 Context-aware responses using FAISS vector storage
- 🔄 Conversation history management
- 📊 Administrative statistics and monitoring
- 🔒 Secure API endpoints with key authentication
- 🎯 Specialized knowledge in:
  - Baby care and development
  - Maternal care and recovery
  - Practical parenting advice
  - Social support and benefits

## 🚦 Quick Start

1. Visit <a href="http://www.mami-bot.cz" target="_blank">www.mami-bot.cz</a>
2. Type your question about maternity, baby care, or parenting
3. Get instant, reliable advice in Czech

For healthcare professionals and organizations wanting to integrate the chatbot, see the technical setup below.

## 🛠️ Technical Stack

- 🐍 Python 3.12
- ⚡ FastAPI
- 🧠 OpenAI GPT-4
- 🔍 FAISS Vector Store
- 🗄️ SQLAlchemy + PostgreSQL/SQLite
- 🎨 Jinja2 Templates

## 📁 Project Structure
```
mat_gram/
├── app/
│   ├── static/
│   │   ├── style.css
│   │   ├── script.js
│   │   └── pic/
│   ├── templates/
│   │   └── index.html
│   └── main.py
├── data/
│   └── vector_store/
├── environment.yml
└── requirements.txt
```

## 🚀 Development Setup

### Prerequisites

- Python 3.12
- Conda (recommended for environment management)
- OpenAI API key
- PostgreSQL (for production) or SQLite (for development)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JaroslavKotrba/Materska-Gramotnost.git
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate mat_gram
```

3. **Set up environment variables:**
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
ADMIN_API_KEY=your_admin_api_key
```

4. **Run the application:**
```bash
uvicorn app.main:app --reload
```

## 🔌 API Endpoints

- `GET /` - Web interface
- `POST /chat` - Send message to chatbot
- `POST /clear` - Clear conversation history
- `GET /stats` - Get usage statistics (protected)
- `GET /health` - System health check

## 📊 Monitoring

The application includes comprehensive monitoring via the `/health` endpoint:
- System metrics
- Component status
- Configuration details
- Memory usage
- Uptime statistics

## 🔒 Security

- API key authentication for admin endpoints
- CORS protection
- Environment variable configuration
- Error handling and logging

## 💾 Database

The application supports:
- MySql for production (via JawsDB on Heroku)
- SQLite for local development
- Automatic schema creation
- Interaction logging and analytics

## 👥 Support

For support:
- Visit <a href="http://www.mami-bot.cz" target="_blank">www.mami-bot.cz</a>
- Email: jaroslav.kotrba@gmail.com
- Technical issues: Submit an issue on GitHub

## 🙏 Acknowledgments

- 🤖 OpenAI for GPT models
- ⚡ FastAPI community
- 🔗 LangChain framework