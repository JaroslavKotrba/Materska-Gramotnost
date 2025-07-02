# TO RUN APP

# conda env create -f environment.yml
# conda create -n mat_gram python=3.12
# conda env remove -n mat_gram

# uvicorn app.main:app --reload

# psycopg2-binary==2.9.9 (Heroku Postgres)
# mysqlclient==2.2.1 (JAWSDB MySQL)

# conda env export --name mat_gram > environment.yml
# pip list --format=freeze > requirements.txt

import os
import logging
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .core.chatbot import ChatbotConfig, CoreChatbot
from .routers import endpoints

# Path
os.getcwd()

# Port configuration for Heroku
port = int(os.getenv("PORT", 8000))

# Configure logging to track chatbot operations and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Start time for uptime tracking
START_TIME = time.time()

# FastAPI app initialization
app = FastAPI(
    title="Maternity Chatbot API",
    description="API for a Czech maternity advisory chatbot",
    version="1.0.0",
)

# CLIENT DOMAIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://jaroslavkotrba.com",
        "https://mat-gram-c3a70edf9532.herokuapp.com",  # Heroku
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JavaScript)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Chatbot initialization
try:
    config = ChatbotConfig()
    chatbot = CoreChatbot(config)
except Exception as e:
    logger.error(f"Failed to initialize chatbot: {str(e)}")
    raise

# Initialize the router with dependencies
endpoints.init_router(config, chatbot, START_TIME)

# Include the endpoints router
app.include_router(endpoints.router)
