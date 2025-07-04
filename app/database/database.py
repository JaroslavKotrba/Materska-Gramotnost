import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# Database initialization (base from sqlalchemy)
Base = declarative_base()  # Base


class Database:
    """Connect database"""

    def __init__(self):
        # Get JAWSDB_URL from Heroku
        database_url = os.getenv("JAWSDB_URL")

        if database_url:
            # We're on Heroku, use JawsDB MySQL
            self.engine = create_engine(database_url)
        else:
            # We're local, use SQLite
            self.engine = create_engine("sqlite:///app/database/chatbot.db")

        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()
