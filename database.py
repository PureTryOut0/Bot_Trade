from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from datetime import datetime
import os

Base = declarative_base()
engine = create_engine('sqlite:///crypto_analyzer.db')
SessionLocal = scoped_session(sessionmaker(bind=engine))

class CryptoAsset(Base):
    __tablename__ = 'crypto_assets'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False)
    name = Column(String(100), nullable=False)
    sentiment_score = Column(Float)
    volume_24h = Column(Float)
    price_change_24h = Column(Float)
    mention_count = Column(Integer)
    last_updated = Column(DateTime)
    risk_score = Column(Float)
    investment_rating = Column(Integer)  # 1-5 scale
    
    # Relationship with SentimentLog
    sentiment_logs = relationship("SentimentLog", back_populates="crypto_asset")
    
    def __repr__(self):
        return f"<CryptoAsset(symbol='{self.symbol}', sentiment={self.sentiment_score})>"

class SentimentLog(Base):
    __tablename__ = 'sentiment_logs'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'))
    source = Column(String(50))  # reddit, twitter, news, etc.
    sentiment = Column(Float)
    timestamp = Column(DateTime)
    content = Column(String(1000))
    confidence = Column(Float)
    
    # Relationship with CryptoAsset
    crypto_asset = relationship("CryptoAsset", back_populates="sentiment_logs")
    
    def __repr__(self):
        return f"<SentimentLog(asset_id={self.asset_id}, source='{self.source}')>"

class Opportunity(Base):
    __tablename__ = 'investment_opportunities'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('crypto_assets.id'))
    score = Column(Float)
    confidence = Column(Float)
    risk_level = Column(Integer)  # 1-5 scale
    detection_time = Column(DateTime)
    valid_until = Column(DateTime)
    reason = Column(String(500))
    
    # Relationship with CryptoAsset
    crypto_asset = relationship("CryptoAsset")
    
    def __repr__(self):
        return f"<Opportunity(asset_id={self.asset_id}, score={self.score})>"

def init_db():
    Base.metadata.create_all(engine)
    return engine

def get_session():
    return SessionLocal()
