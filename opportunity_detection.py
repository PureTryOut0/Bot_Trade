import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import zscore
from database import get_session, CryptoAsset, SentimentLog, Opportunity
import json
import asyncio
from telegram_service import telegram_notifier

class OpportunityDetector:
    def __init__(self):
        self.session = get_session()
        self.sentiment_thresholds = {
            'positive': 0.7,
            'neutral': 0.5,
            'negative': 0.7
        }
        self.opportunity_threshold = 0.8
        self.threshold = 0.7  # Confidence threshold for opportunities
        self.risk_threshold = 0.3  # Risk threshold
        self.volume_threshold = 1000000  # Minimum volume threshold
        self.mention_threshold = 10  # Minimum mentions threshold

    def calculate_sentiment_score(self, sentiment_logs):
        """Calculate weighted sentiment score"""
        if not sentiment_logs:
            return 0.0
            
        scores = []
        weights = []
        
        for log in sentiment_logs:
            weight = 1.0
            if log.source == 'reddit':
                weight *= 1.2
            elif log.source == 'twitter':
                weight *= 1.1
            elif log.source == 'news':
                weight *= 1.3
                
            scores.append(log.sentiment)
            weights.append(weight)
            
        return np.average(scores, weights=weights)

    def calculate_risk_score(self, asset):
        """Calculate risk score based on volatility and sentiment"""
        if not asset.sentiment_logs:
            return 0.0
            
        # Calculate volatility
        price_changes = [log.price_change_24h for log in asset.sentiment_logs]
        if price_changes:
            volatility = np.std(price_changes)
        else:
            volatility = 0.0
            
        # Calculate sentiment volatility
        sentiment_changes = [abs(log.sentiment - log.sentiment_logs[i-1].sentiment) 
                           for i, log in enumerate(asset.sentiment_logs) 
                           if i > 0]
        sentiment_volatility = np.mean(sentiment_changes) if sentiment_changes else 0.0
        
        return (volatility + sentiment_volatility) / 2

    def calculate_investment_rating(self, asset):
        """Calculate investment rating (1-5)"""
        if not asset.sentiment_logs:
            return 0
            
        sentiment_score = self.calculate_sentiment_score(asset.sentiment_logs)
        risk_score = self.calculate_risk_score(asset)
        volume_score = np.log1p(asset.volume_24h) if asset.volume_24h else 0
        
        # Normalize scores
        normalized_sentiment = (sentiment_score + 1) / 2  # Scale to 0-1
        normalized_risk = 1 - (risk_score / 100)  # Inverse risk
        normalized_volume = volume_score / 10000  # Scale volume
        
        # Weighted average
        rating = (normalized_sentiment * 0.4 + 
                 normalized_risk * 0.3 + 
                 normalized_volume * 0.3) * 5  # Scale to 1-5
        
        return min(max(round(rating), 1), 5)

    async def detect_opportunities(self, min_mentions=5, min_confidence=0.7):
        """Detect investment opportunities"""
        opportunities = []
        
        # Get all assets with recent activity
        assets = self.session.query(CryptoAsset).filter(
            CryptoAsset.last_updated >= datetime.now() - timedelta(days=1)
        ).all()
        
        for asset in assets:
            # Filter by minimum mentions and confidence
            if len(asset.sentiment_logs) < min_mentions:
                continue
                
            # Calculate scores
            sentiment_score = self.calculate_sentiment_score(asset.sentiment_logs)
            risk_score = self.calculate_risk_score(asset)
            investment_rating = self.calculate_investment_rating(asset)
            
            # Calculate opportunity score
            opportunity_score = (
                sentiment_score * 0.4 +
                (1 - risk_score) * 0.3 +
                investment_rating * 0.3
            )
            
            # Check if it's a strong opportunity
            if opportunity_score >= self.opportunity_threshold:
                opportunities.append({
                    'asset': asset.symbol,
                    'score': opportunity_score,
                    'confidence': min_confidence,
                    'risk_level': risk_score,
                    'investment_rating': investment_rating,
                    'reason': self._generate_reason(asset)
                })
        
        # Sort and return top opportunities
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        
        for opportunity in opportunities[:5]:
            # Save to database
            await self.save_opportunity(opportunity)
            
            # Send Telegram notification
            await telegram_notifier.send_opportunity_notification(opportunity)
        
        return opportunities[:5]

    def _generate_reason(self, asset):
        """Generate a reason for the opportunity"""
        sentiment = self.calculate_sentiment_score(asset.sentiment_logs)
        risk = self.calculate_risk_score(asset)
        volume = asset.volume_24h
        
        reasons = []
        
        if sentiment > 0.7:
            reasons.append("Strong positive sentiment")
        if risk < 0.5:
            reasons.append("Low risk profile")
        if volume > 1000000:
            reasons.append("High trading volume")
        
        return ", ".join(reasons) if reasons else "Emerging opportunity"

    async def save_opportunity(self, opportunity):
        """Save opportunity to database"""
        try:
            new_opportunity = Opportunity(
                asset_id=self.session.query(CryptoAsset).filter_by(symbol=opportunity['asset']).first().id,
                score=opportunity['score'],
                confidence=opportunity['confidence'],
                risk_level=opportunity['risk_level'],
                detection_time=datetime.now(),
                valid_until=datetime.now() + timedelta(days=1),
                reason=opportunity['reason']
            )
            self.session.add(new_opportunity)
            self.session.commit()
            
        except Exception as e:
            print(f"Error saving opportunity: {e}")
            await telegram_notifier.send_error_notification(f"Error saving opportunity: {e}")

    def get_historical_trends(self, asset_symbol, days=30):
        """Get historical sentiment trends for an asset"""
        try:
            logs = self.session.query(SentimentLog).filter(
                SentimentLog.asset_id == self.session.query(CryptoAsset).filter_by(symbol=asset_symbol).first().id,
                SentimentLog.timestamp >= datetime.now() - timedelta(days=days)
            ).order_by(SentimentLog.timestamp).all()
            
            return pd.DataFrame([
                {
                    'timestamp': log.timestamp,
                    'sentiment': log.sentiment,
                    'price': log.price,
                    'volume': log.volume
                }
                for log in logs
            ])
        except Exception as e:
            print(f"Error getting historical trends: {e}")
            return pd.DataFrame()
