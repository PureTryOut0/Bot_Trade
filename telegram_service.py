import os
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import Application, CommandHandler
from datetime import datetime
from config import Config

load_dotenv()

import asyncio

class TelegramNotifier:
    def __init__(self):
        self.bot = Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.loop = asyncio.get_event_loop()
        
    def send_opportunity_notification_sync(self, opportunity):
        """Synchronous wrapper for sending opportunity notifications"""
        return self.loop.run_until_complete(self.send_opportunity_notification(opportunity))
    
    def send_error_notification_sync(self, error_message):
        """Synchronous wrapper for sending error notifications"""
        return self.loop.run_until_complete(self.send_error_notification(error_message))
    
    async def send_opportunity_notification(self, opportunity):
        """Send notification about a new investment opportunity"""
        try:
            symbol = opportunity['symbol']
            direction = "📈 Alta" if opportunity['score'] > 0 else "📉 Baixa"
            confidence = opportunity['confidence'] * 100
            risk_level = opportunity['risk_level']
            reason = opportunity['reason']
            
            message = f"""
🚨 Nova Oportunidade de Investimento

Crypto: {symbol}
Direção: {direction}
Confiança: {confidence:.1f}%
Risco: {risk_level}/5

Motivo: {reason}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            print(f"Error sending Telegram notification: {str(e)}")
    
    async def send_error_notification(self, error_message):
        """Send notification about system errors"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"""⚠️ Erro no Sistema:
{error_message}""",
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f"Error sending error notification: {str(e)}")

# Create a global instance for easier access
telegram_notifier = TelegramNotifier()
