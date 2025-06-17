from flask import Flask, jsonify, request
from flask.logging import default_handler
import logging
from logging.handlers import RotatingFileHandler
from database import init_db, get_session
from data_collection import DataCollector
from sentiment_analysis import SentimentAnalyzer
from opportunity_detection import OpportunityDetector
from datetime import datetime, timedelta
import schedule
import time
import threading
import os
from dotenv import load_dotenv
from logging_utils import setup_logging, log_request, log_response, log_error
from rate_limiter import rate_limit
from cache import cache_result

load_dotenv()

# Setup logging
logger = setup_logging()

app = Flask(__name__)

# Initialize components
data_collector = DataCollector()
data_collector.collect_data_sync()
sentiment_analyzer = SentimentAnalyzer()
opportunity_detector = OpportunityDetector()
session = get_session()

# Initialize database
init_db()

@app.before_request
def before_request():
    """Log request details"""
    log_request(request)
    request.start_time = time.time()

@app.after_request
def after_request(response):
    """Log response details"""
    request_time = time.time() - request.start_time
    log_response(response, request_time)
    return response

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    log_error(error, request)
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    log_error(error, request)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred'
    }), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Handle all other exceptions"""
    log_error(error, request)
    return jsonify({
        'error': 'Unexpected Error',
        'message': str(error)
    }), 500

@app.route('/api/opportunities', methods=['GET'])
@rate_limit()
@cache_result()
def get_opportunities():
    """Get investment opportunities"""
    try:
        opportunities = opportunity_detector.detect_opportunities()
        return jsonify(opportunities)
    except Exception as e:
        log_error(e, request)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends', methods=['GET'])
@rate_limit()
@cache_result()
def get_trends():
    """Get market trends"""
    try:
        # Get market data
        market_data = data_collector.get_market_data()
        
        # Get sentiment trends
        sentiment_trends = opportunity_detector.get_sentiment_trends()
        
        return jsonify({
            'market_data': market_data,
            'sentiment_trends': sentiment_trends
        })
    except Exception as e:
        log_error(e, request)
        return jsonify({'error': str(e)}), 500

@app.route('/api/sentiment', methods=['POST'])
@rate_limit()
def analyze_sentiment():
    """Analyze sentiment of text"""
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'Text is required'}), 400
            
        result = sentiment_analyzer.analyze_sentiment(text, 'api')
        return jsonify(result)
    except Exception as e:
        log_error(e, request)
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
    try:
        data = request.json
        opportunity_id = data.get('opportunity_id')
        feedback = data.get('feedback')
        
        if not opportunity_id or not feedback:
            return jsonify({'error': 'opportunity_id and feedback are required'}), 400
            
        # Store feedback in database
        feedback_record = Feedback(
            opportunity_id=opportunity_id,
            feedback=feedback,
            timestamp=datetime.now()
        )
        session.add(feedback_record)
        session.commit()
        
        return jsonify({'message': 'Feedback submitted successfully'})
    except Exception as e:
        log_error(e, request)
        return jsonify({'error': str(e)}), 500

# Schedule periodic data collection and analysis
def run_scheduler():
    """Run scheduled tasks"""
    schedule.every(5).minutes.do(collect_and_analyze_data)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

def collect_and_analyze_data():
    """Collect and analyze data"""
    try:
        logger.info("Starting data collection and analysis")
        
        # Collect data
        data = data_collector.process_data()
        
        # Analyze sentiment
        for item in data['processed_data']:
            sentiment = sentiment_analyzer.analyze_sentiment(item['content'], item['source'])
            # Store sentiment in database
            sentiment_log = SentimentLog(
                asset_id=item['metadata']['id'],
                source=item['source'],
                sentiment=sentiment['score'],
                content=item['content'],
                confidence=sentiment['confidence'],
                timestamp=datetime.now()
            )
            session.add(sentiment_log)
            session.commit()
            
        # Detect opportunities
        opportunities = opportunity_detector.detect_opportunities()
        
        # Store opportunities in database
        for opp in opportunities:
            opportunity = Opportunity(
                asset_id=opp['asset_id'],
                score=opp['score'],
                confidence=opp['confidence'],
                risk_level=opp['risk_level'],
                detection_time=datetime.now(),
                valid_until=datetime.now() + timedelta(days=1),
                reason=opp['reason']
            )
            session.add(opportunity)
            session.commit()
            
        logger.info(f"Processed {len(data['processed_data'])} items and detected {len(opportunities)} opportunities")
    except Exception as e:
        logger.error(f"Error in data collection and analysis: {str(e)}")

if __name__ == '__main__':
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    print("Scheduler started")
    
    # Run Flask app
    print("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000)
