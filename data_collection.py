import aiohttp
import tweepy
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
from transformers import pipeline
import asyncio
import asyncpraw

load_dotenv()

class DataCollector:
    def __init__(self):
        """Initialize the DataCollector with all data sources"""
        # Initialize Twitter
        twitter_api_key = os.getenv('TWITTER_API_KEY')
        twitter_api_secret = os.getenv('TWITTER_API_SECRET')
        twitter_access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        twitter_access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        
        if all([twitter_api_key, twitter_api_secret, twitter_access_token, twitter_access_secret]):
            self.twitter_auth = tweepy.OAuth1UserHandler(
                twitter_api_key,
                twitter_api_secret,
                twitter_access_token,
                twitter_access_secret
            )
            self.twitter_api = tweepy.API(self.twitter_auth)
        else:
            print("Warning: Twitter API not configured")
            self.twitter_api = None

        # Initialize Reddit
        self.reddit = None  # Will be initialized later in collect_data_sync()
        
        # Initialize other components
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.token_patterns = self._load_token_patterns()
        self.news_sources = [
            "https://news.bitcoin.com/feed/",
            "https://cointelegraph.com/rss",
            "https://coindesk.com/feed/"
        ]
        self.reddit_subreddits = ["cryptocurrency", "bitcoin", "ethereum"]
        self.twitter_keywords = ["crypto", "bitcoin", "ethereum", "blockchain"]

    async def initialize_reddit(self):
        """Initialize Reddit client asynchronously"""
        self.reddit = asyncpraw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT')
        )
        return self.reddit

    async def collect_data(self):
        """Collect data from all sources"""
        await self.initialize_reddit()
        # Rest of your data collection logic here
        pass

    def collect_data_sync(self):
        """Synchronous wrapper for data collection"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.collect_data())

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _load_token_patterns(self):
        """Load patterns for token detection"""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get('https://api.coingecko.com/api/v3/coins/list') as response:
                if response.status == 200:
                    coins = await response.json()
                    patterns = {}
                    for coin in coins[:100]:  # Limit to top 100 coins
                        patterns[coin['id']] = {
                            'symbol': coin['symbol'].upper(),
                            'name': coin['name']
                        }
                    return patterns
        except Exception as e:
            print(f"Error loading token patterns: {str(e)}")
            return {}

    async def collect_reddit_data(self):
        """Collect data from Reddit"""
        try:
            subreddit = await self.reddit.subreddit('cryptocurrency')
            posts = []
            # Use a smaller limit to avoid rate limits
            async for post in subreddit.new(limit=10):
                try:
                    # Only consider posts from the last 24 hours
                    if (datetime.now() - datetime.fromtimestamp(post.created_utc)).days < 1:
                        posts.append({
                            'title': post.title,
                            'text': post.selftext,
                            'score': post.score,
                            'created_utc': post.created_utc,
                            'num_comments': post.num_comments
                        })
                except Exception as e:
                    print(f"Error processing Reddit post: {str(e)}")
                    continue
            return posts
        except Exception as e:
            print(f"Error collecting Reddit data: {str(e)}")
            return []

    async def collect_twitter_data(self, keywords=['crypto', 'bitcoin', 'ethereum'], limit=100):
        """Collect data from Twitter"""
        data = []
        try:
            for keyword in keywords:
                tweets = self.twitter_api.search_tweets(
                    q=keyword,
                    lang='en',
                    result_type='recent',
                    count=limit
                )
                for tweet in tweets:
                    data.append({
                        'source': 'twitter',
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'retweets': tweet.retweet_count,
                        'favorites': tweet.favorite_count,
                        'user_followers': tweet.user.followers_count
                    })
        except Exception as e:
            print(f"Error collecting Twitter data: {e}")
        return data

    async def collect_news_data(self):
        """Collect data from news sources"""
        data = []
        sources = [
            'https://news.google.com/rss/search?q=cryptocurrency',
            'https://cryptopanic.com/rss/'
        ]
        
        for source in sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries:
                    if (datetime.now() - datetime.fromtimestamp(entry.published_parsed)).days < 1:
                        data.append({
                            'source': 'news',
                            'text': f"{entry.title} {entry.summary}",
                            'created_at': datetime.fromtimestamp(entry.published_parsed),
                            'url': entry.link
                        })
            except Exception as e:
                print(f"Error collecting news data: {e}")
        return data

    async def get_market_data(self, symbols):
        """Get real-time market data"""
        data = {}
        try:
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': ','.join(symbols),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    result = await response.json()
                    for symbol, info in result.items():
                        data[symbol] = {
                            'price': info['usd'],
                            'change_24h': info['usd_24h_change'],
                            'volume_24h': info['usd_24h_vol']
                        }
        except Exception as e:
            print(f"Error getting market data: {e}")
        return data

    async def detect_tokens(self, text):
        """Detect cryptocurrency tokens in text"""
        tokens = set()
        for token, patterns in self.token_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text.lower():
                    tokens.add(token)
        return list(tokens)

    def filter_low_quality(self, data, min_score=10, min_length=20):
        """Filter out low-quality content"""
        return [
            item for item in data
            if (item.get('score', 0) >= min_score and 
                len(item.get('text', '')) >= min_length)
        ]

    async def _load_token_patterns(self):
        """Load token patterns from file"""
        try:
            with open('token_patterns.json', 'r') as f:
                return json.load(f)
        except:
            return {}

    async def analyze_sentiment(self, text, source):
        """Analyze sentiment using Hugging Face transformer pipeline"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'sentiment': result['label'],
                'score': result['score'],
                'confidence': result['score'],
                'source': source,
                'method': 'hf'
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {str(e)}")
            return {
                'sentiment': 'neutral',
                'score': 0.5,
                'confidence': 0.5,
                'source': source,
                'method': 'fallback'
            }

    async def process_data(self):
        """Process all collected data"""
        # Collect data from all sources
        reddit_data = await self.collect_reddit_data()
        twitter_data = await self.collect_twitter_data()
        news_data = await self.collect_news_data()
        market_data = await self.get_market_data(['bitcoin', 'ethereum'])

        # Process each source
        processed_data = []
        for item in reddit_data + twitter_data + news_data:
            # Analyze sentiment
            sentiment = await self.analyze_sentiment(item['text'], 'reddit')
            
            # Detect tokens
            tokens = await self.detect_tokens(item['text'])
            
            # Add to processed data
            processed_data.append({
                'source': 'reddit',
                'content': item['text'],
                'sentiment': sentiment,
                'tokens': tokens,
                'metadata': item
            })

        return {
            'processed_data': processed_data,
            'market_data': market_data
        }
