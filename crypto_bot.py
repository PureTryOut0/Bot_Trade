import json
import os
import re
import time
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from transformers import pipeline
from twilio.rest import Client
import snscrape.modules.twitter as sntwitter
from snscrape.base import ScraperException
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Patch para ignorar SSL no requests (alternativa emergencial)
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

class UnsafeAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['cert_reqs'] = 'CERT_NONE'
        kwargs['assert_hostname'] = False
        return super().init_poolmanager(*args, **kwargs)

requests_session = requests.Session()
requests_session.mount('https://', UnsafeAdapter())

# ============================
# CONFIGURAÇÕES
# ============================
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxxxc4xxxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxxxx1xxxxxxxxxxxxxxxxxxd"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+1xxxxxxxxx"
DESTINATARIO_WHATSAPP = "whatsapp:+xxxxxxx"

KEYWORDS = ["bitcoin", "solana", "crypto", "ethereum", "bnb", "altcoin", "blockchain"]

TOKEN_MAP = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "bnb": "BNB", "binance coin": "BNB",
    "altcoin": "ALT", "crypto": "GEN", "blockchain": "GEN"
}

HISTORICO_ANALISES_FILE = "historico_analises.json"
PRECOS_MONITORADOS_FILE = "precos_monitorados.json"

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

def analisar_sentimento(texto):
    try:
        resultado = sentiment_pipeline(texto[:512])
        return resultado[0]['label']
    except Exception:
        return "NEUTRAL"

def detectar_tokens(texto):
    texto = texto.lower()
    tokens_encontrados = set()
    for chave, simbolo in TOKEN_MAP.items():
        if re.search(rf"\b{re.escape(chave)}\b", texto):
            tokens_encontrados.add(simbolo)
    return list(tokens_encontrados)

def coletar_tweets():
    tweets_filtrados = []
    try:
        for termo in KEYWORDS:
            for tweet in sntwitter.TwitterSearchScraper(
                    f"{termo} since:{(datetime.utcnow() - timedelta(days=1)).date()}").get_items():
                if tweet.user and tweet.user.followersCount >= 1000:
                    texto = tweet.content
                    sentimento = analisar_sentimento(texto)
                    tokens = detectar_tokens(texto)
                    tweets_filtrados.append({
                        "plataforma": "twitter",
                        "autor": tweet.user.username,
                        "seguidores": tweet.user.followersCount,
                        "texto": texto,
                        "likes": tweet.likeCount,
                        "retweets": tweet.retweetCount,
                        "data": (tweet.date.isoformat()
                                 if tweet.date else datetime.utcnow().isoformat()),
                        "sentimento": sentimento,
                        "tokens_mencionados": tokens
                    })
    except ScraperException as e:
        print(f"Erro ao coletar tweets: {e}")
    return tweets_filtrados

def coletar_reddit_sem_praw():
    headers = {'User-Agent': 'Mozilla/5.0'}
    subreddits = ["CryptoCurrency", "Bitcoin", "Solana", "CryptoMarkets"]
    posts_filtrados = []

    for subreddit in subreddits:
        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=50"
        try:
            resp = requests_session.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for post in data['data']['children']:
                post_data = post['data']
                if post_data.get("ups", 0) >= 10:
                    texto_completo = post_data.get("title", "") + ". " + post_data.get("selftext", "")
                    sentimento = analisar_sentimento(texto_completo)
                    tokens = detectar_tokens(texto_completo)
                    posts_filtrados.append({
                        "plataforma": "reddit",
                        "autor": post_data.get("author"),
                        "karma": post_data.get("ups"),
                        "titulo": post_data.get("title"),
                        "texto": post_data.get("selftext"),
                        "upvotes": post_data.get("ups"),
                        "subreddit": subreddit,
                        "data": datetime.utcfromtimestamp(post_data.get("created_utc")).isoformat(),
                        "sentimento": sentimento,
                        "tokens_mencionados": tokens
                    })
        except Exception as e:
            print(f"Erro ao acessar subreddit {subreddit}: {e}")

    return posts_filtrados

def enviar_alerta_whatsapp(mensagem):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=mensagem,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=DESTINATARIO_WHATSAPP
    )

def verificar_alertas(dados, limite=5):
    contagem = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0})
    mensagens = []

    for item in dados:
        for token in item.get("tokens_mencionados", []):
            contagem[token][item.get("sentimento", "NEUTRAL")] += 1

    for token, sentimentos in contagem.items():
        if sentimentos["POSITIVE"] >= limite:
            mensagens.append(f"\U0001F4C8\U0001F4E2 *ALTA* detectada para *{token}*: {sentimentos['POSITIVE']} menções positivas nas últimas 24h.")
        if sentimentos["NEGATIVE"] >= limite:
            mensagens.append(f"\U0001F4C9⚠️ *QUEDA* detectada para *{token}*: {sentimentos['NEGATIVE']} menções negativas nas últimas 24h.")

    for msg in mensagens:
        print(msg)
        enviar_alerta_whatsapp(f"\U0001F4E2 *Alerta de Cripto!*\n\n{msg}")

def salvar_em_json(dados, arquivo):
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

def carregar_de_json(arquivo):
    if os.path.exists(arquivo):
        with open(arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def atualizar_historico_analises(novas_analises):
    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    for item in novas_analises:
        historico[item['texto']] = item
    salvar_em_json(historico, HISTORICO_ANALISES_FILE)

def pegar_preco_alternative_api(token_symbol):
    symbol_to_id = {
        "BTC": "btc-bitcoin", "ETH": "eth-ethereum",
        "SOL": "sol-solana", "BNB": "bnb-binance-coin"
    }
    coin_id = symbol_to_id.get(token_symbol)
    if not coin_id:
        return None
    url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}"
    try:
        resp = requests_session.get(url)
        resp.raise_for_status()
        data = resp.json()
        return float(data['quotes']['USD']['price'])
    except Exception as e:
        print(f"Erro ao obter preço de {token_symbol}: {e}")
        return None

def salvar_preco_monitorado(token_symbol):
    if token_symbol in ["GEN", "ALT"]:
        return
    precos = carregar_de_json(PRECOS_MONITORADOS_FILE)
    agora = datetime.utcnow().isoformat()
    preco_atual = pegar_preco_alternative_api(token_symbol)
    if preco_atual is not None:
        precos[token_symbol] = {"preco": preco_atual, "data": agora}
        salvar_em_json(precos, PRECOS_MONITORADOS_FILE)

def executar_bot():
    print("\u23F3 Coletando dados...")
    dados_twitter = coletar_tweets()
    dados_reddit = coletar_reddit_sem_praw()
    todos_os_dados = dados_twitter + dados_reddit

    atualizar_historico_analises(todos_os_dados)
    verificar_alertas(todos_os_dados)
    for item in todos_os_dados:
        for token in item.get("tokens_mencionados", []):
            salvar_preco_monitorado(token)

    print("\u2705 Execução concluída.")

if __name__ == "__main__":
    executar_bot()
