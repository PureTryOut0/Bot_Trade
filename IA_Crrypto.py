# crypto_news_ai_collector.py

import tweepy
import praw
import json
from datetime import datetime, timedelta
from transformers import pipeline
import re
from collections import defaultdict
from twilio.rest import Client
import os
import requests

# ============================
# CONFIGURAÇÕES
# ============================
TWITTER_BEARER_TOKEN = "xxxxxxxxxx"
REDDIT_CLIENT_ID = "xxxxxxxxxxxxxxx"
REDDIT_SECRET = "xxxxxxxxxxxx"
REDDIT_USER_AGENT = "xxxxxxxxxxx"

# Twilio WhatsApp
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxx"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+xxxxxxxxxxx"
DESTINATARIO_WHATSAPP = "whatsapp:+xxxxxxxxxxxxx"

# Palavras-chave para busca
KEYWORDS = ["bitcoin", "solana", "crypto", "ethereum", "bnb", "altcoin", "blockchain"]

# Dicionário de tokens reconhecíveis
TOKEN_MAP = {
    "bitcoin": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "eth": "ETH",
    "solana": "SOL",
    "sol": "SOL",
    "bnb": "BNB",
    "binance coin": "BNB",
    "altcoin": "ALT",
    "crypto": "GEN",
    "blockchain": "GEN"
}

# ============================
# ANÁLISE DE SENTIMENTO
# ============================
def analisar_sentimento(texto):
    try:
        resultado = sentiment_pipeline(texto[:512])
        return resultado[0]['label']
    except Exception:
        return "NEUTRAL"

# Detecta criptos mencionadas
def detectar_tokens(texto):
    texto = texto.lower()
    tokens_encontrados = set()
    for chave, simbolo in TOKEN_MAP.items():
        if re.search(rf"\b{re.escape(chave)}\b", texto):
            tokens_encontrados.add(simbolo)
    return list(tokens_encontrados)

# Carrega pipeline de sentimento (RoBERTa treinado para Twitter)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# ============================
# COLETOR DE TWITTER
# ============================
def coletar_tweets():
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    tweets_filtrados = []

    for termo in KEYWORDS:
        resposta = client.search_recent_tweets(query=termo, tweet_fields=["author_id", "public_metrics", "created_at"], user_fields=["username", "public_metrics"], expansions="author_id", max_results=50)

        if not resposta.data:
            continue

        usuarios = {u["id"]: u for u in resposta.includes["users"]}

        for tweet in resposta.data:
            usuario = usuarios.get(tweet.author_id)
            if usuario and usuario.public_metrics["followers_count"] >= 1000:
                texto = tweet.text
                sentimento = analisar_sentimento(texto)
                tokens = detectar_tokens(texto)
                tweets_filtrados.append({
                    "plataforma": "twitter",
                    "autor": usuario.username,
                    "seguidores": usuario.public_metrics["followers_count"],
                    "texto": texto,
                    "likes": tweet.public_metrics["like_count"],
                    "retweets": tweet.public_metrics["retweet_count"],
                    "data": tweet.created_at.isoformat(),
                    "sentimento": sentimento,
                    "tokens_mencionados": tokens
                })

    return tweets_filtrados

# ============================
# COLETOR DE REDDIT
# ============================
def coletar_reddit():
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    subreddits = ["CryptoCurrency", "Bitcoin", "Solana", "CryptoMarkets"]
    posts_filtrados = []

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).new(limit=50):
            if post.author and post.author.link_karma >= 500 and post.score >= 10:
                texto_completo = post.title + ". " + post.selftext
                sentimento = analisar_sentimento(texto_completo)
                tokens = detectar_tokens(texto_completo)
                posts_filtrados.append({
                    "plataforma": "reddit",
                    "autor": str(post.author),
                    "karma": post.author.link_karma,
                    "titulo": post.title,
                    "texto": post.selftext,
                    "upvotes": post.score,
                    "subreddit": subreddit,
                    "data": datetime.utcfromtimestamp(post.created_utc).isoformat(),
                    "sentimento": sentimento,
                    "tokens_mencionados": tokens
                })

    return posts_filtrados

# ============================
# ENVIAR ALERTA WHATSAPP
# ============================
def enviar_alerta_whatsapp(mensagem):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=mensagem,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=DESTINATARIO_WHATSAPP
    )

# ============================
# OBTÉM PREÇO ATUAL (usando CoinGecko)
# ============================
def obter_preco(token):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={token.lower()}&vs_currencies=usd"
    resposta = requests.get(url)
    if resposta.status_code == 200:
        return resposta.json().get(token.lower(), {}).get("usd")
    return None

# ============================
# ALERTAS AUTOMÁTICOS + RASTREAMENTO DE PREÇO
# ============================

def verificar_alertas(dados, limite=5):
    contagem = defaultdict(lambda: {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0})
    mensagens = []
    precos_iniciais = {}

    for item in dados:
        for token in item["tokens_mencionados"]:
            contagem[token][item["sentimento"]] += 1

    print("\n===== ALERTAS AUTOMÁTICOS =====")
    for token, sentimentos in contagem.items():
        if sentimentos["POSITIVE"] >= limite:
            msg = f"🔔 ALERTA: {token} em alta! {sentimentos['POSITIVE']} menções positivas recentes."
            print(msg)
            mensagens.append(msg)

            preco = obter_preco(token.lower())
            if preco:
                registro = {
                    "token": token,
                    "data": datetime.utcnow().isoformat(),
                    "preco_usd": preco
                }
                salvar_preco_inicial(registro)
        if sentimentos["NEGATIVE"] >= limite:
            msg = f"⚠️ ALERTA: {token} em queda! {sentimentos['NEGATIVE']} menções negativas recentes."
            print(msg)
            mensagens.append(msg)

    for mensagem in mensagens:
        enviar_alerta_whatsapp(mensagem)

# ============================
# SALVAR PREÇO INICIAL
# ============================
def salvar_preco_inicial(registro):
    if os.path.exists("precos_iniciais.json"):
        with open("precos_iniciais.json", "r") as f:
            precos = json.load(f)
    else:
        precos = []

    precos.append(registro)
    with open("precos_iniciais.json", "w") as f:
        json.dump(precos, f, indent=4)

# ============================
# VERIFICAR VARIAÇÃO APÓS 1 DIA
# ============================
def verificar_variacao():
    if not os.path.exists("precos_iniciais.json"):
        print("Nenhum dado salvo para verificação de variação.")
        return

    with open("precos_iniciais.json", "r") as f:
        precos = json.load(f)

    for registro in precos:
        data_registro = datetime.fromisoformat(registro["data"])
        if datetime.utcnow() - data_registro >= timedelta(days=1):
            preco_atual = obter_preco(registro["token"].lower())
            if preco_atual:
                variacao = ((preco_atual - registro["preco_usd"]) / registro["preco_usd"]) * 100
                msg = f"📊 VARIAÇÃO: {registro['token']} mudou {variacao:.2f}% em 24h."
                print(msg)
                enviar_alerta_whatsapp(msg)

# ============================
# SALVAR RESULTADOS
# ============================
def salvar_em_json(dados, nome_arquivo):
    with open(nome_arquivo, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

# ============================
# EXECUÇÃO PRINCIPAL
# ============================
if __name__ == "__main__":
    print("Coletando dados do Twitter e Reddit...")
    dados_twitter = coletar_tweets()
    dados_reddit = coletar_reddit()

    todos_dados = dados_twitter + dados_reddit
    salvar_em_json(todos_dados, "noticias_filtradas.json")
    print(f"Coleta finalizada. {len(todos_dados)} registros salvos em 'noticias_filtradas.json'")

    verificar_alertas(todos_dados, limite=5)
    verificar_variacao()
