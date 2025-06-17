import json
import os
import re
import time
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from transformers import pipeline
from twilio.rest import Client
import praw
import snscrape.modules.twitter as sntwitter

# ============================
# CONFIGURAÇÕES
# ============================
REDDIT_CLIENT_ID = "eYxxxxxxxxxxxxxxxg"
REDDIT_SECRET = "F8hxxxxxxxxxxxxxxxxxxxxxxxx"
REDDIT_USER_AGENT = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxx"
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxx"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+1xxxxxxxxxxxxx"
DESTINATARIO_WHATSAPP = "whatsapp:+5xxxxxxxxxxxxxxxxxxx"

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
    for termo in KEYWORDS:
        for tweet in sntwitter.TwitterSearchScraper(f"{termo} since:{(datetime.utcnow() - timedelta(days=1)).date()}").get_items():
            if tweet.user.followersCount >= 1000:
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
                    "data": tweet.date.isoformat(),
                    "sentimento": sentimento,
                    "tokens_mencionados": tokens
                })
    return tweets_filtrados

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
        for token in item["tokens_mencionados"]:
            contagem[token][item["sentimento"]] += 1

    for token, sentimentos in contagem.items():
        if sentimentos["POSITIVE"] >= limite:
            mensagens.append(f"🔔 ALERTA: {token} em alta! {sentimentos['POSITIVE']} menções positivas recentes.")
        if sentimentos["NEGATIVE"] >= limite:
            mensagens.append(f"⚠️ ALERTA: {token} em queda! {sentimentos['NEGATIVE']} menções negativas recentes.")

    for mensagem in mensagens:
        print(mensagem)
        enviar_alerta_whatsapp(mensagem)

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

def pegar_preco_coingecko(token_symbol):
    symbol_to_id = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "BNB": "binancecoin"}
    coin_id = symbol_to_id.get(token_symbol)
    if not coin_id:
        return None
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    try:
        resposta = requests.get(url)
        resposta.raise_for_status()
        dados = resposta.json()
        return dados[coin_id]["usd"]
    except Exception:
        return None

def salvar_preco_monitorado(token_symbol):
    precos = carregar_de_json(PRECOS_MONITORADOS_FILE)
    agora = datetime.utcnow().isoformat()
    preco_atual = pegar_preco_coingecko(token_symbol)
    if preco_atual is not None:
        precos[token_symbol] = {"preco": preco_atual, "data": agora}
        salvar_em_json(precos, PRECOS_MONITORADOS_FILE)

def verificar_variacao_preco():
    precos = carregar_de_json(PRECOS_MONITORADOS_FILE)
    removidos = []
    for token, info in precos.items():
        data_salva = datetime.fromisoformat(info["data"])
        if datetime.utcnow() - data_salva >= timedelta(days=1):
            preco_atual = pegar_preco_coingecko(token)
            if preco_atual:
                preco_antigo = info["preco"]
                variacao = ((preco_atual - preco_antigo) / preco_antigo) * 100
                msg = f"📊 {token}: Variação após 1 dia = {variacao:.2f}% (de {preco_antigo} USD para {preco_atual} USD)"
                print(msg)
                enviar_alerta_whatsapp(msg)
            removidos.append(token)
    for token in removidos:
        precos.pop(token)
    salvar_em_json(precos, PRECOS_MONITORADOS_FILE)

def executar_bot():
    print("⏳ Coletando dados...")
    tweets = coletar_tweets()
    reddit = coletar_reddit()
    dados_combinados = tweets + reddit

    print("💾 Atualizando histórico...")
    atualizar_historico_analises(dados_combinados)

    print("📈 Verificando alertas de sentimento...")
    verificar_alertas(dados_combinados)

    print("📉 Verificando variação de preços...")
    verificar_variacao_preco()

if __name__ == "__main__":
    executar_bot()

