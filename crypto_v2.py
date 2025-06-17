import json 
import os
import re
import time
import requests
import logging
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from transformers import pipeline
from twilio.rest import Client
import praw
from bs4 import BeautifulSoup
import gdown

# ============================ 
# CONFIGURAÇÕES
# ============================
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.getenv("REDDIT_SECRET")
REDDIT_USER_AGENT = "CryptoNewsBot by /u/Mysterious-Air-1431"

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"
DESTINATARIO_WHATSAPP = "whatsapp:+5511949595295"

KEYWORDS = ["bitcoin", "solana", "crypto", "ethereum", "bnb", "altcoin", "blockchain"]

INTERVALO_ANALISE = 3600  # intervalo em segundos (exemplo: 1 hora)

TOKEN_MAP = {
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
    "solana": "SOL", "sol": "SOL",
    "bnb": "BNB", "binance coin": "BNB",
    "altcoin": "ALT", "crypto": "GEN", "blockchain": "GEN"
}

PASTA_DADOS = "mnt/data"
os.makedirs(PASTA_DADOS, exist_ok=True)

HISTORICO_ANALISES_FILE = os.path.join(PASTA_DADOS, "historico_analises.json")
PRECOS_MONITORADOS_FILE = os.path.join(PASTA_DADOS, "precos_monitorados.json")
CACHE_TWITTER_FILE = os.path.join(PASTA_DADOS, "cache_twitter.json")
LINK_PUBLICO_DRIVE = "16wlLPO0iax1-MHshefTjTyg15OqIsqC8"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def carregar_de_json(caminho):
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def salvar_em_json(objeto, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(objeto, f, ensure_ascii=False, indent=2)

def carregar_cache():
    return carregar_de_json(CACHE_TWITTER_FILE)

def salvar_cache(cache):
    salvar_em_json(cache, CACHE_TWITTER_FILE)

def baixar_json_drive(file_id, destino_local):
    try:
        gdown.download(id=file_id, output=destino_local, quiet=False)
        logging.info(f"Arquivo baixado de ID '{file_id}' para {destino_local}")
    except Exception as e:
        logging.error(f"Erro ao baixar JSON do Google Drive: {e}")

def analisar_sentimento(texto):
    try:
        resultado = sentiment_pipeline(texto[:512])
        label = resultado[0]['label']
        score = resultado[0]['score']
        if score < 0.6:
            return "NEUTRAL"
        if label == "LABEL_0":
            return "NEGATIVE"
        elif label == "LABEL_2":
            return "POSITIVE"
        else:
            return "NEUTRAL"
    except Exception as e:
        logging.warning(f"Erro ao analisar sentimento: {e}")
        return "NEUTRAL"

def detectar_tokens(texto):
    texto = texto.lower()
    tokens_encontrados = set()
    for chave, simbolo in TOKEN_MAP.items():
        if re.search(rf"\\b{re.escape(chave)}\\b", texto):
            tokens_encontrados.add(simbolo)
    return list(tokens_encontrados) if tokens_encontrados else ["GEN"]

def coletar_tweets_nitter():
    tweets_filtrados = []
    mirrors = [
        "https://nitter.poast.org",
        "https://nitter.privacydev.net",
        "https://nitter.in.projectsegfau.lt",
        "https://nitter.catsarch.com"
    ]
    headers = {"User-Agent": "Mozilla/5.0"}

    data_ontem = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    cache = carregar_cache()

    for termo in KEYWORDS:
        cache_key = f"{termo}_{data_ontem}"
        if cache_key in cache:
            logging.info(f"Usando cache para '{termo}' ({data_ontem})")
            tweets_filtrados.extend(cache[cache_key])
            continue

        sucesso = False
        for base_url in mirrors:
            url = f"{base_url}/search?f=tweets&q={termo}&since={data_ontem}"
            try:
                resposta = requests.get(url, headers=headers, timeout=15)
                if resposta.status_code == 200 and "timeline-item" in resposta.text:
                    soup = BeautifulSoup(resposta.text, "html.parser")
                    resultados = []
                    for tweet_div in soup.select("div.timeline-item"):
                        texto_tag = tweet_div.select_one("div.tweet-content")
                        autor_tag = tweet_div.select_one("a.username")
                        if texto_tag and autor_tag:
                            texto = texto_tag.text.strip()
                            autor = autor_tag.text.strip()
                            sentimento = analisar_sentimento(texto)
                            tokens = detectar_tokens(texto)
                            resultados.append({
                                "plataforma": "twitter",
                                "autor": autor,
                                "seguidores": 0,
                                "texto": texto,
                                "likes": 0,
                                "retweets": 0,
                                "data": datetime.now(timezone.utc).isoformat(),
                                "sentimento": sentimento,
                                "tokens_mencionados": tokens
                            })
                    if resultados:
                        cache[cache_key] = resultados
                        tweets_filtrados.extend(resultados)
                        sucesso = True
                        break
                else:
                    logging.warning(f"{base_url} falhou ou retornou HTML inválido.")
            except requests.RequestException as e:
                logging.error(f"Erro de conexão com {base_url}: {e}")
        if not sucesso:
            logging.warning(f"Não foi possível obter resultados para '{termo}' em nenhum mirror.")
    salvar_cache(cache)
    return tweets_filtrados

def coletar_posts_reddit():
    posts = []
    for termo in KEYWORDS:
        for submission in reddit.subreddit("all").search(termo, sort="new", time_filter="day", limit=10):
            texto = submission.title + "\n" + submission.selftext
            sentimento = analisar_sentimento(texto)
            tokens = detectar_tokens(texto)
            posts.append({
                "plataforma": "reddit",
                "autor": submission.author.name if submission.author else "unknown",
                "seguidores": 0,
                "texto": texto,
                "likes": submission.score,
                "retweets": 0,
                "data": datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc).isoformat(),
                "sentimento": sentimento,
                "tokens_mencionados": tokens
            })
    return posts

def enviar_mensagem_whatsapp(mensagem):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=mensagem,
            to=DESTINATARIO_WHATSAPP
        )
        logging.info(f"Mensagem enviada com SID: {message.sid}")
    except Exception as e:
        logging.error(f"Erro ao enviar mensagem via WhatsApp: {e}")

def executar_analise():
    try:
        logging.info("Iniciando análise...")
        baixar_json_drive(LINK_PUBLICO_DRIVE, HISTORICO_ANALISES_FILE)
        dados_twitter = coletar_tweets_nitter()
        dados_reddit = coletar_posts_reddit()
        dados_combinados = dados_twitter + dados_reddit
        resumo = defaultdict(lambda: defaultdict(int))

        for item in dados_combinados:
            for token in item["tokens_mencionados"]:
                resumo[token][item["sentimento"]] += 1

        mensagem = "📊 Análise de Sentimento\n"
        for token, sentimentos in resumo.items():
            mensagem += f"{token}: 👍 {sentimentos['POSITIVE']} | 😐 {sentimentos['NEUTRAL']} | 👎 {sentimentos['NEGATIVE']}\n"

        enviar_mensagem_whatsapp(mensagem)
    except Exception as e:
        logging.exception("Erro inesperado durante execução da análise")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Executar em loop contínuo")
    args = parser.parse_args()

    if args.loop:
        while True:
            executar_analise()
            logging.info(f"Aguardando {INTERVALO_ANALISE} segundos para nova análise...\n")
            time.sleep(INTERVALO_ANALISE)
    else:
        executar_analise()
