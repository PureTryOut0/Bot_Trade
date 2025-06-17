import os
import re
import json
import time
import logging
import requests
import schedule

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import praw
import gdown

from dotenv import load_dotenv

# =========================
# Carrega variáveis de ambiente
# =========================
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN")
# (os.environ.get das chaves v1.1, caso queira trocar por Tweepy / OAuth 1.1)
TWITTER_ACCESS_TOKEN = os.environ.get("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.environ.get("TWITTER_ACCESS_TOKEN_SECRET")

REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.environ.get("REDDIT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")

LINK_PUBLICO_DRIVE = os.environ.get("LINK_PUBLICO_DRIVE")

# =========================
# Configurações gerais / Paths
# =========================

PASTA_DADOS = "./dados"
os.makedirs(PASTA_DADOS, exist_ok=True)

HISTORICO_ANALISES_FILE = os.path.join(PASTA_DADOS, "historico_analises.json")
PRECOS_MONITORADOS_FILE = os.path.join(PASTA_DADOS, "precos_monitorados.json")
MODELO_LOCAL_PATH = os.path.join(PASTA_DADOS, "modelo_sentimento_5classes.joblib")

LOG_FILE = os.path.join(PASTA_DADOS, "bot_cripto.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =========================
# Mapeamento de rótulos 5 classes
# =========================
label_map_inv_5 = {
    0: "VERY_NEGATIVE",
    1: "NEGATIVE",
    2: "NEUTRAL",
    3: "POSITIVE",
    4: "VERY_POSITIVE"
}
label_map_str_to_num_5 = {v: k for k, v in label_map_inv_5.items()}

# =========================
# Keywords iniciais
# =========================
KEYWORDS = ["bitcoin", "solana", "crypto", "ethereum", "bnb", "altcoin", "blockchain"]

# =========================
# Conjunto dinâmico de ativos Binance
# =========================
BINANCE_ASSETS = set()

# ============================================
# 1. FUNÇÕES AUXILIARES (JSON, Drive, Logging)
# ============================================

def carregar_de_json(caminho: str) -> dict:
    if os.path.exists(caminho):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.warning("JSON inválido em %s. Retornando {}.", caminho)
            return {}
        except Exception as e:
            logger.exception("Erro ao carregar JSON de %s: %s", caminho, e)
            return {}
    return {}


def salvar_em_json(dados: dict, caminho: str):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)
        logger.info("Salvou JSON em %s", caminho)
    except Exception as e:
        logger.exception("Erro ao salvar JSON em %s: %s", caminho, e)


def baixar_json_drive(link_drive: str, destino_local: str):
    """
    Baixa um arquivo público do Google Drive via gdown.
    Ex.: LINK_PUBLICO_DRIVE configurado como 
    https://drive.google.com/file/d/<ID>/view?usp=sharing
    """
    try:
        match = re.search(r"/d/([A-Za-z0-9_-]+)", link_drive)
        if not match:
            logger.error("Link Drive inválido: %s", link_drive)
            return
        arquivo_id = match.group(1)
        url = f"https://drive.google.com/uc?id={arquivo_id}"
        gdown.download(url, destino_local, quiet=True)
        logger.info("Baixou histórico de Drive para %s", destino_local)
    except Exception as e:
        logger.exception("Erro ao baixar JSON do Drive: %s", e)

# ======================================================
# 2. Funções de Sentimento: carregar / treinar / avaliar
# ======================================================

def auto_treinar_modelo_5classes(historico_path: str, modelo_path: str):
    dados = carregar_de_json(historico_path)
    textos, labels = [], []

    for item in dados.values():
        texto = item.get("texto", "").strip()
        sentimento = item.get("sentimento", "NEUTRAL").upper()
        if not texto or sentimento not in label_map_str_to_num_5:
            continue
        textos.append(texto)
        labels.append(label_map_str_to_num_5[sentimento])

    if len(textos) < 10:
        logger.warning("Dados insuficientes (%d) para treinar modelo.", len(textos))
        return None, None

    logger.info("Treinando modelo 5-classes com %d exemplos...", len(textos))
    try:
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(textos)
        y = labels

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=500, multi_class='ovr')
        clf.fit(X_train, y_train)

        acc = clf.score(X_val, y_val)
        logger.info("Modelo treinado. Acurácia validação: %.2f", acc)

        joblib.dump((vectorizer, clf), modelo_path)
        logger.info("Salvou modelo em %s", modelo_path)
        return vectorizer, clf

    except Exception as e:
        logger.exception("Erro ao treinar modelo: %s", e)
        return None, None


def carregar_modelo_local_5classes(modelo_path: str):
    if os.path.exists(modelo_path):
        try:
            vectorizer, classificador = joblib.load(modelo_path)
            logger.info("Carregou modelo de %s", modelo_path)
            return vectorizer, classificador
        except Exception as e:
            logger.exception("Falha ao carregar modelo: %s", e)
            return None, None
    return None, None


def analisar_sentimento_local_5classes(texto: str, vectorizer, clf) -> str:
    try:
        X = vectorizer.transform([texto])
        pred = clf.predict(X)[0]
        return label_map_inv_5.get(pred, "NEUTRAL")
    except Exception as e:
        logger.debug("Erro ao classificar texto: %s | %s", texto, e)
        return "NEUTRAL"

# ======================================================
# 3. Funções para obter lista dinâmica de ativos Binance
# ======================================================

def obter_lista_binance():
    """
    Consulta os símbolos 'baseAsset' ativos (status=TRADING) na API Binance e preenche BINANCE_ASSETS.
    """
    global BINANCE_ASSETS
    BINANCE_ASSETS.clear()
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for symbol_info in data.get("symbols", []):
            if symbol_info.get("status") == "TRADING":
                base = symbol_info.get("baseAsset")
                if base:
                    BINANCE_ASSETS.add(base.upper())
        logger.info("Ativos Binance carregados: %d símbolos", len(BINANCE_ASSETS))
    except Exception as e:
        logger.exception("Erro ao obter lista Binance: %s", e)


def detectar_tokens(texto: str) -> list:
    """
    Varre cada palavra alfanumérica do texto e verifica se está em BINANCE_ASSETS.
    Retorna lista de símbolos encontrados ou ["GEN"] se nenhum.
    """
    texto_lower = texto.lower()
    encontrados = set()
    palavras = re.findall(r"\b[a-zA-Z0-9]+\b", texto_lower)
    for palavra in palavras:
        sib = palavra.upper()
        if sib in BINANCE_ASSETS:
            encontrados.add(sib)
    return list(encontrados) if encontrados else ["GEN"]

# ======================================================
# 4. Coleta de tweets via Twitter API v2 com timestamp
# ======================================================

def coletar_tweets_twitter(vectorizer, clf):
    tweets_filtrados = []
    if not TWITTER_BEARER_TOKEN:
        logger.error("TWITTER_BEARER_TOKEN não configurado. Pulando Twitter.")
        return tweets_filtrados

    headers = {
        "Authorization": f"Bearer {TWITTER_BEARER_TOKEN}",
        "User-Agent": "CryptoNewsBot"
    }
    inicio_24h = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat("T") + "Z"

    for termo in KEYWORDS:
        query = f"{termo} lang:en -is:retweet"
        url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            "query": query,
            "start_time": inicio_24h,
            "max_results": 50,
            "tweet.fields": "author_id,created_at,text,public_metrics",
            "expansions": "author_id",
            "user.fields": "username,public_metrics"
        }
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            tweets = data.get("data", [])
            users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

            for tweet in tweets:
                texto = tweet.get("text", "").strip()
                author_id = tweet.get("author_id")
                user_info = users.get(author_id, {})
                seguidores = user_info.get("public_metrics", {}).get("followers_count")
                created_at = tweet.get("created_at")  # ex: "2025-06-02T14:30:00.000Z"
                metrics = tweet.get("public_metrics", {})
                likes = metrics.get("like_count")
                retweets = metrics.get("retweet_count")

                sentimento = analisar_sentimento_local_5classes(texto, vectorizer, clf)
                tokens = detectar_tokens(texto)

                # Obter preço histórico para cada token no timestamp exato
                precos_no_momento = {}
                for token in tokens:
                    if token != "GEN":
                        preco_hist = pegar_preco_historico_binance(token, created_at)
                        precos_no_momento[token] = preco_hist

                tweets_filtrados.append({
                    "plataforma": "twitter",
                    "autor": user_info.get("username", ""),
                    "seguidores": seguidores,
                    "texto": texto,
                    "likes": likes,
                    "retweets": retweets,
                    "data": created_at,
                    "sentimento": sentimento,
                    "tokens_mencionados": tokens,
                    "precos_no_momento": precos_no_momento
                })

        except requests.RequestException as e:
            logger.warning("Erro ao acessar Twitter para '%s': %s", termo, e)
        except Exception as e:
            logger.exception("Erro ao processar tweets de '%s': %s", termo, e)

    logger.info("Tweets coletados via API Twitter: %d", len(tweets_filtrados))
    return tweets_filtrados

# ======================================================
# 5. Coleta de posts no Reddit (PRAW) com timestamp
# ======================================================

def coletar_reddit(vectorizer, clf):
    posts_filtrados = []
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
    except Exception as e:
        logger.exception("Não foi possível autenticar no Reddit: %s", e)
        return posts_filtrados

    subreddits = ["CryptoCurrency", "Bitcoin", "Solana", "CryptoMarkets"]
    for subreddit in subreddits:
        try:
            for post in reddit.subreddit(subreddit).new(limit=50):
                if post.author and post.author.link_karma >= 500 and post.score >= 10:
                    texto_completo = f"{post.title}. {post.selftext or ''}"
                    sentimento = analisar_sentimento_local_5classes(texto_completo, vectorizer, clf)
                    tokens = detectar_tokens(texto_completo)
                    created_at = datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat()

                    # Obter preço histórico de cada token no timestamp
                    precos_no_momento = {}
                    for token in tokens:
                        if token != "GEN":
                            preco_hist = pegar_preco_historico_binance(token, created_at)
                            precos_no_momento[token] = preco_hist

                    posts_filtrados.append({
                        "plataforma": "reddit",
                        "autor": str(post.author),
                        "karma": post.author.link_karma,
                        "titulo": post.title,
                        "texto": post.selftext or "",
                        "upvotes": post.score,
                        "subreddit": subreddit,
                        "data": created_at,
                        "sentimento": sentimento,
                        "tokens_mencionados": tokens,
                        "precos_no_momento": precos_no_momento
                    })
        except Exception as e:
            logger.exception("Erro no subreddit '%s': %s", subreddit, e)

    logger.info("Posts coletados no Reddit: %d", len(posts_filtrados))
    return posts_filtrados

# ======================================================
# 6. Preço atual e preço histórico via Binance API
# ======================================================

def pegar_preco_binance(token_symbol: str):
    """
    Consulta preço atual (ticker) do par token_symbol + USDT na Binance.
    """
    try:
        pair = f"{token_symbol}USDT"
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        price_str = data.get("price")
        if price_str:
            return float(price_str)
    except Exception:
        return None
    return None


def pegar_preco_historico_binance(token_symbol: str, timestamp_iso: str):
    """
    Busca o fechamento (close) mais próximo do timestamp_iso (ISO 8601) no Kline 1m da Binance.
    """
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        ts_ms = int(dt.timestamp() * 1000)
        pair = f"{token_symbol}USDT"
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={pair}"
            f"&interval=1m"
            f"&limit=1"
            f"&endTime={ts_ms}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            close_price = float(data[0][4])  # índice 4 é o preço de fechamento
            return close_price
    except Exception as e:
        logger.debug("Erro preço histórico %s em %s: %s", token_symbol, timestamp_iso, e)
    return None


def salvar_preco_monitorado(token_symbol: str):
    precos = carregar_de_json(PRECOS_MONITORADOS_FILE)
    agora_iso = datetime.now(timezone.utc).isoformat()
    preco_atual = pegar_preco_binance(token_symbol)
    if preco_atual is not None:
        precos[token_symbol] = {"preco": preco_atual, "data": agora_iso}
        salvar_em_json(precos, PRECOS_MONITORADOS_FILE)
        logger.info("Salvou preço atual %s = %.4f USDT", token_symbol, preco_atual)
    else:
        logger.debug("Preço atual de %s indisponível.", token_symbol)

# ======================================================
# 7. Histórico de análises e verificação de alertas
# ======================================================

def atualizar_historico_analises(novas_analises: list):
    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    for item in novas_analises:
        chave = item.get("texto", "")[:200]
        historico[chave] = item
    salvar_em_json(historico, HISTORICO_ANALISES_FILE)


def verificar_alertas(dados: list, limite: int = 2):
    """
    Conta ocorrências de sentimentos positivos/negativos por token.
    Se ultrapassar o limite, envia alerta e salva preço atual.
    """
    contagem = defaultdict(lambda: {
        "VERY_POSITIVE": 0, "POSITIVE": 0, "NEUTRAL": 0,
        "NEGATIVE": 0, "VERY_NEGATIVE": 0
    })
    tokens_em_alta, tokens_em_baixa = [], []

    for item in dados:
        for token in item.get("tokens_mencionados", []):
            sentimento = item.get("sentimento", "NEUTRAL")
            contagem[token][sentimento] += 1

    for token, sentimentos in contagem.items():
        if token == "GEN":
            continue
        pos_count = sentimentos["VERY_POSITIVE"] + sentimentos["POSITIVE"]
        neg_count = sentimentos["VERY_NEGATIVE"] + sentimentos["NEGATIVE"]
        if pos_count >= limite:
            tokens_em_alta.append((token, pos_count))
            salvar_preco_monitorado(token)
        if neg_count >= limite:
            tokens_em_baixa.append((token, neg_count))
            salvar_preco_monitorado(token)

    if tokens_em_alta:
        lista = ", ".join([f"{t[0]} ({t[1]})" for t in tokens_em_alta])
        msg = f"🔔 *POSSÍVEL ALTA* detectada para: {lista}"
        logger.info(msg)
        enviar_alerta_telegram(msg)

    if tokens_em_baixa:
        lista = ", ".join([f"{t[0]} ({t[1]})" for t in tokens_em_baixa])
        msg = f"⚠️ *POSSÍVEL QUEDA* detectada para: {lista}"
        logger.info(msg)
        enviar_alerta_telegram(msg)

# ======================================================
# 8. Função para enviar mensagem no Telegram
# ======================================================

def enviar_alerta_telegram(mensagem: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Telegram não configurado (verifique TELEGRAM_TOKEN e TELEGRAM_CHAT_ID).")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": mensagem,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Enviado para Telegram: %s", mensagem)
    except requests.RequestException as e:
        logger.error("Erro ao enviar mensagem no Telegram: %s", e)

# ======================================================
# 9. Ranking de possíveis aumentos/quedas (envio a cada 2 h)
# ======================================================

def enviar_ranking_periodico():
    """
    Monta um ranking de tokens que mais tiveram ocorrências positivas e negativas
    nas últimas análises e envia via Telegram.
    """
    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    contagem_positiva = defaultdict(int)
    contagem_negativa = defaultdict(int)

    for item in historico.values():
        data_iso = item.get("data")
        # Considere apenas itens das últimas 24h
        try:
            dt_item = datetime.fromisoformat(data_iso.replace("Z", "+00:00"))
        except:
            continue
        if dt_item < datetime.now(timezone.utc) - timedelta(hours=24):
            continue

        sentimento = item.get("sentimento", "NEUTRAL")
        tokens = item.get("tokens_mencionados", [])
        for token in tokens:
            if token == "GEN":
                continue
            if sentimento in ("POSITIVE", "VERY_POSITIVE"):
                contagem_positiva[token] += 1
            elif sentimento in ("NEGATIVE", "VERY_NEGATIVE"):
                contagem_negativa[token] += 1

    # Ordena tokens por contagem decrescente
    ranking_alta = sorted(contagem_positiva.items(), key=lambda x: x[1], reverse=True)[:5]
    ranking_baixa = sorted(contagem_negativa.items(), key=lambda x: x[1], reverse=True)[:5]

    msg = "*Ranking último 24h*\n\n"
    if ranking_alta:
        msg += "🚀 Top 5 tokens com mais sentimento POSITIVO:\n"
        for token, cnt in ranking_alta:
            msg += f"• {token}: {cnt} menções positivas\n"
    else:
        msg += "🚀 Nenhum token com menções positivas suficientes.\n"

    msg += "\n"

    if ranking_baixa:
        msg += "📉 Top 5 tokens com mais sentimento NEGATIVO:\n"
        for token, cnt in ranking_baixa:
            msg += f"• {token}: {cnt} menções negativas\n"
    else:
        msg += "📉 Nenhum token com menções negativas suficientes.\n"

    enviar_alerta_telegram(msg)
    logger.info("Envio periódico de ranking realizado.")

# ======================================================
# 10. Função principal (setup + primeira execução)
# ======================================================

def obter_modelo():
    vectorizer, clf = carregar_modelo_local_5classes(MODELO_LOCAL_PATH)
    if vectorizer is None or clf is None:
        logger.info("Modelo local não existe. Baixando histórico e treinando...")
        baixar_json_drive(LINK_PUBLICO_DRIVE, HISTORICO_ANALISES_FILE)
        vectorizer, clf = auto_treinar_modelo_5classes(HISTORICO_ANALISES_FILE, MODELO_LOCAL_PATH)
        if vectorizer is None or clf is None:
            logger.warning("Falha ao treinar. Usando fallback NEUTRAL.")
    return vectorizer, clf


def main():
    logger.info("========== Iniciando CryptoNewsBot ==========")

    # 1. Buscar lista de ativos Binance
    obter_lista_binance()

    # 2. Carregar ou treinar modelo
    vectorizer, clf = obter_modelo()

    # 3. Coletar dados do Twitter e Reddit
    if vectorizer and clf:
        tweets = coletar_tweets_twitter(vectorizer, clf)
        posts_reddit = coletar_reddit(vectorizer, clf)
    else:
        # se modelo não carregou, coleta mas marca sentimento = NEUTRAL e sem preços históricos
        tweets = coletar_tweets_twitter(None, None)
        posts_reddit = coletar_reddit(None, None)
        for item in tweets + posts_reddit:
            item["sentimento"] = "NEUTRAL"
            item["precos_no_momento"] = {token: None for token in item.get("tokens_mencionados", [])}

    todos_dados = tweets + posts_reddit
    logger.info("Total de itens coletados (tweets+reddit): %d", len(todos_dados))

    if todos_dados:
        # 4. Atualizar histórico
        atualizar_historico_analises(todos_dados)
        # 5. Verificar e enviar alertas imediatos
        verificar_alertas(todos_dados)
    else:
        logger.info("Nenhum dado foi coletado nesta execução.")

    # 6. Agendar envio de ranking a cada 2 horas
    schedule.every(1).hours.do(enviar_ranking_periodico)
    logger.info("Agendado ranking a cada 2 horas.")

    # 7. Loop de schedule
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Execução interrompida manualmente.")
    except Exception as e:
        logger.exception("Erro no loop de agendamento: %s", e)


if __name__ == "__main__":
    main()
