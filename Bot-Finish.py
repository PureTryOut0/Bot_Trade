import os
import re
import json
import time
import logging
import requests
import schedule
import feedparser

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
from bs4 import BeautifulSoup
import gdown
import praw

from dotenv import load_dotenv

# =========================
# 1. Carrega variáveis de ambiente
# =========================
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Credenciais CoinMarketCap (não estritamente necessárias para histórico)
CMC_API_KEY = os.environ.get("COINMARKETCAP_API_KEY")

# Credenciais Reddit (PRAW)
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID")
REDDIT_SECRET = os.environ.get("REDDIT_SECRET")
REDDIT_USER_AGENT = os.environ.get("REDDIT_USER_AGENT")

# Lista de subreddits a monitorar
REDDIT_SUBREDDITS = ["CryptoCurrency", "Bitcoin", "Solana", "CryptoMarkets"]

LINK_PUBLICO_DRIVE = os.environ.get("LINK_PUBLICO_DRIVE")

# =========================
# 2. Configurações gerais / Paths
# =========================

PASTA_DADOS = "./dados"
os.makedirs(PASTA_DADOS, exist_ok=True)

HISTORICO_ANALISES_FILE = os.path.join(PASTA_DADOS, "historico_analises.json")
PRECOS_MONITORADOS_FILE = os.path.join(PASTA_DADOS, "precos_monitorados.json")
MODELO_LOCAL_PATH = os.path.join(PASTA_DADOS, "modelo_sentimento_5classes.joblib")

# Garante existência dos arquivos JSON (caso não existam, cria vazios)
for caminho in [HISTORICO_ANALISES_FILE, PRECOS_MONITORADOS_FILE]:
    if not os.path.exists(caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump({}, f)

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
# 3. Mapeamento de rótulos 5 classes (para notícias e Reddit)
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
# 4. Keywords iniciais para análise de texto
# =========================
KEYWORDS = ["bitcoin", "solana", "crypto", "ethereum", "bnb", "altcoin", "blockchain"]

# =========================
# 5. Conjunto dinâmico de ativos Binance
# =========================
BINANCE_ASSETS = set()

# =========================
# 6. Controle de updates do Telegram
# =========================
LAST_UPDATE_ID = None

# ============================================
# 7. FUNÇÕES AUXILIARES (JSON, Drive, Logging)
# ============================================

def carregar_de_json(caminho: str) -> dict:
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Falha ao ler JSON %s: %s. Retornando vazio.", caminho, e)
        return {}

def salvar_em_json(dados: dict, caminho: str):
    """
    Grava o dicionário 'dados' em JSON no caminho especificado.
    """
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)
        logger.info("✅ Gravado JSON em %s (itens: %d)", caminho, len(dados))
    except Exception as e:
        logger.exception("Erro ao salvar JSON em %s: %s", caminho, e)

def baixar_json_drive(link_drive: str, destino_local: str):
    """
    Baixa um arquivo público do Google Drive via gdown.
    """
    try:
        match = re.search(r"/d/([A-Za-z0-9_-]+)", link_drive)
        if not match:
            logger.warning("Link Drive inválido: %s", link_drive)
            return
        arquivo_id = match.group(1)
        url = f"https://drive.google.com/uc?id={arquivo_id}"
        gdown.download(url, destino_local, quiet=True)
        logger.info("✅ Baixou histórico de Drive para %s", destino_local)
    except Exception as e:
        logger.exception("Erro ao baixar JSON do Drive: %s", e)

# ======================================================
# 8. Funções de Sentimento: carregar / treinar / avaliar
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

    if len(textos) < 10 or len(set(labels)) < 2:
        logger.warning("⚠️ Dados insuficientes (%d) ou sem variedade para treinar.", len(textos))
        return None, None

    try:
        logger.info("🚀 Treinando modelo 5-classes com %d exemplos...", len(textos))
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(textos)
        y = labels

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = LogisticRegression(max_iter=500, multi_class='ovr')
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        logger.info("✅ Modelo treinado. Acurácia validação: %.2f", acc)

        joblib.dump((vectorizer, clf), modelo_path)
        logger.info("💾 Modelo salvo em %s", modelo_path)
        return vectorizer, clf

    except Exception as e:
        logger.exception("Erro ao treinar modelo: %s", e)
        return None, None

def carregar_modelo_local_5classes(modelo_path: str):
    if os.path.exists(modelo_path):
        try:
            logger.info("🔄 Carregando modelo local de %s", modelo_path)
            return joblib.load(modelo_path)
        except Exception as e:
            logger.exception("Falha ao carregar modelo: %s", e)
            return None, None
    return None, None

def analisar_sentimento_local_5classes(texto: str, vectorizer, clf) -> str:
    try:
        X = vectorizer.transform([texto])
        pred = clf.predict(X)[0]
        return label_map_inv_5.get(pred, "NEUTRAL")
    except Exception:
        return "NEUTRAL"

# ======================================================
# 9. Funções para obter lista dinâmica de ativos Binance
# ======================================================

def obter_lista_binance():
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
        logger.info("✅ Ativos Binance carregados: %d símbolos", len(BINANCE_ASSETS))
    except Exception as e:
        logger.exception("Erro ao obter lista Binance: %s", e)

def detectar_tokens(texto: str) -> list:
    texto_lower = texto.lower()
    encontrados = set()
    palavras = re.findall(r"\b[a-zA-Z0-9]+\b", texto_lower)
    for palavra in palavras:
        sib = palavra.upper()
        if sib in BINANCE_ASSETS:
            encontrados.add(sib)
    return list(encontrados) if encontrados else ["GEN"]

# ======================================================
# 10. Coleta de notícias via RSS (feedparser)
# ======================================================

RSS_SOURCES = [
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CryptoSlate", "https://cryptoslate.com/feed/")
]

def coletar_noticias_rss(vectorizer, clf):
    noticias_filtradas = []
    limite_data = datetime.now(timezone.utc) - timedelta(hours=24)

    for fonte, url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if not hasattr(entry, "published_parsed"):
                    continue
                dt_entry = datetime.fromtimestamp(
                    time.mktime(entry.published_parsed), timezone.utc
                )
                if dt_entry < limite_data:
                    continue
                titulo = entry.get("title", "")
                summary = entry.get("summary", "")
                texto = f"{titulo}. {BeautifulSoup(summary, 'html.parser').get_text()}"
                sentimento = analisar_sentimento_local_5classes(texto, vectorizer, clf) if vectorizer and clf else "NEUTRAL"
                tokens = detectar_tokens(texto)

                precos_no_momento = {}
                for token in tokens:
                    if token != "GEN":
                        preco_hist = pegar_preco_historico_binance(token, dt_entry.isoformat())
                        precos_no_momento[token] = preco_hist

                noticias_filtradas.append({
                    "plataforma": "news",
                    "autor": fonte,
                    "texto": texto,
                    "data": dt_entry.isoformat(),
                    "sentimento": sentimento,
                    "tokens_mencionados": tokens,
                    "precos_no_momento": precos_no_momento
                })
        except Exception as e:
            logger.exception("Erro ao coletar RSS de %s: %s", fonte, e)

    logger.info("📡 Notícias coletadas via RSS: %d", len(noticias_filtradas))
    return noticias_filtradas

# ======================================================
# 11. Coleta de posts no Reddit via PRAW
# ======================================================

def coletar_reddit_praw(vectorizer, clf):
    posts_filtrados = []
    if not (REDDIT_CLIENT_ID and REDDIT_SECRET and REDDIT_USER_AGENT):
        logger.warning("Credenciais Reddit ausentes; pulando coleta Reddit.")
        return posts_filtrados

    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
    except Exception as e:
        logger.exception("Erro ao autenticar no Reddit: %s", e)
        return posts_filtrados

    limite_data = datetime.now(timezone.utc) - timedelta(hours=24)
    for subreddit in REDDIT_SUBREDDITS:
        try:
            for post in reddit.subreddit(subreddit).new(limit=100):
                if post.author and post.author.link_karma >= 500 and post.score >= 10:
                    dt_obj = datetime.fromtimestamp(post.created_utc, timezone.utc)
                    if dt_obj < limite_data:
                        continue
                    dt_iso = dt_obj.isoformat()
                    texto_completo = f"{post.title}. {post.selftext or ''}"
                    sentimento = analisar_sentimento_local_5classes(texto_completo, vectorizer, clf) if vectorizer and clf else "NEUTRAL"
                    tokens = detectar_tokens(texto_completo)

                    precos_no_momento = {}
                    for token in tokens:
                        if token != "GEN":
                            preco_hist = pegar_preco_historico_binance(token, dt_iso)
                            precos_no_momento[token] = preco_hist

                    posts_filtrados.append({
                        "plataforma": "reddit",
                        "autor": str(post.author),
                        "karma": post.author.link_karma,
                        "titulo": post.title,
                        "texto": post.selftext or "",
                        "upvotes": post.score,
                        "subreddit": subreddit,
                        "data": dt_iso,
                        "sentimento": sentimento,
                        "tokens_mencionados": tokens,
                        "precos_no_momento": precos_no_momento
                    })
        except Exception as e:
            logger.exception("Erro ao coletar do subreddit %s: %s", subreddit, e)

    logger.info("📡 Posts coletados via Reddit PRAW: %d", len(posts_filtrados))
    return posts_filtrados

# ======================================================
# 12. Funções para dados de mercado Binance
# ======================================================

def coletar_dados_24h_binance(symbols: list):
    """
    Retorna um dicionário com estatísticas 24h para cada symbol em symbols.
    Usa endpoint /api/v3/ticker/24hr do Binance.
    """
    resultados = {}
    for token in symbols:
        pair = f"{token}USDT"
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={pair}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            resultados[token] = {
                "preco_atual": float(data.get("lastPrice", 0)),
                "volume": float(data.get("volume", 0)),
                "var_24h": float(data.get("priceChangePercent", 0)),
                "alta_24h": float(data.get("highPrice", 0)),
                "baixa_24h": float(data.get("lowPrice", 0))
            }
        except Exception:
            resultados[token] = None
    logger.info("📊 Dados 24h Binance coletados para %d tokens", len(symbols))
    return resultados

# ======================================================
# 13. Funções para dados de mercado CoinMarketCap
# ======================================================

def coletar_dados_cmc(tokens: list):
    """
    Retorna dicionário com preço, market cap e mudança 24h via CoinMarketCap API.
    Espera lista de symbols (ex.: ["BTC", "ETH"]).
    """
    resultados = {}
    if not CMC_API_KEY or not tokens:
        return resultados

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"symbol": ",".join(tokens), "convert": "USD"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        for sym in tokens:
            entry = data.get(sym)
            if not entry:
                resultados[sym] = None
                continue
            quote = entry["quote"]["USD"]
            resultados[sym] = {
                "preco_atual": quote.get("price"),
                "market_cap": quote.get("market_cap"),
                "var_24h": quote.get("percent_change_24h"),
                "volume_24h": quote.get("volume_24h")
            }
    except Exception:
        pass
    logger.info("📈 Dados CMC coletados para %d tokens", len(tokens))
    return resultados

# ======================================================
# 14. Preço histórico via Binance API (para timestamps passados)
# ======================================================

def pegar_preco_historico_binance(token_symbol: str, timestamp_iso: str):
    """
    Busca fechamento mais próximo do timestamp_iso no Kline 1m da Binance.
    """
    try:
        dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
        ts_ms = int(dt.timestamp() * 1000)
        pair = f"{token_symbol}USDT"
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={pair}&interval=1m&limit=1&endTime={ts_ms}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return float(data[0][4])  # índice 4 é preço de fechamento
    except Exception:
        return None

def salvar_preco_monitorado(token_symbol: str):
    precos = carregar_de_json(PRECOS_MONITORADOS_FILE)
    agora_iso = datetime.now(timezone.utc).isoformat()
    preco_atual = None

    # Tenta CMC primeiro
    cmc_data = coletar_dados_cmc([token_symbol]).get(token_symbol)
    if cmc_data and cmc_data.get("preco_atual") is not None:
        preco_atual = cmc_data["preco_atual"]
    else:
        preco_atual = pegar_preco_historico_binance(token_symbol, agora_iso.replace("Z", "+00:00"))

    if preco_atual is not None:
        precos[token_symbol] = {"preco": preco_atual, "data": agora_iso}
        salvar_em_json(precos, PRECOS_MONITORADOS_FILE)

# ======================================================
# 15. Histórico de análises e verificação de alertas
# ======================================================

def atualizar_historico_analises(novas_analises: list):
    """
    Adiciona cada item de 'novas_analises' ao JSON de histórico.
    Usa o campo 'texto' como chave (até 200 caracteres) para evitar duplicatas exatas.
    """
    if not novas_analises:
        logger.info("📝 Nenhuma análise nova para adicionar ao histórico.")
        return

    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    antes = len(historico)
    for item in novas_analises:
        chave = item.get("texto", "")[:200]
        historico[chave] = item
    depois = len(historico)
    salvar_em_json(historico, HISTORICO_ANALISES_FILE)
    logger.info("📝 Histórico atualizado: %d → %d entradas", antes, depois)

def verificar_alertas(dados: list, limite: int = 2):
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
        lista = ", ".join(f"{t[0]} ({t[1]})" for t in tokens_em_alta)
        msg = f"🔔 *POSSÍVEL ALTA* detectada para: {lista}"
        enviar_alerta_telegram(msg)
    if tokens_em_baixa:
        lista = ", ".join(f"{t[0]} ({t[1]})" for t in tokens_em_baixa)
        msg = f"⚠️ *POSSÍVEL QUEDA* detectada para: {lista}"
        enviar_alerta_telegram(msg)

# ======================================================
# 16. Função para enviar mensagem no Telegram
# ======================================================

def enviar_alerta_telegram(mensagem: str, chat_id: str = TELEGRAM_CHAT_ID):
    if not TELEGRAM_TOKEN or not chat_id:
        logger.warning("⚠️ Telegram não configurado; não será enviado alerta.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        logger.error("Erro ao enviar mensagem no Telegram: %s", e)

# ======================================================
# 17. Ranking de possíveis aumentos/quedas (envio a cada 2 h)
# ======================================================

def enviar_ranking_periodico():
    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    contagem_positiva = defaultdict(int)
    contagem_negativa = defaultdict(int)

    limite_data = datetime.now(timezone.utc) - timedelta(hours=24)
    for item in historico.values():
        data_iso = item.get("data")
        try:
            dt_item = datetime.fromisoformat(data_iso.replace("Z", "+00:00"))
        except:
            continue
        if dt_item < limite_data:
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

# ======================================================
# 18. Responder /start no Telegram com tabela de top tokens + dados de mercado
# ======================================================

def get_telegram_updates(offset=None):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {}
    if offset:
        params['offset'] = offset
    try:
        resp = requests.get(url, params=params, timeout=10)
        return resp.json()
    except Exception:
        return {}

def responder_start(chat_id):
    historico = carregar_de_json(HISTORICO_ANALISES_FILE)
    contagem_positiva = defaultdict(int)
    contagem_negativa = defaultdict(int)

    limite_data = datetime.now(timezone.utc) - timedelta(hours=24)
    for item in historico.values():
        data_iso = item.get("data")
        try:
            dt_item = datetime.fromisoformat(data_iso.replace("Z", "+00:00"))
        except:
            continue
        if dt_item < limite_data:
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

    top_positivos = sorted(contagem_positiva.items(), key=lambda x: x[1], reverse=True)[:5]
    top_negativos = sorted(contagem_negativa.items(), key=lambda x: x[1], reverse=True)[:5]

    # Extrair lista de símbolos para consultar dados de mercado
    symbols_cm = [t[0] for t in top_positivos + top_negativos]
    symbols_bin = symbols_cm.copy()

    dados_cmc = coletar_dados_cmc(symbols_cm) if symbols_cm else {}
    dados_binance = coletar_dados_24h_binance(symbols_bin) if symbols_bin else {}

    def montar_tabela_mercado(tupla_lista, titulo):
        if not tupla_lista:
            return f"{titulo}\nNenhum token suficiente.\n\n"
        texto = f"{titulo}\n"
        texto += "| Token | Preço CMC (USD) | Δ24h CMC (%) | Preço Binance (USD) | Δ24h BIN (%) |\n"
        texto += "|-------|----------------|--------------|---------------------|--------------|\n"
        for token, cnt in tupla_lista:
            cmc = dados_cmc.get(token) or {}
            bin24 = dados_binance.get(token) or {}
            preco_cmc = cmc.get("preco_atual", 0)
            var_cmc = cmc.get("var_24h", 0)
            preco_bin = bin24.get("preco_atual", 0)
            var_bin = bin24.get("var_24h", 0)
            texto += f"| {token} | {preco_cmc:.4f} | {var_cmc:+.2f}% | {preco_bin:.4f} | {var_bin:+.2f}% |\n"
        texto += "\n"
        return texto

    mensagem = "*Resumo de Mercado + Top Tokens 24h*\n\n"
    mensagem += montar_tabela_mercado(top_positivos, "🚀 *Top 5 Positivos*")
    mensagem += montar_tabela_mercado(top_negativos, "📉 *Top 5 Negativos*")

    enviar_alerta_telegram(mensagem, chat_id)

def processar_telegram_commands():
    global LAST_UPDATE_ID
    updates = get_telegram_updates(LAST_UPDATE_ID)
    if not updates.get("ok"):
        return
    for result in updates.get("result", []):
        update_id = result["update_id"]
        LAST_UPDATE_ID = update_id + 1
        message = result.get("message") or {}
        text = message.get("text", "")
        chat_id = message.get("chat", {}).get("id")
        if text.strip() == "/start" and chat_id:
            logger.info("📩 Recebido /start de chat_id %s", chat_id)
            responder_start(chat_id)

# ======================================================
# 19. Função de coleta periódica (rodar a cada 5 min)
# ======================================================

def executar_coleta_periodica(vectorizer, clf):
    noticias = coletar_noticias_rss(vectorizer, clf)
    reddit_posts = coletar_reddit_praw(vectorizer, clf)
    todos_dados = noticias + reddit_posts

    if todos_dados:
        logger.info("📥 Coleta periódica: %d itens coletados (RSS+Reddit)", len(todos_dados))
        # Atualiza o histórico SEMPRE que houver novos itens
        atualizar_historico_analises(todos_dados)
        verificar_alertas(todos_dados)
    else:
        logger.info("📡 Coleta periódica: nenhum dado novo nas últimas 24h.")

# ======================================================
# 20. Função principal (setup + primeira execução)
# ======================================================

def obter_modelo():
    vectorizer, clf = carregar_modelo_local_5classes(MODELO_LOCAL_PATH)
    if vectorizer is None or clf is None:
        logger.info("🔄 Modelo local não existe. Baixando histórico e treinando...")
        if LINK_PUBLICO_DRIVE:
            baixar_json_drive(LINK_PUBLICO_DRIVE, HISTORICO_ANALISES_FILE)
        vectorizer, clf = auto_treinar_modelo_5classes(HISTORICO_ANALISES_FILE, MODELO_LOCAL_PATH)
        if vectorizer is None or clf is None:
            logger.warning("⚠️ Falha ao treinar. Usando fallback NEUTRAL para análises.")
    return vectorizer, clf

def main():
    logger.info("========== Iniciando CryptoNewsBot ==========")
    obter_lista_binance()
    vectorizer, clf = obter_modelo()

    # Execução inicial de coleta (antes de agendar)
    executar_coleta_periodica(vectorizer, clf)

    # Agendar coleta a cada 5 minutos
    schedule.every(5).minutes.do(executar_coleta_periodica, vectorizer, clf)
    logger.info("⏰ Agendado coleta a cada 5 minutos.")

    # Agendar envio de ranking a cada 2 horas
    schedule.every(2).hours.do(enviar_ranking_periodico)
    logger.info("⏰ Agendado ranking a cada 2 horas.")

    try:
        while True:
            schedule.run_pending()
            processar_telegram_commands()
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("⏹️ Execução interrompida manualmente.")

if __name__ == "__main__":
    main()
