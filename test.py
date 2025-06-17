import os
import logging
import time
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import numpy as np

# --- Configuração do Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Carregar Variáveis de Ambiente ---
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# --- Configurações Globais (Cruciais para Lucratividade) ---
# Você pode ajustar todos os parâmetros aqui
CONFIG = {
    "FEE_RATE": 0.001,  # Taxa padrão da Binance (0.1%). Use 0.00075 se pagar com BNB.
    "DESIRED_NET_PROFIT_PERCENTAGE": 0.04,  # Lucro líquido desejado (4%).
    "STOP_LOSS_PERCENTAGE": 0.03,  # Perda máxima aceitável (3%).
    "CAPITAL_PER_TRADE_PERCENTAGE": 0.05,  # Usar 5% do capital USDT por trade.
    "VOLATILITY_WINDOW_HOURS": 4, # Janela para análise de volatilidade.
    "PRICE_CHANGE_THRESHOLD": 12.0, # Variação de preço mínima de 12% para ser considerado volátil.
    "RSI_PERIOD": 14,
    "RSI_OVERSOLD_THRESHOLD": 30,
    "PRICE_DROP_FROM_PEAK_PERCENTAGE": 0.05, # Queda de 5% do pico recente para gatilho de compra.
    "KLINES_INTERVAL_ANALYSIS": Client.KLINE_INTERVAL_1MINUTE,
    "BASE_ASSET": "USDT"
}

class TradingBot:
    def __init__(self, api_key, api_secret, config):
        self.client = Client(api_key, api_secret)
        self.config = config
        self.symbol_info_cache = {}
        self.current_position = None # Para garantir apenas uma operação por vez

        logging.info("Bot de trade inicializado com sucesso.")

    def _get_symbol_info(self, symbol):
        """Busca e armazena em cache as informações de precisão de um símbolo."""
        if symbol not in self.symbol_info_cache:
            try:
                info = self.client.get_symbol_info(symbol)
                if not info:
                    logging.warning(f"Não foi possível obter informações para o símbolo {symbol}")
                    return None
                self.symbol_info_cache[symbol] = info
                logging.info(f"Informações para {symbol} cacheadas.")
            except BinanceAPIException as e:
                logging.error(f"Erro ao buscar informações para {symbol}: {e}")
                return None
        return self.symbol_info_cache[symbol]

    def _format_value(self, value, step_size):
        """Formata um valor para a precisão correta de acordo com o step_size."""
        precision = int(round(-math.log(float(step_size), 10), 0))
        return f"{value:.{precision}f}"

    def _get_usdt_balance(self):
        """Retorna o saldo disponível em USDT."""
        try:
            balance = self.client.get_asset_balance(asset=self.config["BASE_ASSET"])
            return float(balance['free'])
        except BinanceAPIException as e:
            logging.error(f"Erro ao obter saldo USDT: {e}")
            return 0

    def _procurar_oportunidades(self):
        """Escaneia todos os pares USDT em busca de alta volatilidade."""
        logging.info("Buscando pares com alta volatilidade...")
        try:
            all_tickers = self.client.get_ticker()
            usdt_pairs = [t for t in all_tickers if t['symbol'].endswith(self.config["BASE_ASSET"])]
            
            watchlist = []
            for ticker in usdt_pairs:
                symbol = ticker['symbol']
                # Filtro para evitar moedas de baixa liquidez ou stablecoins pareadas
                if 'UP' in symbol or 'DOWN' in symbol or 'BUSD' in symbol or float(ticker.get('quoteVolume', 0)) < 10000000:
                    continue

                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=f'{self.config["VOLATILITY_WINDOW_HOURS"]}h',
                    limit=1
                )
                if not klines: continue
                
                high = float(klines[0][2])
                low = float(klines[0][3])
                price_change = ((high - low) / low) * 100

                if price_change > self.config["PRICE_CHANGE_THRESHOLD"]:
                    watchlist.append({'symbol': symbol, 'volatility': price_change})
                    logging.info(f"Candidato encontrado: {symbol} com volatilidade de {price_change:.2f}%")

            # Ordena por volatilidade e retorna os melhores
            watchlist.sort(key=lambda x: x['volatility'], reverse=True)
            return [item['symbol'] for item in watchlist[:15]] # Top 15% pode ser muito, limitamos a 15
        except BinanceAPIException as e:
            logging.error(f"Erro ao procurar oportunidades: {e}")
            return []

    def _analisar_e_executar_compra(self, symbol):
        """Analisa um símbolo e executa a compra se os critérios forem atendidos."""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=self.config["KLINES_INTERVAL_ANALYSIS"], limit=100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['high'] = pd.to_numeric(df['high'])
            df['close'] = pd.to_numeric(df['close'])

            # Análise Técnica
            df['rsi'] = ta.rsi(df['close'], length=self.config["RSI_PERIOD"])
            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            recent_peak = df['high'].rolling(window=50).max().iloc[-1] # Pico das últimas 50 velas de 1 min

            # Gatilho de Compra
            is_price_dropped = current_price < recent_peak * (1 - self.config["PRICE_DROP_FROM_PEAK_PERCENTAGE"])
            is_rsi_oversold = current_rsi < self.config["RSI_OVERSOLD_THRESHOLD"]
            
            logging.info(f"Analisando {symbol}: Preço Atual={current_price:.4f}, RSI={current_rsi:.2f}, Pico Recente={recent_peak:.4f}")

            if is_price_dropped and is_rsi_oversold:
                logging.info(f"GATILHO DE COMPRA PARA {symbol} ACIONADO!")
                
                usdt_balance = self._get_usdt_balance()
                capital_to_use = usdt_balance * self.config["CAPITAL_PER_TRADE_PERCENTAGE"]

                if capital_to_use < 10: # Mínimo da Binance é geralmente 10 USDT
                    logging.warning(f"Capital por trade ({capital_to_use:.2f} USDT) é menor que o mínimo necessário.")
                    return None

                logging.info(f"Executando ordem de compra a mercado para {symbol} com {capital_to_use:.2f} USDT.")
                order = self.client.order_market_buy(symbol=symbol, quoteOrderQty=capital_to_use)
                
                # Calcula o preço médio de compra a partir dos 'fills'
                avg_price = sum([float(f['price']) * float(f['qty']) for f in order['fills']]) / sum([float(f['qty']) for f in order['fills']])
                quantity = float(order['executedQty'])

                logging.info(f"ORDEM DE COMPRA EXECUTADA: {quantity:.6f} {symbol} @ Preço Médio: {avg_price:.6f}")
                
                self.current_position = {'symbol': symbol, 'quantity': quantity, 'purchase_price': avg_price}
                return self.current_position

        except BinanceAPIException as e:
            logging.error(f"Erro de API durante a análise/compra de {symbol}: {e}")
        except Exception as e:
            logging.error(f"Erro inesperado durante a análise/compra de {symbol}: {e}")
        
        return None

    def _criar_ordem_oco_venda(self):
        """Cria uma ordem OCO para a posição atual, calculando o break-even primeiro."""
        if not self.current_position:
            return

        symbol = self.current_position['symbol']
        quantity = self.current_position['quantity']
        preco_de_compra = self.current_position['purchase_price']
        fee_rate = self.config["FEE_RATE"]
        
        # --- Lógica crucial de cálculo de lucratividade ---
        # 1. Cálculo do Ponto de Equilíbrio (Break-Even)
        preco_break_even = preco_de_compra / (1 - fee_rate) # Cobre a taxa de venda
        preco_break_even_real = preco_de_compra / ((1 - fee_rate)**2) # Cobre ambas as taxas (compra e venda)
        
        # 2. Cálculo do Preço de Take Profit (Lucro Líquido)
        preco_take_profit = preco_break_even_real * (1 + self.config["DESIRED_NET_PROFIT_PERCENTAGE"])
        
        # 3. Cálculo do Preço de Stop Loss
        preco_stop_loss = preco_de_compra * (1 - self.config["STOP_LOSS_PERCENTAGE"])
        
        logging.info(f"--- CÁLCULO DE ORDEM OCO PARA {symbol} ---")
        logging.info(f"Preço de Compra: {preco_de_compra:.6f}")
        logging.info(f"Ponto de Equilíbrio (Break-Even): {preco_break_even_real:.6f}")
        logging.info(f"Meta de Lucro Líquido (Take Profit): {preco_take_profit:.6f}")
        logging.info(f"Meta de Parada (Stop Loss): {preco_stop_loss:.6f}")
        
        # Formatação dos preços para a precisão da API
        info = self._get_symbol_info(symbol)
        if not info: return

        price_filter = next(f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER')
        lot_size_filter = next(f for f in info['filters'] if f['filterType'] == 'LOT_SIZE')

        tick_size = price_filter['tickSize']
        step_size = lot_size_filter['stepSize']

        quantity_str = self._format_value(quantity, step_size)
        take_profit_price_str = self._format_value(preco_take_profit, tick_size)
        stop_loss_price_str = self._format_value(preco_stop_loss, tick_size)

        try:
            logging.info("Enviando ordem OCO para a Binance...")
            self.client.create_oco_order(
                symbol=symbol,
                side=Client.SIDE_SELL,
                quantity=quantity_str,
                price=take_profit_price_str,  # Preço da ordem LIMIT (Take Profit)
                stopPrice=stop_loss_price_str,  # Preço de gatilho do Stop Loss
                stopLimitPrice=stop_loss_price_str,  # Preço limite da ordem Stop Loss (pode ser o mesmo do gatilho)
                stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC
            )
            logging.info(f"Ordem OCO para {symbol} criada com sucesso!")
        except BinanceAPIException as e:
            logging.error(f"Falha ao criar ordem OCO para {symbol}: {e}")
            self.current_position = None # Libera para nova tentativa se a OCO falhar

    def _monitorar_posicao(self):
        """Verifica se a posição aberta foi fechada."""
        if not self.current_position:
            return True # Nenhuma posição para monitorar, está livre

        symbol = self.current_position['symbol']
        try:
            # Uma forma simples de verificar é checar se há ordens abertas para o par.
            # Uma abordagem mais robusta envolveria rastrear o ID da ordem OCO.
            open_orders = self.client.get_open_orders(symbol=symbol)
            if not open_orders:
                logging.info(f"Posição em {symbol} foi fechada. Procurando nova oportunidade.")
                self.current_position = None
                return True
            else:
                logging.info(f"Aguardando fechamento da posição em {symbol}. Ordens abertas: {len(open_orders)}")
                return False
        except BinanceAPIException as e:
            logging.error(f"Erro ao monitorar posição de {symbol}: {e}")
            # Em caso de erro, por segurança, consideramos a posição como ativa para evitar abrir outra.
            return False

    def run(self):
        """Loop principal do bot."""
        while True:
            try:
                # Se não há posição aberta, procure uma.
                if self._monitorar_posicao():
                    watchlist = self._procurar_oportunidades()
                    if watchlist:
                        for symbol in watchlist:
                            if not self.current_position: # Garante que não abrimos outra trade enquanto iteramos
                                position_opened = self._analisar_e_executar_compra(symbol)
                                if position_opened:
                                    self._criar_ordem_oco_venda()
                                    break # Sai do loop de watchlist para começar a monitorar a nova posição
                    else:
                        logging.info("Nenhuma oportunidade encontrada. Aguardando...")
                
                # Aguarda antes do próximo ciclo para não sobrecarregar a API
                time.sleep(60)

            except Exception as e:
                logging.critical(f"Erro crítico no loop principal: {e}")
                time.sleep(120) # Espera mais tempo em caso de erro grave

if __name__ == "__main__":
    import math # Import math aqui para a função _format_value
    
    if not API_KEY or not API_SECRET:
        logging.error("As chaves da API (BINANCE_API_KEY, BINANCE_API_SECRET) não foram encontradas no arquivo .env")
    else:
        bot = TradingBot(API_KEY, API_SECRET, CONFIG)
        bot.run()