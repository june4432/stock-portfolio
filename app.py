import os
import json
import re
import yfinance as yf
import feedparser
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from werkzeug.utils import secure_filename
from urllib.parse import urljoin, urlparse

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
DATA_FILE = "data/portfolio.json"

# 업로드 허용 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_ocr_text(text):
    """OCR 텍스트에서 거래 정보를 추출"""
    lines = text.strip().split('\n')
    transactions = []
    
    print(f"Processing OCR text lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"Line {i}: '{line.strip()}'")
    
    # 한국어 앱 패턴 처리
    korean_info = extract_korean_trading_info(text)
    if korean_info:
        transactions.extend(korean_info)
        return transactions
    
    # 기존 영문 패턴
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 기본 패턴: SYMBOL BUY/SELL QUANTITY @ PRICE
        pattern = r'([A-Z]{1,5})\s+(BUY|SELL)\s+(\d+(?:\.\d+)?)\s*@?\s*\$?(\d+(?:\.\d{1,2})?)'
        match = re.search(pattern, line, re.IGNORECASE)
        
        if match:
            symbol, action, quantity, price = match.groups()
            try:
                transactions.append({
                    'ticker': symbol.upper(),
                    'type': action.lower(),
                    'quantity': float(quantity),
                    'price': float(price),
                    'date': datetime.now().strftime('%Y-%m-%d')
                })
            except ValueError:
                continue
    
    return transactions

def extract_korean_trading_info(text):
    """한국어 거래 앱에서 정보 추출"""
    transactions = []
    
    # 모든 매칭 결과를 출력해서 디버깅
    print(f"=== Full OCR Text ===")
    print(text)
    print(f"=== End OCR Text ===")
    
    # 종목 심볼 찾기 - 더 정확한 패턴 사용
    # NVDA처럼 독립적으로 나오는 3-5글자 대문자만
    symbol_pattern = r'\b([A-Z]{3,5})\b'
    symbol_matches = re.findall(symbol_pattern, text)
    
    # 잘못된 심볼 제외 (LTE, APP 등 일반적인 단어들)
    excluded_symbols = {'LTE', 'APP', 'CEO', 'USD', 'KRW', 'GMT'}
    valid_symbols = [s for s in symbol_matches if s not in excluded_symbols]
    
    # 주식수 패턴 (0.093438주 형태)
    shares_pattern = r'(\d+(?:\.\d+)?)\s*주'
    shares_matches = re.findall(shares_pattern, text)
    
    # 구매 금액 패턴 - 더 정확하게 매칭
    # "구매 접수 금액 20,000원" 또는 "구매 금액 19,991원" 형태
    buy_amount_pattern = r'구매.*?(\d{1,3}(?:,\d{3})*)\s*원'
    buy_amount_matches = re.findall(buy_amount_pattern, text)
    
    # 일반 금액 패턴도 백업으로 사용
    amount_pattern = r'(\d{1,3}(?:,\d{3})*)\s*원'
    amount_matches = re.findall(amount_pattern, text)
    
    # 달러 가격 패턴 (체결가)
    dollar_pattern = r'(\d+(?:\.\d+)?)\s*달러'
    dollar_matches = re.findall(dollar_pattern, text)
    
    # 구매/판매 키워드 감지
    buy_keywords = ['구매', '매수', '구입']
    sell_keywords = ['판매', '매도', '판']
    
    action_type = 'buy'  # 기본값
    for keyword in buy_keywords:
        if keyword in text:
            action_type = 'buy'
            break
    for keyword in sell_keywords:
        if keyword in text:
            action_type = 'sell'
            break
    
    print(f"Valid Symbols: {valid_symbols}")
    print(f"Shares: {shares_matches}")
    print(f"Buy Amount: {buy_amount_matches}")
    print(f"All Amount: {amount_matches}")
    print(f"Dollar Price: {dollar_matches}")
    print(f"Action: {action_type}")
    
    if valid_symbols and shares_matches:
        try:
            symbol = valid_symbols[0]
            quantity = float(shares_matches[0])
            
            # 구매 금액 우선, 없으면 일반 금액에서 적절한 것 선택
            amount = None
            if buy_amount_matches:
                amount_str = buy_amount_matches[0].replace(',', '')
                amount = float(amount_str)
            elif amount_matches:
                # 20,000원이나 19,991원 같은 합리적인 금액 선택
                for amt in amount_matches:
                    amt_num = float(amt.replace(',', ''))
                    if 1000 <= amt_num <= 10000000:  # 1천원~1천만원 범위
                        amount = amt_num
                        break
            
            if amount and quantity:
                # 달러 가격이 있으면 사용, 없으면 계산
                if dollar_matches:
                    price_usd = float(dollar_matches[0])
                else:
                    # 1주당 가격 계산 (원화 → 달러)
                    price_krw = amount / quantity
                    price_usd = price_krw / 1300  # 대략적인 환율
                
                transactions.append({
                    'ticker': symbol.upper(),
                    'type': action_type,
                    'quantity': quantity,
                    'price': round(price_usd, 2),
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'original_amount_krw': amount,
                    'extracted_dollar_price': dollar_matches[0] if dollar_matches else None
                })
                
        except (ValueError, IndexError) as e:
            print(f"Error processing Korean trading info: {e}")
    
    return transactions

# JSON 파일 불러오기/저장 함수
def load_data():
    if not os.path.exists(DATA_FILE):
        return {"portfolio": {}, "transactions": []}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        os.rename(DATA_FILE, DATA_FILE + ".corrupted.json")
        return {"portfolio": {}, "transactions": []}

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stocks/v1/transactions", methods=["POST"])
def add_transaction():
    req = request.get_json()
    symbol = req["ticker"]
    price = float(req["price"])
    quantity = float(req["quantity"])
    tx_type = req["type"]
    date = req.get("date") or datetime.today().strftime("%Y-%m-%d")

    data = load_data()
    transactions = data.setdefault("transactions", [])
    transactions.append({
        "ticker": symbol,
        "type": tx_type,
        "price": price,
        "quantity": quantity,
        "date": date
    })

    update_portfolio(data)
    save_data(data)
    return jsonify({"message": "Transaction recorded."})

@app.route("/api/stocks/v1/transactions", methods=["GET"])
def get_transactions():
    data = load_data()
    return jsonify(data.get("transactions", []))

@app.route("/api/stocks/v1/transactions/<int:index>", methods=["DELETE"])
def delete_transaction(index):
    data = load_data()
    transactions = data.get("transactions", [])
    if 0 <= index < len(transactions):
        del transactions[index]
        update_portfolio(data)
        save_data(data)
        return jsonify({"message": "Transaction deleted."})
    return jsonify({"error": "Transaction not found."}), 404

@app.route("/api/stocks/v1/portfolio", methods=["GET"])
def get_portfolio():
    data = load_data()
    return jsonify(data.get("portfolio", {}))

@app.route("/api/stocks/v1/portfolio/<symbol>", methods=["DELETE"])
def delete_stock(symbol):
    data = load_data()
    transactions = data.get("transactions", [])
    transactions = [tx for tx in transactions if tx["ticker"] != symbol]
    data["transactions"] = transactions
    update_portfolio(data)
    save_data(data)
    return jsonify({"message": f"{symbol} deleted from portfolio."})

@app.route("/api/stocks/v1/prices", methods=["GET"])
def get_current_prices():
    data = load_data()
    portfolio = data.get("portfolio", {})
    result = {}

    for symbol in portfolio.keys():
        try:
            stock = yf.Ticker(symbol)
            result[symbol] = stock.info["regularMarketPrice"]
        except Exception as e:
            print(f"[ERROR] 가격 정보 실패 - {symbol}: {e}")
            result[symbol] = None

    return jsonify(result)

def update_portfolio(data):
    transactions = data.get("transactions", [])
    portfolio = {}
    for tx in transactions:
        t_symbol = tx["ticker"]
        t_price = float(tx["price"])
        t_quantity = float(tx["quantity"])
        t_type = tx["type"]

        if t_type == "buy":
            if t_symbol in portfolio:
                prev_qty = portfolio[t_symbol]["shares"]
                prev_avg = portfolio[t_symbol]["average_price"]
                total_qty = prev_qty + t_quantity
                total_cost = (prev_avg * prev_qty) + (t_price * t_quantity)
                new_avg = round(total_cost / total_qty, 4)
                portfolio[t_symbol] = {"shares": total_qty, "average_price": new_avg}
            else:
                portfolio[t_symbol] = {"shares": t_quantity, "average_price": t_price}
        elif t_type == "sell":
            if t_symbol in portfolio and portfolio[t_symbol]["shares"] >= t_quantity:
                portfolio[t_symbol]["shares"] -= t_quantity
                if portfolio[t_symbol]["shares"] == 0:
                    del portfolio[t_symbol]

    data["portfolio"] = portfolio


@app.route("/api/stocks/v1/symbol-search")
def symbol_search():
    query = request.args.get("q", "").lower()
    with open("static/symbols.json", encoding="utf-8") as f:
        all_symbols = json.load(f)  # [{"symbol": "AAPL", "name": "Apple Inc."}, ...]
    matches = [s for s in all_symbols if query in s["symbol"].lower() or query in s["name"].lower()]
    return jsonify(matches[:10])

@app.route("/api/stocks/v1/upload-screenshot", methods=["POST"])
def upload_screenshot():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # 이미지 처리
        image = Image.open(file.stream)
        
        # OCR 수행 (한국어 + 영어)
        try:
            raw_text = pytesseract.image_to_string(image, lang='kor+eng')
            print(f"OCR Raw Text: {raw_text}")  # 디버깅용
        except Exception as e:
            print(f"OCR with kor+eng failed, trying eng only: {e}")
            try:
                raw_text = pytesseract.image_to_string(image, lang='eng')
                print(f"OCR Raw Text (eng): {raw_text}")
            except Exception as e2:
                return jsonify({'error': f'OCR processing failed: {str(e2)}'}), 500
        
        # 텍스트에서 거래 정보 추출
        parsed_transactions = process_ocr_text(raw_text)
        
        return jsonify({
            'success': True,
            'raw_text': raw_text,
            'parsed_data': parsed_transactions,
            'count': len(parsed_transactions)
        })
        
    except Exception as e:
        print(f"Screenshot upload error: {str(e)}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route("/api/stocks/v1/save-sample", methods=["POST"])
def save_sample():
    """샘플 파일 저장용 임시 엔드포인트"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # 샘플 파일 저장 디렉토리 생성
        sample_dir = 'samples'
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        # 안전한 파일명으로 저장
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(sample_dir, filename)
        
        # 파일 저장
        file.save(filepath)
        
        # 이미지 처리 및 OCR
        image = Image.open(filepath)
        try:
            raw_text = pytesseract.image_to_string(image, lang='kor+eng')
            print(f"Sample OCR Raw Text: {raw_text}")
        except Exception as e:
            print(f"Sample OCR with kor+eng failed, trying eng only: {e}")
            try:
                raw_text = pytesseract.image_to_string(image, lang='eng')
                print(f"Sample OCR Raw Text (eng): {raw_text}")
            except Exception as e2:
                raw_text = f"OCR Error: {str(e2)}"
        
        # 이미지 정보 수집
        image_info = {
            'filename': filename,
            'filepath': filepath,
            'size': image.size,
            'format': image.format,
            'mode': image.mode,
            'file_size': os.path.getsize(filepath)
        }
        
        return jsonify({
            'success': True,
            'message': f'Sample saved as {filename}',
            'image_info': image_info,
            'raw_text': raw_text
        })
        
    except Exception as e:
        print(f"Sample save error: {str(e)}")
        return jsonify({'error': f'Save failed: {str(e)}'}), 500

@app.route("/api/stocks/v1/news", methods=["GET"])
def get_stock_news():
    try:
        data = load_data()
        portfolio = data.get("portfolio", {})
        
        if not portfolio:
            return jsonify([])
        
        all_news = []
        
        for symbol in list(portfolio.keys())[:5]:  # 최대 5개 종목
            try:
                # Seeking Alpha RSS 피드
                feed_url = f"https://seekingalpha.com/api/sa/combined/{symbol}.xml"
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:3]:  # 종목당 최대 3개 뉴스
                    all_news.append({
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.get('published', ''),
                        'symbol': symbol,
                        'summary': entry.get('summary', '')[:200] + '...' if entry.get('summary') else ''
                    })
            except Exception as e:
                print(f"[WARNING] News fetch failed for {symbol}: {e}")
                continue
        
        # 날짜순 정렬 (최신순)
        try:
            all_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        except Exception as e:
            print(f"[WARNING] Could not sort news items due to parsing error: {e}")
        
        return jsonify(all_news[:10])  # 최대 10개 뉴스
        
    except Exception as e:
        print(f"[ERROR] News API error: {e}")
        return jsonify([])

@app.route("/api/stocks/v1/asset-history", methods=["GET"])
def get_asset_history():
    try:
        data = load_data()
        portfolio = data.get("portfolio", {})
        transactions = data.get("transactions", [])
        
        if not portfolio:
            return jsonify([])
        
        # 종목별로 시간대별 자산 값 계산
        history_data = {}
        symbols = list(portfolio.keys())
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1mo", interval="1d")
                
                if not hist.empty:
                    history_data[symbol] = []
                    for date, row in hist.iterrows():
                        shares = portfolio[symbol]["shares"]
                        value = shares * row['Close']
                        history_data[symbol].append({
                            'date': date.strftime('%Y-%m-%d'),
                            'value': float(value),
                            'price': float(row['Close'])
                        })
            except Exception as e:
                print(f"[ERROR] History fetch failed for {symbol}: {e}")
                continue
        
        return jsonify(history_data)
        
    except Exception as e:
        print(f"[ERROR] Asset history error: {e}")
        return jsonify({})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9010)