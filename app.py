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

def extract_purchase_date(text):
    """OCR 텍스트에서 구매일 추출 (mm월 dd일 (요일) hh:mm 형식)"""
    # 텍스트를 줄 단위로 분할
    lines = text.strip().split('\n')
    
    # 각 줄을 검사하여 출금 관련 줄들을 찾고 제외
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if line and '출금' not in line and '완료되었습니다' not in line:
            filtered_lines.append(line)
    
    # 필터링된 텍스트 재구성
    filtered_text = '\n'.join(filtered_lines)
    print(f"필터링 후 텍스트 (출금 관련 줄 제거):")
    print(filtered_text)
    print("=" * 50)
    
    # 구매일시 패턴을 먼저 찾기 (6월 30일 (월) 10:30 형식)
    purchase_datetime_pattern = r'(\d{1,2})월\s*(\d{1,2})일\s*\([가-힣]\)\s*(\d{1,2}):(\d{2})'
    
    # 전체 텍스트에서 모든 날짜/시간 매치를 찾기
    all_matches = []
    for i, line in enumerate(lines):
        matches = re.findall(purchase_datetime_pattern, line)
        for match in matches:
            month, day, hour, minute = match
            # 출금 관련 줄이 아닌 경우만 추가
            if '출금' not in line and '완료되었습니다' not in line:
                all_matches.append((month, day, hour, minute, line, i))
                print(f"발견된 날짜/시간: {month}월 {day}일 {hour}:{minute} (줄 {i}: {line.strip()})")
    
    if all_matches:
        # 가장 위쪽에 나오는 날짜/시간을 사용 (거래일시가 먼저 나옴)
        month, day, hour, minute, source_line, line_num = all_matches[0]
        try:
            current_year = datetime.now().year
            purchase_date = datetime(current_year, int(month), int(day), int(hour), int(minute))
            result = purchase_date.strftime('%Y-%m-%d %H:%M')
            print(f"구매일시 추출 성공: {month}월 {day}일 {hour}:{minute} → {result}")
            print(f"출처: {source_line.strip()}")
            return result
        except ValueError as e:
            print(f"구매일시 파싱 오류: {e}")
    
    # 시간 정보가 있는 패턴 우선 검색 (구매 관련 정보만)
    # 예: "6월 30일 (월) 10:30"
    datetime_pattern = r'(\d{1,2})월\s*(\d{1,2})일\s*(?:\([가-힣]\))?\s*(\d{1,2}):(\d{2})'
    datetime_matches = re.findall(datetime_pattern, filtered_text)
    
    if datetime_matches:
        # 가장 마지막에 나오는 날짜/시간 사용 (구매 정보는 보통 하단에 위치)
        month, day, hour, minute = datetime_matches[-1]
        try:
            current_year = datetime.now().year
            purchase_date = datetime(current_year, int(month), int(day), int(hour), int(minute))
            result = purchase_date.strftime('%Y-%m-%d %H:%M')
            print(f"구매일 시간 포함 패턴 매칭: {datetime_matches[-1]} → {result}")
            return result
        except ValueError as e:
            print(f"시간 포함 날짜 파싱 오류: {e}")
    
    # 시간 정보가 없는 패턴도 시도 (구매 관련 정보 근처만)
    date_only_pattern = r'(\d{1,2})월\s*(\d{1,2})일\s*(?:\([가-힣]\))?'
    date_matches = re.findall(date_only_pattern, filtered_text)
    
    if date_matches:
        # 가장 마지막 날짜 사용 (구매 정보는 보통 하단에 위치)
        month, day = date_matches[-1]
        try:
            current_year = datetime.now().year
            # 시간 정보가 없으면 00:00으로 설정
            purchase_date = datetime(current_year, int(month), int(day), 0, 0)
            result = purchase_date.strftime('%Y-%m-%d %H:%M')
            print(f"구매일 날짜만 패턴 매칭: {date_matches[-1]} → {result}")
            return result
        except ValueError as e:
            print(f"날짜만 파싱 오류: {e}")
    
    print(f"구매일 패턴 매칭 실패")
    return None

def extract_korean_trading_info(text):
    """한국어 거래 앱에서 정보 추출"""
    transactions = []
    
    # 모든 매칭 결과를 출력해서 디버깅
    print(f"=== Full OCR Text ===")
    print(text)
    print(f"=== End OCR Text ===")
    
    # 구매일 추출
    purchase_date = extract_purchase_date(text)
    print(f"Extracted purchase date: {purchase_date}")
    
    # 종목 심볼 찾기 - 더 정확한 패턴 사용
    # NVDA처럼 독립적으로 나오는 3-5글자 대문자만
    symbol_pattern = r'\b([A-Z]{3,5})\b'
    symbol_matches = re.findall(symbol_pattern, text)
    
    # 잘못된 심볼 제외 (LTE, APP 등 일반적인 단어들)
    excluded_symbols = {'LTE', 'APP', 'CEO', 'USD', 'KRW', 'GMT'}
    valid_symbols = [s for s in symbol_matches if s not in excluded_symbols]
    
    # 진행 중인 거래 화면 감지 패턴
    progress_keywords = ['구매 접수', '구매 처리', '구매 완료', '출금 완료', '출금 중']
    is_progress_screen = any(keyword in text for keyword in progress_keywords)
    
    if is_progress_screen:
        print("⚠️  진행 중인 거래 화면이 감지되었습니다.")
        # 진행 중 화면에서는 예상 정보만 추출 가능
        
        # 체결가 패턴 (1주당 체결가: 177.05달러)
        price_per_share_pattern = r'1주당\s*체결가[:\s]*(\d+(?:\.\d+)?)\s*달러'
        price_matches = re.findall(price_per_share_pattern, text)
        
        # 구매 금액과 주식수로 역산
        buy_amount_pattern = r'구매\s*금액[:\s]*(\d{1,3}(?:,\d{3})*)\s*원'
        buy_amount_matches = re.findall(buy_amount_pattern, text)
        
        # 주식수 패턴
        shares_pattern = r'(\d+(?:\.\d+)?)\s*주'
        shares_matches = re.findall(shares_pattern, text)
        
        if valid_symbols and (price_matches or (buy_amount_matches and shares_matches)):
            symbol = valid_symbols[0]
            
            # 가격 정보 결정
            if price_matches:
                price_usd = float(price_matches[0])
            elif buy_amount_matches and shares_matches:
                # 구매 금액과 주식수로 달러 가격 역산
                amount_krw = float(buy_amount_matches[0].replace(',', ''))
                quantity = float(shares_matches[0])
                price_krw_per_share = amount_krw / quantity
                price_usd = price_krw_per_share / 1300  # 대략적인 환율
            else:
                print("가격 정보를 찾을 수 없습니다.")
                return transactions
                
            # 주식수 정보
            if shares_matches:
                quantity = float(shares_matches[0])
            else:
                # 구매 금액과 체결가로 주식수 역산
                if buy_amount_matches and price_matches:
                    amount_krw = float(buy_amount_matches[0].replace(',', ''))
                    price_usd = float(price_matches[0])
                    quantity = amount_krw / (price_usd * 1300)  # 대략적 계산
                else:
                    print("주식수 정보를 찾을 수 없습니다.")
                    return transactions
            
            transactions.append({
                'ticker': symbol.upper(),
                'type': 'buy',  # 진행 중 화면은 대부분 구매
                'quantity': round(quantity, 6),
                'price': round(price_usd, 2),
                'date': purchase_date[:10] if purchase_date else datetime.now().strftime('%Y-%m-%d'),
                'purchase_datetime': purchase_date,
                'original_amount_krw': float(buy_amount_matches[0].replace(',', '')) if buy_amount_matches else None,
                'extracted_dollar_price': float(price_matches[0]) if price_matches else None,
                'is_progress_screen': True  # 진행 중 화면임을 표시
            })
            
            print(f"진행 중 화면에서 추출된 정보:")
            print(f"  종목: {symbol}")
            print(f"  수량: {quantity}")
            print(f"  가격: ${price_usd}")
            print(f"  구매금액: {buy_amount_matches[0] if buy_amount_matches else 'N/A'}원")
            
            return transactions
    
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
                    'date': purchase_date[:10] if purchase_date else datetime.now().strftime('%Y-%m-%d'),  # YYYY-MM-DD 형식만 사용
                    'purchase_datetime': purchase_date,  # 전체 날짜/시간 정보
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
    # Generate a unique version string (e.g., timestamp) for cache busting
    cache_buster = datetime.now().strftime("%Y%m%d%H%M%S")
    return render_template("index.html", cache_buster=cache_buster)

@app.route("/api/stocks/v1/transactions", methods=["POST"])
def add_transaction():
    req = request.get_json()
    symbol = req["ticker"]
    price = float(req["price"])
    quantity = float(req["quantity"])
    tx_type = req["type"]
    date = req.get("date") or datetime.today().strftime("%Y-%m-%d")
    purchase_datetime = req.get("purchase_datetime")  # OCR에서 추출된 구매일시

    data = load_data()
    transactions = data.setdefault("transactions", [])
    transaction_data = {
        "ticker": symbol,
        "type": tx_type,
        "price": price,
        "quantity": quantity,
        "date": date
    }
    
    # purchase_datetime이 있으면 추가
    if purchase_datetime:
        transaction_data["purchase_datetime"] = purchase_datetime
    
    transactions.append(transaction_data)

    update_portfolio(data)
    save_data(data)
    return jsonify({"message": "Transaction recorded."})

@app.route("/api/stocks/v1/transactions", methods=["GET"])
def get_transactions():
    data = load_data()
    return jsonify(data.get("transactions", []))

@app.route("/api/stocks/v1/transactions/<int:index>", methods=["PUT"])
def update_transaction(index):
    req = request.get_json()
    data = load_data()
    transactions = data.get("transactions", [])
    
    if not (0 <= index < len(transactions)):
        return jsonify({"error": "Transaction not found."}), 404
    
    # 새로운 거래 정보로 업데이트
    symbol = req["ticker"]
    price = float(req["price"])
    quantity = float(req["quantity"])
    tx_type = req["type"]
    date = req.get("date") or datetime.today().strftime("%Y-%m-%d")
    
    transactions[index] = {
        "ticker": symbol,
        "type": tx_type,
        "price": price,
        "quantity": quantity,
        "date": date
    }
    
    update_portfolio(data)
    save_data(data)
    return jsonify({"message": "Transaction updated."})

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
        
        # 이미지 전처리 - OCR 품질 향상
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)  # 밝기 20% 증가
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # 대비 30% 증가
        
        # OCR 수행 (한국어 + 영어, PSM 6 모드)
        try:
            raw_text = pytesseract.image_to_string(image, lang='kor+eng', config='--psm 6')
            print(f"OCR Raw Text (enhanced): {raw_text}")  # 디버깅용
        except Exception as e:
            print(f"OCR with enhanced settings failed, trying basic: {e}")
            try:
                raw_text = pytesseract.image_to_string(image, lang='kor+eng')
                print(f"OCR Raw Text (basic): {raw_text}")
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
        
        # 이미지 전처리 - OCR 품질 향상
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)  # 밝기 20% 증가
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # 대비 30% 증가
        
        try:
            raw_text = pytesseract.image_to_string(image, lang='kor+eng', config='--psm 6')
            print(f"Sample OCR Raw Text (enhanced): {raw_text}")
        except Exception as e:
            print(f"Sample OCR with enhanced settings failed, trying basic: {e}")
            try:
                raw_text = pytesseract.image_to_string(image, lang='kor+eng')
                print(f"Sample OCR Raw Text (basic): {raw_text}")
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