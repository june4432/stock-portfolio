import os
import json
import yfinance as yf
from flask import Flask, request, jsonify, render_template
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")
DATA_FILE = "data/portfolio.json"

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9010)