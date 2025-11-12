import yfinance as yf

print("TEST 6501.T:", not yf.download("6501.T", period="1mo", progress=False).empty)
print("TEST 7203.T:", not yf.download("7203.T", period="1mo", progress=False).empty)
print("TEST 3778.T:", not yf.download("3778.T", period="1mo", progress=False).empty)