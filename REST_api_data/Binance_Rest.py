# REST_api_data/Binance_Rest.py

import time
import requests
import pandas as pd
from datetime import datetime

pd.set_option('expand_frame_repr', False)

SYMBOL      = "BTCUSDT"
INTERVAL    = "1h"
LIMIT       = 1000
DATA_FOLDER = "."
YEARS       = 2

# how many ms in one year? ~365d
ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000
WINDOW_MS   = YEARS * ONE_YEAR_MS
now_ms      = int(time.time() * 1000)
start_ms    = now_ms - WINDOW_MS
period_ms   = 60 * 60 * 1000  # 1h

while start_ms < now_ms:
    end_ms = min(start_ms + LIMIT * period_ms, now_ms)
    url = (
        f"https://api.binance.com/api/v3/klines"
        f"?symbol={SYMBOL}"
        f"&interval={INTERVAL}"
        f"&limit={LIMIT}"
        f"&startTime={start_ms}"
        f"&endTime={end_ms}"
    )
    #print (url)
    #print(f"→ Fetching bars {start_ms} → {end_ms}")
    resp = requests.get(url)
    data = resp.json()
    if not data:
        break

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)

    # **keep open_time & close_time as raw ms!**
    # index on open_time
    df.set_index("open_time", inplace=True)

    out_path = f"{DATA_FOLDER}/{end_ms}.csv"
    df.to_csv(out_path)
    print(f"   • wrote {len(df)} rows → {out_path}")

    start_ms = end_ms + 1
    time.sleep(0.3)

print(f"✅ Done fetching {YEARS} years of {INTERVAL} bars.")




# minute data extraction
# import requests
# import time
# import pandas as pd
#
# pd.set_option('expand_frame_repr', False)
#
# base_url = 'https://api.binance.com'
# limit = 1000
# end_time = (int(time.time()) // 60) * 60 * 1000
# start_time = int(end_time - limit * 60 * 1000)
# # 获取一年数据
# while True:
#     url = base_url + '/api/v3/klines' + '?symbol=BTCUSDT&interval=1m&limit=' + str(limit) + '&startTime=' + str(
#         start_time) + '&endTime=' + str(end_time)
#     print(url)
#     resp = requests.get(url)
#     data = resp.json()
#
#     cols = [
#         "open_time", "open", "high", "low", "close", "volume",
#         "close_time", "quote_vol", "trades",
#         "taker_base_vol", "taker_quote_vol", "ignore"
#     ]
#     df = pd.DataFrame(data, columns=cols)
#     df.set_index('open_time', inplace=True)
#     print(df)
#     df.to_csv(str(end_time) + '.csv')
#
#     if len(df)<1000:
#         break
#     end_time = start_time
#     start_time = int(end_time - limit * 60 * 1000)
