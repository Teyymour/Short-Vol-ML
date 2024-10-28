import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas_market_calendars import get_calendar

polygon_api_key = os.getenv("POLYGON_API_KEY")

calendar = get_calendar("NYSE")
trading_dates = calendar.schedule(start_date="2023-04-20", end_date=datetime.today()).index.strftime("%Y-%m-%d").values

def build_spread_backtest_dataset(dates, ticker, index_ticker, options_ticker, trade_time, move_adjustment, spread_width):
    """
    Builds a comprehensive dataset to backtest the base strategy using a walk-forward approach, 
    handling everything from strike selection to fetching quotes.

    Parameters:
    -----------
    dates : list of str
        A list of dates (in "YYYY-MM-DD" format) for which the backtest dataset is generated.
    ticker : str
        The ticker of the underlying asset
    index_ticker : str
        The ticker of the volatility index
    options : str
        The ticker of underlying of the options
    trade_time : str
        The time (EST) at which to take the trade
    move_adjustment : float
        A float representing how much to discount the implied move 
        (e.g 0.5 would discount by 50%)
    spread_width : int
        An integer representing how many strikes apart the long and
        short option will be 

    Returns:
    --------
    backtest_dataset : pandas.DataFrame. Importnat columns are:
        - cost: The cost of the spread (i.e the premium received for selling the spread).
        - direction: The trading direction (0 for negative trend, 1 for positive).
        - side: Type of spread ('call' or 'put') based on the direction.
        - underlying_closing_price: The closing price of the underlying asset on that date.
        - expected_move: The calculated expected move for the underlying asset, expressed as
          a decimal proportion of its current price.
        - underlying_price_at_trade: Price of the underlying asset at trade time.
        - lower_price, upper_price: Calculated bounds for the expected price move.
        - vix1d_value: The VIX1D at the trade time (used for expected move calculation).
        - short_strike, long_strike: Strike prices for the short and long options in the spread.

    Notes:
    ------
    - The function makes multiple API calls to the Polygon API to retrieve historical data for 
      the underlying asset and option chains and uses environment variables to retrieve API keys, 
      so the key should be set as an environment variable (e.g., `POLYGON_API_KEY`).
    """

    backtest_dataset = pd.DataFrame()

    for date in dates:
        try:
            prior_day = trading_dates[np.where(trading_dates == date)[0][0] - 1]
            
            daily_underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2020-01-01/{prior_day}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            daily_underlying_data.index = pd.to_datetime(daily_underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
        
            sl_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            sl_data.index = pd.to_datetime(sl_data.index, unit="ms", utc=True).tz_convert("America/New_York")

            underlying_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            underlying_data.index = pd.to_datetime(underlying_data.index, unit="ms", utc=True).tz_convert("America/New_York")
            
            index_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{index_ticker}/range/1/minute/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={polygon_api_key}").json()["results"]).set_index("t")
            index_data.index = pd.to_datetime(index_data.index, unit="ms", utc=True).tz_convert("America/New_York")
            
            underlying_data = underlying_data[(underlying_data.index.time >= pd.Timestamp(trade_time).time()) & (underlying_data.index.time <= pd.Timestamp("16:00").time())].copy()
            index_data = index_data[index_data.index.time >= pd.Timestamp(trade_time).time()].copy()
            
            index_price = index_data["c"].iloc[0]        
            price = underlying_data["c"].iloc[0]
            closing_value = underlying_data["c"].iloc[-1]
            
            expected_move = (round((index_price / np.sqrt(252)), 2)/100)*move_adjustment
            
            lower_price = round(price - (price * expected_move))
            upper_price = round(price + (price * expected_move))
            
            exp_date = date
            
            regime_df = pd.concat([daily_underlying_data, underlying_data.head(1)], axis=0)
            regime_df["1_mo_avg"] = regime_df["c"].rolling(window=20).mean()
            regime_df["3_mo_avg"] = regime_df["c"].rolling(window=60).mean()
            regime_df['regime'] = regime_df.apply(lambda row: 1 if row['c'] > row['1_mo_avg'] else 0, axis=1)

            direction = regime_df['regime'].iloc[-1]
            minute_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours=pd.Timestamp(trade_time).time().hour, minutes=pd.Timestamp(trade_time).time().minute))
            quote_timestamp = minute_timestamp.value
            close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours=16, minutes=0)).value
            
            if direction == 0:
                valid_calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=call&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
                valid_calls = valid_calls[valid_calls["ticker"].str.contains("SPXW")].copy()
                valid_calls["days_to_exp"] = (pd.to_datetime(valid_calls["expiration_date"]) - pd.to_datetime(date)).dt.days
                valid_calls["distance_from_price"] = abs(valid_calls["strike_price"] - price)
                
                otm_calls = valid_calls[valid_calls["strike_price"] >= upper_price]
                
                short_call = otm_calls.iloc[[0]]
                long_call = otm_calls.iloc[[spread_width]]
                
                short_strike = short_call["strike_price"].iloc[0]
                long_strike = long_call["strike_price"].iloc[0]
                
                short_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{short_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                short_call_quotes.index = pd.to_datetime(short_call_quotes.index, unit="ns", utc=True).tz_convert("America/New_York")
                short_call_quotes["mid_price"] = round((short_call_quotes["bid_price"] + short_call_quotes["ask_price"]) / 2, 2)
                short_call_quotes = short_call_quotes[short_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
                
                short_call_quote = short_call_quotes.median(numeric_only=True).to_frame().copy().T
                short_call_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
                
                long_call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{long_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                long_call_quotes.index = pd.to_datetime(long_call_quotes.index, unit="ns", utc=True).tz_convert("America/New_York")
                long_call_quotes["mid_price"] = round((long_call_quotes["bid_price"] + long_call_quotes["ask_price"]) / 2, 2)
                long_call_quotes = long_call_quotes[long_call_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
                
                long_call_quote = long_call_quotes.median(numeric_only=True).to_frame().copy().T
                long_call_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
                
                spread = pd.concat([short_call_quote.add_prefix("short_call_"), long_call_quote.add_prefix("long_call_")], axis=1).dropna()
                
                spread["spread_value"] = spread["short_call_mid_price"] - spread["long_call_mid_price"]
                cost = spread["spread_value"].iloc[0]

            
            elif direction == 1:
                valid_puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={options_ticker}&contract_type=put&as_of={date}&expiration_date={exp_date}&limit=1000&apiKey={polygon_api_key}").json()["results"])
                valid_puts = valid_puts[valid_puts["ticker"].str.contains("SPXW")].copy()
                valid_puts["days_to_exp"] = (pd.to_datetime(valid_puts["expiration_date"]) - pd.to_datetime(date)).dt.days
                valid_puts["distance_from_price"] = abs(price - valid_puts["strike_price"])
                
                otm_puts = valid_puts[valid_puts["strike_price"] <= lower_price].sort_values("distance_from_price", ascending=True)
                
                short_put = otm_puts.iloc[[0]]
                long_put = otm_puts.iloc[[spread_width]]
                
                short_strike = short_put["strike_price"].iloc[0]
                long_strike = long_put["strike_price"].iloc[0]
        
                short_put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{short_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                short_put_quotes.index = pd.to_datetime(short_put_quotes.index, unit="ns", utc=True).tz_convert("America/New_York")
                short_put_quotes["mid_price"] = round((short_put_quotes["bid_price"] + short_put_quotes["ask_price"]) / 2, 2)
                short_put_quotes = short_put_quotes[short_put_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
                
                short_put_quote = short_put_quotes.median(numeric_only=True).to_frame().copy().T
                short_put_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")
                
                long_put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{long_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=asc&limit=5000&sort=timestamp&apiKey={polygon_api_key}").json()["results"]).set_index("sip_timestamp")
                long_put_quotes.index = pd.to_datetime(long_put_quotes.index, unit="ns", utc=True).tz_convert("America/New_York")
                long_put_quotes["mid_price"] = round((long_put_quotes["bid_price"] + long_put_quotes["ask_price"]) / 2, 2)
                long_put_quotes = long_put_quotes[long_put_quotes.index.strftime("%Y-%m-%d %H:%M") <= minute_timestamp.strftime("%Y-%m-%d %H:%M")].copy()
                
                long_put_quote = long_put_quotes.median(numeric_only=True).to_frame().copy().T
                long_put_quote["t"] = minute_timestamp.strftime("%Y-%m-%d %H:%M")

                spread = pd.concat([short_put_quote.add_prefix("short_put_"), long_put_quote.add_prefix("long_put_")], axis=1).dropna()
                
                spread["spread_value"] = spread["short_put_mid_price"] - spread["long_put_mid_price"]
                cost = spread["spread_value"].iloc[0]
            
            trade_data = pd.DataFrame([{"date": date, "cost": cost, "ticker": ticker, "direction": direction, "side": 'call' if direction==0 else 'put',
                                            "underlying_closing_price": closing_value,
                                        'expected_move': expected_move, 'underlying_price_at_trade': price, 'underlying_high': sl_data['h'].iloc[0], 'underlying_low': sl_data['l'].iloc[0], 
                                        'lower_price': lower_price, 'upper_price': upper_price,
                                        'vix1d_value': index_price}])
            
            if direction == 1:
                trade_data = pd.concat([trade_data, short_put_quote.add_prefix("short_"), long_put_quote.add_prefix("long_")], axis=1)
                trade_data['short_strike'] = short_strike
                trade_data['long_strike'] = long_strike

            elif direction == 0:
                trade_data = pd.concat([trade_data, short_call_quote.add_prefix("short_"), long_call_quote.add_prefix("long_")], axis=1)
                trade_data['short_strike'] = short_strike
                trade_data['long_strike'] = long_strike

            backtest_dataset = pd.concat([backtest_dataset, trade_data], axis=0)
            
        except Exception as data_error: 
            print(data_error)
            continue

    backtest_dataset = backtest_dataset.set_index('date')
    backtest_dataset.index = pd.to_datetime(backtest_dataset.index)

    return backtest_dataset

backtest_data = build_spread_backtest_dataset(dates=trading_dates[1:], ticker='I:SPX', index_ticker="I:VIX1D", 
                                              options_ticker="SPX", trade_time="09:35", move_adjustment=0.5, spread_width=1)
