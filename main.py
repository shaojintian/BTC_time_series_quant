import alphalens
import pandas as pd

factors = pd.read_csv('path_to_factors.csv',parse_dates=True,index_col=['date','asset'])
prices = pd.read_csv('btc_prices_15min.csv',parse_dates=True,index_col=['date'])


# Ingest and format data
factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factors, 
                                                                   prices, 
                                                                   quantiles=5,
                                                                   periods=(1,5,20))

# Run analysis
alphalens.tears.create_full_tear_sheet(factor_data)

