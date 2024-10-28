# Short-Vol-ML

# Base Strategy:

Our primary strategy/model is relatively simple and aims to capture the volatility risk premium on the S&P 500 every day. We express this through selling an out-of-the-money, 0DTE credit spread. This means that as long as price doesn’t move enough/in the right direction to put these options in the money by the end of the trading day, we get to keep the premium we collected from selling the spread. The methodology for this strategy is as follows:

Every day, at 9:35 a.m. EST (this is our trade time instead of 9:30 market open to give price a moment to react to its overnight move / settle down), we:

1. Determine whether we’ll be selling a call or put spread. We’ll keep this part quite simple, using whether or not price is above its 20-day moving average as a rough proxy. If it’s above, we assume price is in a positive trend/regime and sell a put spread, expecting it to continue moving to the upside and vice versa. 
2. Determine how far out of the money to sell. We want to strike a balance of selling close enough to the money that we receive a sufficiently large premium for the risk we assume, but far enough from the money such that price is unlikely to be volatile enough to reach these levels. Given that we’re selling 0DTE options, the 1-day volatility index (VIX1D) is quite a good estimator of daily volatility. The value of the VIX1D represents a whole number percentage annualized implied move for the S&P 500, calculated using the implied volatilites of 0 and 1 DTE options on the index. Given that we're interested in the daily expected move, we deannualize this figure to obtain this implied move:

Implied move = v / √252, where:

   > v = price (value) of the VIX1D

 There are many ways to approximate realized volatility. For our purposes, we will calculate realized volatility as a whole number percentage (same unit as VIX1D) for our period of interest (trade time to market close) as:
   
 Realized vol = (| p<sub>t</sub> - p<sub>c</sub> | / p<sub>t</sub>) * 100, where:
   
  > p<sub>t</sub> = price at the time of the trade
 
  > p<sub>c</sub> = price at market close

We define the volatility risk premium for a given day as:

VRP = Implied Move - Realized Vol

If this figure is positive, it means that the VIX1D overestimated volatility and vice versa. Let's take a look at some plots over our backtest period (04/24/2023 - 10/25/2024).

![Histogram and Scatter Plot](/Users/teymour/Desktop/qnt-projs/short-vol-github/vrp-plots.png)

Many refer to this type of strategy as  “picking pennies up in front of a steamroller” because it has quite a high win rate but outsized losses. In our case, we see that since the beginning of our testing period, we’ve won __% of the time. However, a corollary of such a high win rate is that the risk profile of this strategy is roughly 1:4, meaning that a single loss negates 4 wins. 

# Meta-Labeling

Meta-labeling is a technique introduced by Dr. Marcos Lopez de Prado, who explains it far better than I ever could:

“Suppose that you have a model for setting the side of the bet (long or short). You just need to learn the size of that bet, which includes the possibility of no bet at all (zero size). This is a situation that practitioners face regularly. We often know whether we want to buy or sell a product, and the only remaining question is how much money we should risk in such a bet. We do not want the ML algorithm to learn the side, just to tell us what is the appropriate size. At this point, it probably does not surprise you to hear that no book or paper has so far discussed this common problem. Thankfully, that misery ends here. I call this problem meta-labeling because we want to build a secondary ML model that learns how to use a primary exogenous model.”

This asymmetric risk profile makes the strategy an excellent candidate for meta-labeling because avoiding even a few of these outsized losses vastly improves performance. In this case, our meta-model will be binarily discerning whether to trade or not on a given day. Should we trade, we’ll use a fixed size of 1 of each contract for simplicity’s sake.

Future steps include building a more robust ML pipeline (hyperparameter tuning, etc…), dynamically adjusting position sizing, and finding a more effective way to determine direction (but if doing so were easy, then we’d all be rich ;))
