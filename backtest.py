
# Importing Libraries
import os
from sklearn.linear_model import LinearRegression as lm 
import pandas as pd 
import numpy as np 
from itertools import combinations
import statsmodels.api as sm
import datetime as date
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.stats import t,ttest_ind,norm,iqr
from sklearn.metrics import r2_score
import pyfolio as pf
from numpy.linalg import inv

# Ignoring Warnings produced
import warnings
warnings.filterwarnings('ignore')

# Importing functions
from stats_func import (
    regression,
    dynamic_regression,
    test_stationarity,
    test_significance,
    cointegration_test,
    robustness
)
from optimize_func import (
    optimize,
    optimize_delta
)


def build_strategy(residuals,kalman=False,delta=1e-4,display=True):
    
    # Defining the dataset to be fit to the analytical OU process equation
    tau=float(1)/252
    spread=pd.DataFrame(residuals)
    spread.columns=['Spread']
    spread['Spreadt-1']=spread['Spread'].shift(1)
    spread.dropna(inplace=True)
    target_y=pd.DataFrame(spread['Spread'])
    target_y.columns=['y']
    spread.drop(['Spread'],axis=1,inplace=True)

    if kalman == False:
        # Calculating OU parameters from linear regression 
        autoregression_coefficient,mean_reverting_term,resids=regression(spread['Spreadt-1'],target_y['y'])
        # Creating an array of OU parameters for each trading session. The parameters in this case are constant for each time session
        mean_reverting_term=np.repeat(mean_reverting_term,len(resids))
        autoregression_coefficient=np.repeat(autoregression_coefficient,len(resids))
    
    else:
        # Calculating the OU parameters using Kalman Filters
        autoregression_coefficient,mean_reverting_term,resids=dynamic_regression(spread['Spreadt-1'],target_y['y'],delta)
               
    # Computing Half life of the process
    speed_of_reversion=-1*np.log(np.absolute(autoregression_coefficient))/tau

    # Computing the mean about which the OU process reverts
    mean=mean_reverting_term/(1-autoregression_coefficient)
    if np.isnan(mean).any():
        mean=np.nan_to_num(mean)
        
    # Computing the instantaneous and the equivalent diffusion for the spread
    diffusion_ou=((2*speed_of_reversion*np.var(residuals))/(1-np.exp(-2*speed_of_reversion*tau)))**0.5
    speed_of_reversion[speed_of_reversion<=0]=1e-15
    diffusion_eq=diffusion_ou/((2*speed_of_reversion)**0.5)
    half_life=np.log(2)/speed_of_reversion
    

    if display == True:
        print ("\nThe spread fitted to the Ornstein Uhlenbeck Process has the following parameters: ")        
        if kalman == False:
            print ("The mean of reversion for this spread is {} \nSigma of reversion for this spread is {} \nThe speed of reversion, short term diffusion and half life of the OU process is {}, {} and {}".format(mean[0],diffusion_eq[0],speed_of_reversion[0],diffusion_ou[0],half_life[0]))
        else:
            
            plt.figure(figsize=(12, 16))
            iqr_mr=iqr(mean)*4
            iqr_sr=iqr(diffusion_eq)*4
            iqr_spr=iqr(speed_of_reversion)*4
            iqr_ds=iqr(diffusion_ou)*4
            lim=[iqr_mr,iqr_sr,iqr_spr,iqr_ds]
            xlabel='Trading Sessions'
            ylabel=['OU Mean','OU Diffusion','Rate of Revesion','Diffusion over short time']
            title=['Mean of Reversion','Sigma of Reversion','Speed of Reversion','Diffusion over short timescale']
            subplots=[411,412,413,414]
            labels=['Mean of Reversion','Sigma of Reversion','Speed of Reversion','Diffusion over short timescale']
            plots=[mean,diffusion_eq,speed_of_reversion,diffusion_ou]
            
            plt.subplots_adjust(hspace=0.5)  # 调整子图间距

            for i in range(0,4):
                plt.subplot(subplots[i])
                plt.title(title[i])
                plt.xlabel(xlabel)
                plt.ylabel(ylabel[i])
                plt.ylim(np.percentile(plots[i],25)-lim[i],np.percentile(plots[i],75)+lim[i])
                plt.plot(plots[i],label=labels[i])
                plt.legend()
        plt.show()
        
    return mean,diffusion_eq 

def trade(data,spread,mean,diffusion_eq,weight,entry_point,slippage=0.05,rfr=0.02,max_trade_exit=np.float('Inf'),stoploss=0.5,plot=True):
    #,diff_spread
    # Initialising trading Flags 
    ydata=data['ydata'].values
    xdata=data['xdata'].values
    diff_spread=data['ydata']-weight*data['xdata']
    diff_spreadd=diff_spread.values
    
    top_trade_executing=0 #Corresponds to a trade entering from above the mean
    bottom_trade_executing=0 #Corresponds to a trade entering from below the mean
    entry_price=0
    trade_executing=0
    
    buy=[]
    sell=[]
    status=[]
    pl=np.zeros(len(spread),dtype=np.float)
    returns=np.zeros(len(spread),dtype=np.float)
    portfolio_value=np.zeros(len(spread),dtype=np.float)
    flag1=0
    flag2=0
    
    if slippage == -999:
        flag1=1
    
    if flag1 != 1:

        slipped_entry_top = np.zeros(len(spread), dtype=float)
        slipped_entry_bottom = np.zeros(len(spread), dtype=float)
           
    # Trying to plot stoploss
    stop = np.zeros(len(spread), dtype=float)
    if stoploss == -999:
        flag2=1
        stoploss=np.float('Inf')
        stop.fill(np.nan)
    
    if max_trade_exit == -999:
        max_trade_exit=np.float('Inf')
    k=0

    for i in range(0,len(spread)):
                
            # Mean is the exit point for an executing trade 
            exit=mean[i]
            top_entry= mean[i] + diffusion_eq[i]*entry_point    # 拟合OU过程得到的diffusion
            bottom_entry= mean[i] - diffusion_eq[i]*entry_point
            
            # Add slippage
            if flag1 != 1:
                    
                if top_entry > 0:
                    slipped_entry_top[i]=float(1-slippage)*top_entry

                else:
                    slipped_entry_top[i]=float(1+slippage)*(top_entry)
                
                if bottom_entry < 0:
                    slipped_entry_bottom[i]=float(1-slippage)*bottom_entry

                else:
                    slipped_entry_bottom[i]=float(1+slippage)*bottom_entry
                
            
            # If no trade in current execution
            if trade_executing == 0:

                ''' Dependent on whether the current price is above or below mean, we calculate the 
                    price at which we can enter a trade and the associated slippage based on the mean and 
                    equivalent diffusion for the given trading session'''
                
                if spread[i] > mean[i]:
                    entry=top_entry
                    if flag1 != 1:
                        slipped_entry=slipped_entry_top[i]
                else:
                    entry=bottom_entry
                    if flag1 != 1:
                        slipped_entry=slipped_entry_bottom[i]
                
                
                ''' We check the session price against the computer range of entry and update 
                    the required flags. The stoploss price for that trade is calculated and the
                    portfolio value is updated'''
                
                if spread[i] > mean[i]:
                    if flag1 == 1:
                        if spread[i]==entry or (i!=0 and spread[i-1]>entry and spread[i]<entry):
                             trade_executing=1   
                    
                    else:
                        # [(1-slippage)*top_entry, top_entry], 滑点范围带中开仓
                        if spread[i] <= entry and  spread[i] >= slipped_entry:
                            trade_executing=1
                            
                    if trade_executing ==1:
                        buy.append(i)
                        top_trade_executing=1
                        status.append(1) 
                        portfolio_value[i]=ydata[i]+weight[i]*xdata[i]
                        entry_price=spread[i]
                                                
                        if entry < 0:
                            stoploss_exit=(1-stoploss)*(entry)
                        else:
                            stoploss_exit=(1+stoploss)*entry
                        
                        stop[i]=stoploss_exit
                            
            
                if spread[i] < mean[i]:
                    if flag1 == 1:
                        if spread[i] == entry or (i!=0 and spread[i-1]<entry and spread[i]>entry):
                             trade_executing=1
                    else:
                        if spread[i] >= entry and spread[i] <= slipped_entry:
                            trade_executing=1
                            
                    if trade_executing ==1:    
                        buy.append(i)
                        bottom_trade_executing=1
                        status.append(-1)
                        portfolio_value[i]=ydata[i]+weight[i]*xdata[i]
                        entry_price=spread[i]
                        
                        if entry > 0:
                            stoploss_exit=(1-stoploss)*(entry)
                        else:
                            stoploss_exit=(1+stoploss)*entry
                        stop[i]=stoploss_exit
            else:
                ''' If no trade is being executed, the portfolio value is first updated, and then the current 
                    price of asset is checked against the previously computed exit range. If the price is within the 
                    exit range, the position is liquidated. If not it is then checked whether the current price
                    breaches the defined stoploss or the trade duration execeeds the maximum allowable time in a trade.
                    In case of a breach the position is again closed'''
                
                stop[i]=stoploss_exit
                
                if top_trade_executing == 1:
                    
                    pl[i]=weight[i]*(xdata[i]-xdata[i-1])+(ydata[i-1]-ydata[i])
                    portfolio_value[i]=portfolio_value[i-1]+pl[i]
                    
                    
                    if spread[i-1]>=exit and spread[i]<=exit:
                        top_trade_executing=0
                                        
                    if top_trade_executing == 0:
                        trade_executing=0
                        sell.append(i)
                        k=k+1
                
                    elif spread[i] > stoploss_exit:
                        trade_executing=0
                        top_trade_executing=0
                        sell.append(i)
                        status[k]=status[k]*3
                        k=k+1
                
                
                if bottom_trade_executing==1:
                    pl[i]=weight[i]*(xdata[i-1]-xdata[i])+(ydata[i]-ydata[i-1])
                    portfolio_value[i]=portfolio_value[i-1]+pl[i]
                    if spread[i-1]<=exit and spread[i]>=exit:
                        bottom_trade_executing=0
                    
                            
                    if bottom_trade_executing == 0:
                        trade_executing=0
                        sell.append(i)
                        k=k+1   
                
                    elif spread[i] < stoploss_exit:
                        trade_executing=0
                        bottom_trade_executing=0
                        sell.append(i)
                        status[k]=status[k]*3
                        k=k+1
                
                
                if trade_executing ==1 and i-buy[k] == max_trade_exit:
                    trade_executing=0
                    bottom_trade_executing=0
                    top_trade_executing=0
                    sell.append(i)
                    status[k]=status[k]*2
                    k=k+1
        
            ''' Based on the trading activity in the current and previous sesison the portfolio 
            value is computed and the returns are calculated'''
            
            if i!=0 and portfolio_value[i-1] != 0 and i-1 not in sell:
                returns[i]=(portfolio_value[i]-portfolio_value[i-1])/(portfolio_value[i-1])
            
            
            ''' If no trading activity takes place, it is assumed that the return is the risk free 
            rate i.e. if capital is not invested in the trade it is kept in a bank account'''
            
            if returns[i] == 0:
                returns[i]+=(rfr+1)**(float(1)/252)-1       
            
    if plot == True:
        plt.figure(1, figsize=(16, 12))

        s_iqr=0.75*iqr(spread)
        plt.ylim(min(spread)-s_iqr,max(spread)+s_iqr)
        plt.plot(mean,label="Mean of Reversion",linestyle='--',linewidth=3)
        plt.plot(mean + diffusion_eq*entry_point,label="Entry Bounds",linestyle='--',linewidth=3)
        plt.plot(mean - diffusion_eq*entry_point,label="Entry Bounds",linestyle='--',linewidth=3)
        plt.plot(spread,label="Reverting Spread",linewidth=2)
        if flag1 != 1:
            plt.plot(slipped_entry_bottom,label='Slippage Range for Trade from below the mean',linestyle='--',linewidth=3)
            plt.plot(slipped_entry_top,label='Slippage Range for Trade from above the mean',linestyle='--',linewidth=3)
        if flag2 != 1:
            plt.plot(stop,label="Stoploss",linestyle=':',linewidth=3)
    
        b_array=np.zeros(len(spread), dtype=float)
        b_array.fill(np.nan)
        b_array[buy]=spread[buy]
    
        s_array=np.zeros(len(spread), dtype=float)
        s_array.fill(np.nan)
        s_array[sell]=spread[sell]

        plt.plot(b_array,marker='o',label="Buy Signals",markersize=15)
        plt.plot(s_array,marker='o',label="Sell Signals",markersize=15)
        
        plt.xlabel('Trading Sessions')
        plt.ylabel('Spread (Rupees)')
        plt.title("Simulation of trades on the spread")
        plt.legend(loc='best')
        
        plt.show()

    return buy,sell,status,portfolio_value,returns

def commission(price_s1,price_s2,commission_short,commission_long):
    comm_s1=abs(price_s1*commission_long if price_s1 > 0 else price_s1*commission_short)
    comm_s2=abs(price_s2*commission_long if price_s2 > 0 else price_s2*commission_short)
    return comm_s1+comm_s2

def trade_sheet(buy,sell,status,data,coint,commission_short=0,commission_long=0):
    #To equal the dates and spread

    dates=data['Date'].values
    s1=data['ydata'].values
    s2=data['xdata'].values
    trade_ticket=pd.DataFrame()
    net_profit=np.zeros(len(buy),dtype=np.float)
    gross_profit=np.zeros(len(buy),dtype=np.float)
    duration=np.zeros(len(buy),dtype=np.float)
    buy_price=np.zeros(len(buy),dtype=np.float)
    sell_price=np.zeros(len(buy),dtype=np.float)
    margin=np.zeros(len(buy),dtype=np.float)
    flag=0
    
    buy_date=[]
    sell_date=[]
    stat=[]
    
    if len(buy) == len(sell):
        tot=len(buy)
    else:
        tot=len(buy)-1
        flag=1

  
    for i in range(0,tot):
         
        if status[i] > 0:
            buy_weights=np.negative(coint[buy[i]])
            sell_weights=np.negative(coint[sell[i]])
        else:    
            buy_weights=coint[buy[i]]
            sell_weights=coint[sell[i]]
        
        # Computing the buy and selling price of inidvidual asset class
        sell_s1=s1[sell[i]]*sell_weights[0]
        sell_s2=-s2[sell[i]]*sell_weights[1]
        buy_s2=-s2[buy[i]]*buy_weights[1]
        buy_s1=s1[buy[i]]*buy_weights[0]
        
        sell_price[i]=sell_s1+sell_s2
        buy_price[i]=buy_s1+buy_s2
        sell_cost=commission(sell_s1,sell_s2,commission_short,commission_long)
        buy_cost=commission(buy_s1,buy_s2,commission_short,commission_long)
        trade_cost=sell_cost+buy_cost
        
        # Computing the profit and margin, duration  for each trade  
        margin[i]=net_profit[i]/(abs(buy_s1)+abs(buy_s2))
        gross_profit[i]=sell_price[i]-buy_price[i]
        net_profit[i]=gross_profit[i]-trade_cost
        duration[i]=sell[i]-buy[i]+1
        
        buy_date.append(dates[buy[i]])
        sell_date.append(dates[sell[i]])
        if abs(status[i]) == 1:
            stat.append("Completed")
        if abs(status[i]) == 2:
            stat.append("Maximum Time Elapsed")
        if abs(status[i]) == 3:
            stat.append("Stoploss Breached")
        
    
    if flag== 1:       
        buy_date.append(dates[buy[tot]])
        duration[tot]=len(s1)-buy[tot]
        stat.append("Ongoing")
        sell_date.append("NA")

    df=pd.DataFrame({'Buy Date':buy_date,'Buy Price':buy_price,'Sell Date':sell_date,'Sell Price':sell_price,'Duration':duration,'Net Profit':net_profit,'Gross Profit':gross_profit,'Status':stat})
        
    return df

def backtest(df,returns,dates,rfr,display=True,window=125):
    
    #Trade Statistics
    flag=0
    try:
    
            # Preparing Dataframe of returns for computing Sharpe Ratio
            df_ret=pd.DataFrame(returns)
            df_ret.columns=['Returns']
            df_ret['Date']=dates[-df_ret.shape[0]:].values
            df_returns=df_ret.drop(df_ret[df_ret['Returns'] == 0].index,axis=0)
            dates_of_trading = pd.DataFrame(df_returns['Date'],columns=['Date'])
            df_rets=pd.DataFrame(df_returns['Date'],columns=['Date'])
            df_rets['Returns']=df_returns['Returns']
            df_returns['Returns+1']=df_returns['Returns']+1
            df_returns['Rf_Returns']=df_returns['Returns']-((rfr+1)**(float(1)/252)-1)
        

            #Finding Fama French 4 Factors
            # ff=pd.read_csv('FourFactors.csv',parse_dates=[0],usecols=['Date','HML %','SMB %','WML %','Rm-Rf %'])
            # ff.dropna(inplace=True)
            # ff_comb=dates_of_trading.merge(ff,on='Date')
            # ff=dates_of_trading.merge(ff,on='Date')
            # ff.set_index(ff['Date'],inplace=True)
            # ff.drop('Date',axis=1,inplace=True)
            # ff_comb.drop('Date',axis=1,inplace=True)
            # ff_comb=ff_comb/100
            # ff=ff/100  
            # beta_ff,alpha_ff,na=regression(ff_comb,df_returns['Rf_Returns'])
            
            # Finding market alpha and beta
            # beta_m,alpha_m,na=regression(ff_comb['Rm-Rf %'],df_returns['Rf_Returns'])
            
            total_trades=df.shape[0]
            complete_trades = df[df['Status']=='Completed'].shape[0]
            incomplete_trades=total_trades-complete_trades  
            average_duration=df['Duration'].mean()
            average_profit=df['Net Profit'].mean()
            total_profit=df['Net Profit'].sum()
            profit_trades=df[df['Net Profit']>0].shape[0]
            average_profit_profittrades=df[df['Net Profit']>0]['Net Profit'].mean()
            risk=np.std(returns)
            ret=np.sort(returns)

            # Calculating the Value at Risk and Expected Shortfall of the Strategy
            dec=0.05*len(returns)%1
            i1=int(np.floor(0.05*len(returns))-1)
            i2=int(np.ceil(0.05*len(returns))-1)
            var=ret[i1]+(ret[i2]-ret[i1])*(dec)
            es=np.append(ret[0:i1],(ret[i2]-ret[i1])*(dec)).mean()
            loss_trades=total_trades-profit_trades
            
            # Calculating the Sharpe and Sortino Ratio of the Strategy
            cumulative_annualised_return=df_returns['Returns+1'].prod()**(float(252)/len(df_returns))-1
            portfolio_sd=df_returns['Returns'].std()*(252**0.5)    
            negative_return_sd=df_returns[df_returns['Returns']<0]['Returns'].std()*(252**0.5)
            sharpe=(cumulative_annualised_return-rfr)/(portfolio_sd)
            sortino=(cumulative_annualised_return-rfr)/(negative_return_sd)
            
            
            if total_trades >0:
                strat_summary={'Total Trades':total_trades,'Complete Trades':complete_trades,'Incomplete Trades':incomplete_trades,
                   'Profit Trades':profit_trades,'Loss Trades':loss_trades,'Total Profit':total_profit,'Average Profit on Profitable Trades':average_profit_profittrades,'Average Trade Duration':average_duration,
                   'Average Profit':average_profit,'Win Ratio':float(profit_trades)/loss_trades,'Standard Deviation of Returns':risk,'Value at Risk':var, 'Expected Shortfall':es,'Sharpe Ratio':sharpe,
                    'Sortino Ratio':sortino,'Cumulative Return':cumulative_annualised_return,
                    # 'Market Alpha':alpha_m,'Market Beta':beta_m,'HML Beta':beta_ff[0], 'SMB Beta' :beta_ff[1], 
                    # 'WML Beta':beta_ff[2], 'Momentum Beta':beta_ff[3],'Fama French Four Factor Alpha':alpha_ff
                    }

                if display == True:
                    print ("\nSummary of all trades made:")
                    for i in strat_summary:
                        print (i,':',strat_summary[i])
                
                    print ("(All Profits calculations are made using the Net Profits, where the commisions have been deducted from the Profits)")
            else:
                flag=1
                strat_summary={'Total Trades':0,'Complete Trades':0,'Incomplete Trades':0,
                   'Profit Trades':0,'Loss Trades':0,'Total Profit':0,'Average Trade Duration':0,
                   'Average Profit':0,'Standard Deviation of Returns':0,'Value at Risk':0, 'Expected Shortfall':0, 'Sharpe Ratio':0,
                    'Sortino Ratio':0,'Cumulative Return':0,'Market Alpha':0,'Market Beta':0,'HML Beta':0,
                    'SMB Beta':0, 'WML Beta':0, 'Momentum Beta':0,'Fama French Four Factor Alpha':0}
                print ("No trades were executed")
                
    except:
            flag=1
            strat_summary={'Total Trades':0,'Complete Trades':0,'Incomplete Trades':0,
                   'Profit Trades':0,'Loss Trades':0,'Total Profit':0,'Average Trade Duration':0,
                   'Average Profit':0,'Standard Deviation of Returns':0,'Value at Risk':0, 'Expected Shortfall':0, 'Sharpe Ratio':0,
                    'Sortino Ratio':0,'Cumulative Return':0,'Market Alpha':0,'Market Beta':0,'HML Beta':0,
                    'SMB Beta':0, 'WML Beta':0, 'Momentum Beta':0,'Fama French Four Factor Alpha':0}
            if display == True:
                print ("No trades were executed")
        
    
        #Time Elapsed Trades
    try: 
        te=df[df['Status']== "Maximum Time Elapsed"]
        te_trades=te.shape[0]
        te_exposure=te['Net Profit'].mean()
        te_summary={'Trades':te_trades,'Net Exposure':te_exposure}
        if display == True:
        
            if te_trades == 0:
                te_summary={'Trades':0,'Net Exposure':0}
                print("\nNo trades that exceeded the maximum duration")
            else:
                print ("\nSummary of all Time Limit Exceeded trades:")
                for i in te_summary:
                        print (i,':',te_summary[i])
    except:
        te_summary={'Trades':0,'Net Exposure':0}
        if display == True:
            print("\nNo trades that exceeded the maximum duration")
    
    #Stoploss Breached Trades 
    try: 
        slb=df[df['Status']=='Stoploss Breached']
        slb_trades=slb.shape[0]
        slb_average_duration=slb['Duration'].mean()
        slb_average_loss=-1*slb['Net Profit'].mean()
        slb_summary={'Trades':slb_trades,'Average Duration':slb_average_duration,
                   'Average Loss':slb_average_loss}
        if display == True:
            if slb_trades == 0:
                print("\nNo trades that breached the stoploss")
                slb_summary={'Trades':0,'Average Duration':0,
                   'Average Loss':0}
            else:
                print ("\nSummary of all Stoploss Breached trades:")
                for i in slb_summary:
                    print (i,':',slb_summary[i])
    except:
        if display == True:
            print("\nNo trades that breached the stoploss")
        slb_summary={'Trades':0,'Average Duration':0,
                   'Average Loss':0}
    
    if display == True:
            
        if flag == 0:
            plt.figure(figsize=(15,10))
            plt.subplot(131)
            
            plt.boxplot(df['Duration'])
            plt.xlabel('Duration')
            plt.ylabel('Number of Trading sessions')
            plt.title('Duration of all trades')
    
            
            plt.subplot(132)
            plt.boxplot(df['Net Profit'])
            plt.xlabel('Profit')
            plt.ylabel('Profit in $''s')
            plt.title('Profit of all trades')
    
            plt.subplot(133)
            plt.boxplot(returns)
            plt.xlabel('Returns')
            plt.ylabel('Returns in %')
            plt.title('Returns of all trades')
            plt.show()
            
            #Rolling Sharpe Ratio
            print( "\n The rolling Sharpe Ratio is: ")
            rolling_returns=((df_returns['Returns+1'].rolling(window).apply(lambda x:x.prod()))**(float(252)/window))-1
            rolling_sharpe = (rolling_returns-rfr)/(df_returns['Returns'].rolling(window).std()*(252**0.5))
            rolling_sharpe.dropna(inplace=True)
            plt.figure(figsize=(10,10))
            plt.plot(rolling_sharpe,label="Rolling Sharpe")
            plt.xlabel('Trading Sessions')
            plt.ylabel('Sharpe Ratio')
            plt.title('Rolling Annual Sharpe Ratio')
            plt.legend()
            plt.show()    
          
            # print ("\n The rolling Alpha, Beta factors on the market are: ")
            # rolling_beta=[]
            # rolling_alpha=[]
            
            # for i in range(0,df_returns.shape[0]-window):
            #     am,bm,na=regression(ff_comb[i:i+window]['Rm-Rf %'],df_returns[i:i+window]['Rf_Returns'])
            #     rolling_beta.append(am)
            #     rolling_alpha.append(bm)

            # plt.figure(figsize=(24,10))
            # plt.subplot(121)
            # plt.plot(rolling_alpha,label='Rolling Alpha')
            # plt.xlabel('Trading Sessions')
            # plt.ylabel('Alpha')
            # plt.title('6M Rolling Market Alpha')
            # plt.legend()
            # plt.subplot(122)
            # plt.plot(rolling_beta,label='Rolling Beta')
            # plt.xlabel('Trading Sessions')
            # plt.ylabel('Beta')
            # plt.title('6M Rolling Market Beta')
            # plt.legend()
            # plt.show()
            
            # # Rolling Fama-French Factors      
            # rolling_alpha_ff=[]
            # rolling_beta_hml=[]
            # rolling_beta_smb=[]
            # rolling_beta_wml=[]
            # rolling_beta_mom=[]
    
            # for i in range(0,df_returns.shape[0]-window):
            #     a,b,resid1=regression(ff_comb[i:i+window],df_returns[i:i+window]['Rf_Returns'])
            #     rolling_beta_hml.append(a[0])
            #     rolling_beta_smb.append(a[1])
            #     rolling_beta_wml.append(a[2])
            #     rolling_beta_mom.append(a[3])
            #     rolling_alpha_ff.append(b)
            # plt.figure(figsize=(24,10))
            # plt.subplot(121)
            # plt.plot(rolling_beta_hml,label='Rolling Beta HML')
            # plt.plot(rolling_beta_smb,label='Rolling Beta SMB')
            # plt.plot(rolling_beta_wml,label='Rolling Beta WML')
            # plt.plot(rolling_beta_mom,label='Rolling Beta MoM')
            # plt.xlabel('Trading Sessions')
            # plt.ylabel('Beta')
            # plt.title('6M Rolling Beta of the 4 Factor model')
            # plt.legend()
            # plt.subplot(122)
            # plt.plot(rolling_alpha_ff,label='Rolling Alpha')
            # plt.xlabel('Trading Sessions')
            # plt.ylabel('Alpha')
            # plt.title('6M Rolling Alpha from the 4 Factor model')
            # plt.legend()
            # plt.show()

            # # Producing tear sheet from returns 
            # df_rets['Date']=pd.to_datetime(df_rets['Date'])
            # df_rets.set_index(df_rets['Date'],inplace=True)
     
            # pf.create_full_tear_sheet(df_rets['Returns'],benchmark_rets=ff['Rm-Rf %'],factor_returns=ff)


        
    return strat_summary,te_summary,slb_summary

# start=train_start, end=train_end
def trading_algorithm(
    name1, name2,
    start='2016-05-30', end='2017-05-30',
    adf_ci=0.95, sig_test_ci=0.95,
    robust_start='2014-05-30', robust_end='2016-05-30',
    rob_ci=0.95,
    if_train=True, if_set_trading_params=True,
    if_backtest=True
):
    print("\nTwo asset classes for cointegration: {}, {}\n".format(name1, name2))

    asset1=pd.read_csv('data/{}.csv'.format(name1),usecols=['Date','Adj Close'],parse_dates=[0])
    asset2=pd.read_csv('data/{}.csv'.format(name2),usecols=['Date','Adj Close'],parse_dates=[0])
    pair=pd.merge(asset1,asset2,how='inner',on=['Date'])
    pair.columns=['Date','ydata','xdata']
    pair.dropna(inplace=True)

    print('\nDate to run cointegration test from {} to {}\n'.format(start, end))
    start=pd.to_datetime(start)
    end=pd.to_datetime(end)
    pairs_training=pair[pair['Date'] > start]
    pairs_training=pairs_training[pairs_training['Date'] < end]

    print('\n The confidence interval for the ADF test: {}\n'.format(adf_ci))
    print('\n The confidence interval for the cointegration significance test: {}\n'.format(sig_test_ci))
    adf_ci=float(adf_ci)
    sig_test_ci=float(sig_test_ci)
    
    # 协整性测试
    flag3=cointegration_test(
        xdata=pairs_training['xdata'],
        ydata=pairs_training['ydata'],
        stat_value_ci=adf_ci,
        sig_value_ci=sig_test_ci,
        s1=name1,
        s2=name2,
        print_summary=False
        )
    if flag3 < 0:
        return False

    # 互换y,x
    if flag3 == 2:
        pairs_training.columns=['Date','xdata','ydata']
        pair.columns=['Date','xdata','ydata']
        temp=name1
        name1=name2
        name2=temp

    if  flag3==1 or flag3==2: 

        # 协整的稳健性测试
        print ("\nA larger date range to asses the robustness of the cointegration from {} to {}\n".format(robust_start, robust_end))
        date1=pd.to_datetime(robust_start)
        date2=pd.to_datetime(robust_end)

        print('\n The confidence interval for the robustness test: {}\n'.format(rob_ci))
        rob_ci=float(rob_ci)
        robust_set=pair[pair['Date'] > date1]
        robust_set=robust_set[robust_set['Date'] < date2]
        robustness(
            xdata=pairs_training['xdata'],
            ydata=pairs_training['ydata'],
            long_xdata=robust_set['xdata'],
            long_ydata=robust_set['ydata'],
            ci=rob_ci
            )
        
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        # Continue with strategy fitting on training data
        if if_train:
            # Calculate residuals using Kalman Filters-1 or Linear Regression-0
            flag1=int(0)
            # Calculate Ornstein Uhlenbeck parameters using Kalman Filters-1 or Linear Regression-0
            flag2=int(0)
            
            if flag1 == 1:
                # Enter Delta value for Kalman Filter for computing Cointegration Weight. Default is 0.0001
                delta_r=float(0.0001)
                coef_tr,intercept_tr,spread_tr=dynamic_regression(pairs_training['xdata'],pairs_training['ydata'],delta_r)
            else:
                coef_tr,intercept_tr,spread_tr=regression(pairs_training['xdata'],pairs_training['ydata'])
            
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print("\nThe cointegration weights and the intercept for the trading strategy are as follows: \n")
            if flag1 == 0:
                print ("{}:{}".format(name1,1))
                print ("{}:{}".format(name2,coef_tr))
                print ("Intercept: {}".format(intercept_tr))
            else:
                print ("{}:{}".format(name1,1))
                plt.figure(figsize=(24,10))
                plt.subplot(121)
                plt.plot(coef_tr,label='Cointegration Weight {}'.format(name2))
                plt.title('Estimate of Cointegration Weight of {}'.format(name2))
                plt.ylabel('Weight')
                plt.xlabel('Trading Sessions')
                plt.legend()
                plt.subplot(122)
                plt.plot(intercept_tr,label='Regression Intercept')
                plt.ylabel('Intercept')
                plt.title('Estimate of Cointegration Intercept')
                plt.xlabel('Trading Sessions')
                plt.legend()
                plt.show()
                
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            if flag2 == 1:
                # Enter Delta value for Kalman Filter for fitting to OU process. Default is 0.0001
                delta_ou=float(0.0001)
                mean_tr,diffeq_tr=build_strategy(
                    residuals=spread_tr,
                    kalman=True,
                    delta=delta_ou
                    )
            else:
                mean_tr,diffeq_tr=build_strategy(spread_tr)
            
            if flag1 == 1:
                coef_tr=coef_tr[-len(mean_tr):]
            else:
                coef_tr=np.repeat(coef_tr,len(mean_tr))
            
            spread_tr=spread_tr[-len(mean_tr):]
            pairs_tr=pairs_training[-len(mean_tr):]
            coint_tr=np.vstack([np.ones(len(coef_tr)),coef_tr]).T
            
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            # Simulate trading on training data using the fitted parameters: Yes-1, No-0
            if if_set_trading_params:
                print ("Enter the Following trading parameters:")
                # Number of standard deviations from the mean at which trade should be initiated
                entry_point=float(0.7)
                # The permissible slippage observed on residual spread: [Enter -999 for no slippage consideration]
                slippage=float(0.1)
                # The permissible stoploss observed on residual spread: [Enter -999 for no stoploss consideration]
                stoploss=float(0.3)
                # The commission on executing a short trade: [Enter 0 for no commission consideration]
                comm_short=float(0.0002)
                # The commission on executing a long trade: [Enter 0 for no commission consideration]
                comm_long=float(0.0002)
                # Risk free rate
                rfr=float(0.0685)
                # Maximum number of days a trade can last post the trade initiation: [Enter -999 for no maximum trade duration consideration]
                max_trade_exit=int(45)
                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print ("\n The trading strategy on the training period is \n")
                # 
                buy_tr,sell_tr,status_tr,portfolio_value_tr,returns_tr=trade(
                                                                        data=pairs_tr,
                                                                        spread=spread_tr,
                                                                        mean=mean_tr,
                                                                        diffusion_eq=diffeq_tr,
                                                                        weight=coef_tr,
                                                                        entry_point=entry_point,
                                                                        slippage=slippage,
                                                                        rfr=rfr,
                                                                        max_trade_exit=max_trade_exit,
                                                                        stoploss=stoploss
                                                                        )
                trade_ticket_tr=trade_sheet(
                                    buy=buy_tr,
                                    sell=sell_tr,
                                    status=status_tr,
                                    data=pairs_tr,
                                    coint=coint_tr,
                                    commission_short=comm_short,
                                    commission_long=comm_long
                                    )
                print ("Details of Trades executed: ")
                print (trade_ticket_tr)

                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                # Print backtesting analysis of trading on training data : Yes-1, No-0
                if if_backtest:
                    pairs_tr=pairs_tr[-len(returns_tr):]
                    backtest(
                        df=trade_ticket_tr,
                        returns=returns_tr,
                        dates=pairs_tr['Date'],
                        rfr=rfr)
                    
                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                if input("\nCompute Optimized parameters for the training spread: Yes-1, No-0: ") == 1:  
                                        
                    if flag1 == 1 and flag2 ==1: 
                        if int(input("\nOptimise over delta parameter in the transition covariance matrix in Kalman Filter for both computing cointegration weight and fitting to OU process: Yes-1, No-0: ")) == 1:
                            opt_delta_resid,opt_delta_mr=optimize_delta(pairs_training,spread_tr,entry_point,pairs_tr['Date'],coef_tr,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long,5)
                            coef_tr,intercept_tr,spread_tr=dynamic_regression(pairs_training['xdata'],pairs_training['ydata'],opt_delta_resid) 
                            mean_tr,diffeq_tr=build_strategy(spread_tr,True,opt_delta_mr,False)
                            coef_tr=coef_tr[-len(mean_tr):]
                            spread_tr=spread_tr[-len(mean_tr):]
                            intercept_tr=intercept_tr[-len(mean_tr):]
                        else:
                            opt_delta_resid,opt_delta_mr=delta_r,delta_ou
                                  
                    elif flag1 == 1:
                        if int(input("\nOptimise over delta parameter in the transition covariance matrix in Kalman Filter for computing cointegration weight: Yes-1, No-0: ")) == 1:
                            if flag2 == 1:
                                opt_delta_resid=optimize_delta(pairs_training,spread_tr,entry_point,pairs_tr['Date'],coef_tr,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long,4,delta_ou)
                                coef_tr,intercept_tr,spread_tr=dynamic_regression(pairs_training['xdata'],pairs_training['ydata'],opt_delta_resid)
                                mean_tr,diffeq_tr=build_strategy(spread_tr,True,delta_ou,False)
                                
                            else:
                                opt_delta_resid=optimize_delta(pairs_training,spread_tr,entry_point,pairs_tr['Date'],coef_tr,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long,2)
                                coef_tr,intercept_tr,spread_tr=dynamic_regression(pairs_training['xdata'],pairs_training['ydata'],opt_delta_resid)
                                mean_tr,diffeq_tr=build_strategy(spread_tr,display=False)
                                
                            spread_tr=spread_tr[-len(mean_tr):]
                            coef_tr=coef_tr[-len(mean_tr):]
                            intercept_tr=intercept_tr[-len(mean_tr):]
                        else:
                            opt_delta_resid=delta_r
                        
                        
                    elif flag2 == 1:
                        if int(input("\nOptimise over delta parameter in the transition covariance matrix in Kalman Filter for fitting to OU process: Yes-1, No-0: ")) == 1:
                        #if 1==1:    
                            opt_delta_mr=optimize_delta(pairs_training,spread_tr,entry_point,pairs_tr['Date'],coef_tr,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long,3)
                            mean_tr,diffeq_tr=build_strategy(spread_tr,True,opt_delta_mr,False)
                        else:
                            opt_delta_mr=delta_ou
                        spread_tr=spread_tr[-len(mean_tr):]
                        coef_tr=coef_tr[-len(mean_tr):]
                            
                    
                    if int(input("\nOptimise over Entry Bound, Slippage, Maximum trade duration, Stoploss: Yes-1, No-0: ")) == 1:
                    #if 1==1:
                        opt_entry_point,opt_slippage,opt_max_trade_exit,opt_stoploss=optimize(pairs_tr,spread_tr,mean_tr,diffeq_tr,entry_point,pairs_tr['Date'],coint_tr,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long)
                        
                    else:
                        opt_entry_point,opt_slippage,opt_max_trade_exit,opt_stoploss=entry_point,slippage,max_trade_exit,stoploss


            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            
            if int(input("\nPerform Testing: Yes-1, No-0: ")) ==1:
                start=pd.to_datetime(input('Start Date: '))
                end=pd.to_datetime(input('End Date: '))
                pairs_testing=pair[pair['Date'] > start]
                pairs_testing=pairs_testing[pairs_testing['Date'] < end]
                pairs_te=pairs_testing
        
                ''' When Kalman Filters are used using live trading, the cointegration weights are rebalanced
                daily. To implement this we compute the coefficient and intercept for all trading session and 
                then shift it one day forward, so that for the current trading session, the cointegrating
                weights are computed from the entire data before the trading sesion. '''
                
                flag4=int(input('Use optimized parameters-1, Use training parameters-2, Enter new trading parameters-3: '))
                if flag4 !=2 and flag4 != 3:
                    flag4=1
                    
                if flag1 == 1:
                    if flag4 == 1:
                        delta_r=opt_delta_resid
                    if flag4 ==3:
                        delta_r=float(input("Delta value for Kalman Filter for computing Cointegration Weight. Default is 0.0001: "))
                    
                    coef_te,intercept_te,spread_te=dynamic_regression(pairs_testing['xdata'],pairs_testing['ydata'],delta_r)                
                    coef_te=coef_te[:-1]
                    intercept_te=intercept_te[:-1]
                    pairs_te=pairs_te[1:]
                    spread_te=pairs_te['ydata']-coef_te*pairs_te['xdata']-intercept_te
                    spread_te=spread_te.values
                    plt.figure(figsize=(24,10))
                    plt.subplot(121)
                    plt.plot(coef_tr,label='Cointegration Weight {}'.format(name2))
                    plt.title('Estimate of Cointegration Weight of {}'.format(name2))
                    plt.ylabel('Weight')
                    plt.xlabel('Trading Sessions')
                    plt.legend()
                    plt.subplot(122)
                    plt.plot(intercept_tr,label='Regression Intercept')
                    plt.ylabel('Intercept')
                    plt.title('Estimate of Cointegration Intercept')
                    plt.xlabel('Trading Sessions')
                    plt.legend()
                    plt.show()

                    
                else:
                    coef_te=coef_tr[0]
                    intercept_te=intercept_tr
                    spread_te=pairs_te['ydata']-coef_te*pairs_te['xdata']-intercept_te
                    spread_te=spread_te.values
                

                if flag2 == 1:
                    if flag4 == 1:
                        delta_ou=opt_delta_mr
                    if flag4 ==3:
                        delta_ou=float(input("Delta value for Kalman Filter for fitting to OU process. Default is 0.0001: "))
                    mean_te,diffeq_te=build_strategy(spread_te,True,delta_ou,True)

                else:
                    mean_te=np.repeat(mean_tr[0],pairs_te.shape[0])
                    diffeq_te=np.repeat(diffeq_tr[0],pairs_te.shape[0])                
        
                if flag1 == 1:
                    coef_te=coef_te[-len(mean_te):]
                else:
                    coef_te=np.repeat(coef_te,len(mean_te))
                    
                spread_te=spread_te[-len(mean_te):]
                coef_te=coef_te[-len(mean_te):]
                pairs_te=pairs_te[-len(mean_te):]
                coint_te=np.vstack([np.ones(len(coef_te)),coef_te]).T
    
                if flag4 == 1:
                    entry_point=opt_entry_point
                    slippage=opt_slippage
                    max_trade_exit=opt_max_trade_exit
                    stoploss=opt_stoploss
                        
                if flag4 == 3:
                    print ("\nEnter the Following trading parameters: ")
                    entry_point=float(input("Number of standard deviations from the mean at which trade should be initiated: "))
                    slippage=float(input("The permissible slippage observed on residual spread: "))
                    max_trade_exit=int(input("Maximum number of days a trade can last post the trade initiation: "))
                    stoploss=float(input("The permissible stoploss observed on the residual spread: "))
                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print ("\n The trading strategy on the testing period is \n")
                
                
                buy_te,sell_te,status_te,portfolio_value_te,returns_te=trade(pairs_te,spread_te,mean_te,diffeq_te,coef_te,entry_point,slippage,rfr,max_trade_exit,stoploss)
                trade_ticket_te=trade_sheet(buy_te,sell_te,status_te,pairs_te,coint_te,comm_short,comm_long)
                print ("\nDetails of Trades executed: ")
                print (trade_ticket_te)
                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print ("\n\nAnalysis of performance of trading on testing data: ")                     
                backtest(trade_ticket_te,returns_te,pairs_te['Date'],rfr)
                
            else:
                print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
                print ("\nNo testing performed. The program is terminated")
        else:
            print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
            print ("\nNo simulation of trading on training or testing analysis performed. The program is terminated")
    else:
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print ("\nNo pairs trading statistical arbitrage strategy exists for this pair. The program is terminated ")


if __name__ == '__main__':
    pass

