

# Importing Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# Ignoring Warnings produced
import warnings
warnings.filterwarnings('ignore')

# Importing functions
from stats_func import (
    dynamic_regression,
)
from backtest import (
    trade,
    trade_sheet,
    backtest,
    build_strategy
)



def optimization_plot(cum_rets,es,sharpe,risk,avg_duration,avg_profit,xdata,xlabel,opt_ind,opt_crit,opt_data):
    plot_data=[avg_profit,cum_rets,sharpe,risk,es,avg_duration]
    plot_labels=['Avg. Profit','Cumulative Return','Sharpe Ratio','Risk','Expected Shortfall','Average Duration']
    if opt_crit not in plot_labels:
        plot_labels.append(opt_crit)
        plot_data.append(opt_data)
    
    plots=len(plot_data)
    plt.figure(figsize=(15, 7))
    for lab,i in zip(plot_labels,range(1,len(plot_labels)+1)):
        plt.subplot(int('{}{}{}'.format(plots,1,i)))
        plt.plot(xdata,plot_data[i-1])
        plt.plot(xdata[opt_ind:opt_ind+1],plot_data[i-1][opt_ind:opt_ind+1],marker='o',label='Optimal Parameter')
        #plt.xlabel(xlabel)
        plt.ylabel(plot_labels[i-1])
        plt.title('{} v/s {}'.format(plot_labels[i-1],xlabel))
        plt.legend()
    
    plt.xlabel(xlabel)
    plt.show()    

def optimization_results(report,lower_bound, upper_bound,optimization_label,optimization_criteria,flag,display=False):
    form=''
    valid_trades=report[report['Total Trades'] > 0]
    # Only optimising where valid trades are made
    if valid_trades.shape[0] > 0:
        if flag==0:
            opt_ind=valid_trades[optimization_criteria].idxmin()
            if opt_ind == len(report)-1:
                if optimization_label == 'Residual Delta/Mean Reversion Delta':
                    optimal_resid_delta=valid_trades[-1:]['Residual Delta'].values
                    optimal_mr_delta=valid_trades[-1:]['Mean Reversion Delta'].values
                else:
                    optimal_parameter=valid_trades[-1:][optimization_label].values
            else:
                if optimization_label == 'Residual Delta/Mean Reversion Delta':
                    optimal_resid_delta=valid_trades[opt_ind:opt_ind+1]['Residual Delta'].values
                    optimal_mr_delta=valid_trades[opt_ind:opt_ind+1]['Mean Reversion Delta'].values
                else:
                    optimal_parameter=valid_trades[opt_ind:opt_ind+1][optimization_label].value
            form='minimisation'
        
        else:
            opt_ind=valid_trades[optimization_criteria].idxmax()
            if opt_ind == len(report)-1:
                if optimization_label == 'Residual Delta/Mean Reversion Delta':
                    optimal_resid_delta=valid_trades[-1:]['Residual Delta'].values
                    optimal_mr_delta=valid_trades[-1:]['Mean Reversion Delta'].values
                else:
                    optimal_parameter=valid_trades[-1:][optimization_label].values
            else:
                if optimization_label == 'Residual Delta/Mean Reversion Delta':
                    optimal_resid_delta=valid_trades[opt_ind:opt_ind+1]['Residual Delta'].values
                    optimal_mr_delta=valid_trades[opt_ind:opt_ind+1]['Mean Reversion Delta'].values
                else:
                    optimal_parameter=valid_trades[opt_ind:opt_ind+1][optimization_label].values
            form='maximisation'
                    
        if display == True:
            print ("The optimization report for {} is :".format(optimization_label))
            print (report)
            
        if optimization_label == 'Residual Delta/Mean Reversion Delta':
            print ("Optimal Residual Delta and Mean Reversion Delta of a trade for this pair with {} of {} is {} and {}".format(form,optimization_criteria,optimal_resid_delta,optimal_mr_delta))
            df_rec=report['Residual Delta'].astype(str)+'/'+report['Mean Reversion Delta'].astype(str)
            print ("\n The optimization plot for {} is: ".format(optimization_label))
            df_rec=report['Residual Delta'].astype(str)+'/'+report['Mean Reversion Delta'].astype(str)
            optimization_plot(report['Cumulative Return'],report['Expected Shortfall'],report['Sharpe Ratio'],report['Standard Deviation of Returns'],report['Average Trade Duration'],report['Average Profit'],df_rec,'Residual Delta/Mean Reversion Delta',opt_ind,optimization_criteria,report[optimization_criteria])
            return optimal_resid_delta[0],optimal_mr_delta[0]
    
        else:
            print ("Optimal {} for this spread with {} of {} is {}".format(optimization_label,form,optimization_criteria,optimal_parameter))
            print ("\n The optimization plot for {} is: ".format(optimization_label))
            optimization_plot(report['Cumulative Return'],report['Expected Shortfall'],report['Sharpe Ratio'],report['Standard Deviation of Returns'],report['Average Trade Duration'],report['Average Profit'],report[optimization_label],optimization_label,opt_ind,optimization_criteria,report[optimization_criteria])
            return optimal_parameter[0]
            
    else:
        print("No trades were made for any specifed value of {}. The return parameters are average of the minimal and maximal bounds".format(optimization_label))
        if optimization_label == 'Residual Delta/Mean Reversion Delta':
            optimal_resid_delta=[(upper_bound+lower_bound)*0.5]
            optimal_mr_delta=[(upper_bound+lower_bound)*0.5]
            return optimal_resid_delta[0],optimal_mr_delta[0]

        else:       
            optimal_parameter=[(upper_bound+lower_bound)*0.5]
            return optimal_parameter[0]

def optimize_parameter(lower_bound, upper_bound,optimization_label,optimization_criteria,flag,val,display=False,*trade_parameters):
    data,spread,mean,entry_point,diffusion_eq,coint,slippage,rfr,max_trade_exit,stoploss,comm_short,comm_long,dates=trade_parameters
    
    # Generating set of probable parameter values
    if optimization_label == 'Maximum Trade Duration':
        probables=np.unique(np.linspace(lower_bound,upper_bound,val,dtype=int))

    else:
        probables=np.linspace(lower_bound,upper_bound,val)
    
    # Generating set of probable parameter values
    form=''
    report=pd.DataFrame()
    weight=np.repeat(coint[0][1],len(mean))
    data=data[-len(mean):]
    for i in probables:
            if optimization_label == 'Entry Bound':
                buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffusion_eq,weight,i,slippage,rfr,max_trade_exit,stoploss,plot=False)
            if optimization_label == 'Slippage':
                buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffusion_eq,weight,entry_point,i,rfr,max_trade_exit,stoploss,plot=False)
            if optimization_label == 'Maximum Trade Duration':
                buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffusion_eq,weight,entry_point,slippage,rfr,i,stoploss,plot=False)
            if optimization_label == 'Stoploss':
                buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffusion_eq,weight,entry_point,slippage,rfr,max_trade_exit,i,plot=False)
            
            df_temp=trade_sheet(buy,sell,status,data,coint,comm_short,comm_long)
            df,temp,tempu=backtest(df_temp,returns,dates,rfr,display=False)
            df[optimization_label]=i
            report=report.append(df,ignore_index=True)
            
    optimal_parameter=optimization_results(report,lower_bound, upper_bound,optimization_label,optimization_criteria,flag)
    return optimal_parameter

def optimize_residual_delta(lower_bound, upper_bound,optimization_criteria,flag,val,ou_flag,display=False,*trade_parameters):

    ou_delta,data,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates=trade_parameters
    # Generating set of probable parameter values
    probables=np.geomspace(lower_bound,upper_bound,val)
    report=pd.DataFrame()
    
    for i in probables:
        #Simulate trading for the given parameter
        a,b,spread=dynamic_regression(data['xdata'],data['ydata'],i)
        if ou_flag == 1:
            mean,diffeq=build_strategy(spread,True,ou_delta,display=False)
        else:
            mean,diffeq=build_strategy(spread,display=False)
            
        spread=spread[-len(mean):]
        data=data[-len(mean):]
        a=a[-len(mean):]
        buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffeq,a,entry_point,slippage,rfr,max_trade_exit,stoploss,plot=False)
        coint=np.vstack([np.ones(len(a)),a]).T
        df_temp=trade_sheet(buy,sell,status,data,coint,commission_short,commission_long)
        df,temp,tempu=backtest(df_temp,returns,dates,rfr,False)
        df['Residual Delta']=i
        report=report.append(df,ignore_index=True)
    
    # Finding the optimal parameter based on the given criteria
    optimal_parameter=optimization_results(report,lower_bound, upper_bound,'Residual Delta',optimization_criteria,flag,display)
    return optimal_parameter

def optimize_mean_reversion_delta(lower_bound,upper_bound,optimization_criteria,flag,val,display=False,*trade_parameters):
    weight,data,spread,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates=trade_parameters
    
    # Generating set of probable parameter values
    probables=np.geomspace(lower_bound,upper_bound,val)
    report=pd.DataFrame()

    for i in probables:
        
        #Simulate trading for the given parameter
        mean,diffeq=build_strategy(spread,True,i,display=False)
        spread=spread[-len(mean):]
        data=data[-len(mean):]
        weight=weight[-len(mean):]
        buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffeq,weight,0.4,slippage,rfr,max_trade_exit,stoploss,plot=False)
        coint=np.vstack([np.ones(len(weight)),weight]).T
        df_temp=trade_sheet(buy,sell,status,data,coint,commission_short,commission_long)
        df,temp,tempu=backtest(df_temp,returns,dates,rfr,False)
        df['Mean Reversion Delta']=i
        report=report.append(df,ignore_index=True)
    
    optimal_parameter=optimization_results(report,lower_bound, upper_bound,'Mean Reversion Delta',optimization_criteria,flag,display)
    return optimal_parameter

def optimize_both_delta(lower_bound, upper_bound,optimization_criteria,flag,val,display=False,*trade_parameters):

    data,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates=trade_parameters
    # Generating set of probable parameter values
    probables_resid=np.geomspace(lower_bound,upper_bound,val)
    probables_ou=np.geomspace(lower_bound,upper_bound,val)
    report=pd.DataFrame()
    
    for i in probables_resid:
        for j in probables_ou:
        #Simulate trading for the given parameter
            a,b,spread=dynamic_regression(data['xdata'],data['ydata'],i)
            mean,diffeq=build_strategy(spread,True,j,display=False)
                    
            spread=spread[-len(mean):]
            data=data[-len(mean):]
            a=a[-len(mean):]
            buy,sell,status,portfolio_value,returns=trade(data,spread,mean,diffeq,a,entry_point,slippage,rfr,max_trade_exit,stoploss,plot=False)
            coint=np.vstack([np.ones(len(a)),a]).T
            df_temp=trade_sheet(buy,sell,status,data,coint,commission_short,commission_long)
            df,temp,tempu=backtest(df_temp,returns,dates,rfr,False)
            df['Residual Delta']=i
            df['Mean Reversion Delta']=j
            report=report.append(df,ignore_index=True)
    
        
    optimal_resid_delta,optimal_mr_delta=optimization_results(report,lower_bound, upper_bound,'Residual Delta/Mean Reversion Delta',optimization_criteria,flag,display)
    return optimal_resid_delta,optimal_mr_delta

def optimize(data,spread,mean,diffusion_eq,entry_point,dates,coint,slippage,rfr,max_trade_exit,stoploss,commission_short=0,commission_long=0):
    
    params=(data,spread,mean,entry_point,diffusion_eq,coint,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates)
    for i in range(0,4):
        inp=int(input("Enter optimisation parameter: Entry Point - 1, Slippage - 2, Max Duration - 3, Stoploss - 4, Exit - 5: "))
        #inp=inp+1
        if inp == 5:
            break
        
        lower_bound=float(input("Lower Bound of the optimization criteria: "))
        upper_bound=float(input("Upper Bound of the optimization criteria: "))
        val=int(input("Number of interpolations for each parameter: "))
        optimization_criteria=input("Optimization attribute: Total Trades, Complete Trades, Incomplete Trades, Profit Trades, Loss Trades, Total Profit, Average Trade Duration, Average Profit, Win Ratio, Average Profit on Profitable Trades, Standard Deviation of Returns, Value at Risk, Expected Shortfall, Sharpe Ratio, Sortino Ratio, Cumulative Return, Market Alpha, Market Beta, HML Beta, SMB Beta, WML Beta, Momentum Beta,Fama French Four Factor Alpha: ")
        flag=int(input("Minimise-0, Maximise-1: "))
        if inp == 1:
            entry_point=optimize_parameter(lower_bound,upper_bound,'Entry Bound',optimization_criteria,flag,val,False,*params)

        elif inp == 2:
            slippage=optimize_parameter(lower_bound,upper_bound,'Slippage',optimization_criteria,flag,val,False,*params)
            
        elif inp == 3:
            
            max_trade_exit=optimize_parameter(lower_bound,upper_bound,'Maximum Trade Duration',optimization_criteria,flag,val,False,*params)
        
        elif inp == 4:
            
            stoploss=optimize_parameter(lower_bound,upper_bound,'Stoploss',optimization_criteria,flag,val,False,*params)


    return entry_point,slippage,max_trade_exit,stoploss

def optimize_delta(data,spread,entry_point,dates,weight,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,flag1,ou_delta=0):
    lower_bound=float(input("Lower Bound of the optimization criteria: "))
    upper_bound=float(input("Upper Bound of the optimization criteria: "))
    val=int(input("Number of interpolations for each parameter: "))
    optimization_criteria=input("Optimization attribute: Total Trades, Complete Trades, Incomplete Trades, Profit Trades, Loss Trades, Total Profit, Average Trade Duration, Average Profit, Win Ratio, Average Profit on Profitable Trades, Standard Deviation of Returns, Value at Risk, Expected Shortfall, Sharpe Ratio, Sortino Ratio, Cumulative Return, Market Alpha, Market Beta, HML Beta, SMB Beta, WML Beta, Momentum Beta,Fama French Four Factor Alpha: ")
    flag2=int(input("Minimise-0, Maximise-1: "))


    if flag1 == 2 or flag1 == 4:
        trade_parameters=(ou_delta,data,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates)
        if flag1 == 4:
            resid_delta=optimize_residual_delta(lower_bound,upper_bound,optimization_criteria,flag2,val,1,False,*trade_parameters)
        else:
            resid_delta=optimize_residual_delta(lower_bound,upper_bound,optimization_criteria,flag2,val,0,False,*trade_parameters)
        return resid_delta
        
    if flag1 == 3:
        trade_parameters=(weight,data,spread,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates)
        return optimize_mean_reversion_delta(lower_bound,upper_bound,optimization_criteria,flag2,val,False,*trade_parameters)
        
    if flag1 == 5:
        trade_parameters=(data,entry_point,slippage,rfr,max_trade_exit,stoploss,commission_short,commission_long,dates)
        return optimize_both_delta(lower_bound,upper_bound,optimization_criteria,flag2,val,False,*trade_parameters)
