
# Importing Libraries
from sklearn.linear_model import LinearRegression as lm 
import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import datetime as date
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.stats import t,ttest_ind,norm,iqr
from sklearn.metrics import r2_score
from numpy.linalg import inv

# Ignoring Warnings produced
import warnings
warnings.filterwarnings('ignore')



def regression(xdata,ydata):
    flag=0
    if isinstance(xdata, pd.DataFrame):
        flag=1
    xdat=pd.DataFrame(xdata)
    xdat['b0']=1
    xdat=xdat.values
    ydata=ydata.values
    n= np.dot(xdat.T,xdat)
    beta = np.dot(np.dot(inv(n),xdat.T),ydata) 
    coef=beta[0:-1]
    intercept=beta[-1]
    
    if flag == 1:
        xdata.drop(labels='b0',axis=1,inplace=True)
        temp=coef*xdata
        residuals=ydata-temp.sum(axis=1)-intercept
    else:
        coef=coef[0]
        residuals=ydata-coef*xdata-intercept
    
    return coef,intercept,residuals.values

# Producing Dynamic Estimates of Regression Parameters
def dynamic_regression(xdata,ydata,delta=1e-4):
    observation_matrix=np.vstack([xdata,np.ones(xdata.shape[0])]).T[:, np.newaxis]
    
    #Delta coefficient will determine the frequency of rebalancing estimates
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=observation_matrix,
                  observation_covariance=1.0,
                  transition_covariance=trans_cov)

    state_means, state_covs = kf.filter(ydata)

    slope=state_means[:,0]
    intercept=state_means[:,1]

    return slope,intercept,ydata.values-slope*xdata.values-intercept

def test_stationarity(residuals):
    '''
    ADF test

    Δy_{t} = η*y_{t-1} + β*Δy_{t-1} + ... 回归

    y_{t-1} 的 t_value < 临界值，拒绝原假设，平稳（不存在单位根） 
    '''

    # Augmenting 1-period lag and 1 period lag of delta of lag into the dataset
    adf_data=pd.DataFrame(residuals)
    adf_data.columns=['y']
    adf_data['drift_constant']=1
    adf_data['y-1']=adf_data['y'].shift(1)
    adf_data.dropna(inplace=True)
    adf_data['deltay1']=adf_data['y']-adf_data['y-1']
    adf_data['deltay-1']=adf_data['deltay1'].shift(1)
    adf_data.dropna(inplace=True)
    target_y=pd.DataFrame(adf_data['deltay1'],columns=['deltay1'])
    adf_data.drop(['y','deltay1'],axis=1,inplace=True)
    
    #Auto regressing the residuals with lag1, drift constant and lagged 1 delta (delta_et-1)
    adf_regressor_model=sm.OLS(target_y,adf_data)
    adf_regressor=adf_regressor_model.fit()

    # Returning the results
    return adf_regressor

def test_significance(xdata,ydata,residuals):
    '''
    ECM

    ecm_{t-1} = y_{t-1} - a0 - a1*x_{t-1}

    Δy_{t} = β*Δx_{t} + gamma*ecm_{t-1} + ... 回归
    
    gamma 称为调整系数
    '''

    # Augmenting 1-period lagged residual into the dataset
    residuals=pd.DataFrame(residuals)
    ecm_data=pd.DataFrame(residuals.shift(1))
    ecm_data.columns=['et-1']
    ecm_data['y1']=ydata.values
    ecm_data['y2']=xdata.values
    ecm_data['deltay']=ecm_data['y1']-ecm_data['y1'].shift(1)
    ecm_data['deltax']=ecm_data['y2']-ecm_data['y2'].shift(1)
    ecm_data.dropna(inplace=True)

    
    target_y=pd.DataFrame(ecm_data['deltay'])
    ecm_data.drop(['y1','y2','deltay'],axis=1,inplace=True)
    
    
    # Regressing the delta y against the delta x and the 1 period lagged residuals 
    ecm_regressor1_model=sm.OLS(target_y,ecm_data)
    ecm_regressor1=ecm_regressor1_model.fit()
    
    # Returning the results of the regression
    return ecm_regressor1  

def cointegration_test(xdata,ydata,stat_value_ci,sig_value_ci,s1,s2,print_summary=False):
    '''
    Return
    - -1 : 未通过平稳性检验
    - -2 : 未通过显著性测试
    '''
    
    adf_critical_values1={'0.99':-3.46, '0.95':-2.88,'0.9':-2.57}
    adf_critical_values2={'0.99':-3.44,'0.95':-2.87,'0.9':-2.57}
    adf_critical_values3={'0.99':-3.43,'0.95':-2.86,'0.9':-2.57}

    # 回归求得残差
    coef1,intercept1,residuals1=regression(xdata,ydata)
    coef2,intercept2,residuals2=regression(ydata,xdata)
    flag=0 
    flag1=0
    
    # 平稳性检验(ADF): 对残差进行自回归，检验残差是否平稳 （残差是否存在单位根）
    stat_test=test_stationarity(residuals=residuals1)
    print("\nThe following is the result of the Augmented Dickey Fuller test")
    if print_summary:
        print(stat_test.summary())
    if len(residuals1) > 500:
        if abs(stat_test.tvalues['y-1']) > abs(adf_critical_values3[str(stat_value_ci)]):
            print('通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is rejected. Hence, no unit root exists and residuals are stationary".format(stat_test.tvalues['y-1']))
            #pass
        else:
            print('【FAIL】未通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is accepted. Hence, a unit root exists and residuals are not stationary and Error Correction Model is not checked for".format(stat_test.tvalues['y-1']))
            return -1
            
    elif len(residuals1) > 250:
        if abs(stat_test.tvalues['y-1']) > abs(adf_critical_values2[str(stat_value_ci)]):
            print('通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is rejected. Hence, no unit root exists and residuals are stationary".format(stat_test.tvalues['y-1']))
            #pass
        else:
            print('【FAIL】未通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is accepted. Hence, a unit root exists and residuals are not stationary and Error Correction Model is not checked for".format(stat_test.tvalues['y-1']))
            return -1
        
    elif len(residuals1) > 100:
        if abs(stat_test.tvalues['y-1']) > abs(adf_critical_values1[str(stat_value_ci)]):
            print('通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is rejected. Hence, no unit root exists and residuals are stationary".format(stat_test.tvalues['y-1']))
            #pass
        else:
            print('【FAIL】未通过平稳性检验!\nt_value={}, 临界值={}\n'.format(stat_test.tvalues['y-1'], adf_critical_values3[str(stat_value_ci)]))
            # print("\nThe t-statistic value of the unit root coefficient is {} and the null hypothesis of a unit root is accepted. Hence, a unit root exists and residuals are not stationary and Error Correction Model is not checked for".format(stat_test.tvalues['y-1']))  
            return -1
    

    ## 若通过平稳性检验，则说明x和y存在一个长期的均衡关系，但是短期内会出现非均衡状态，
    ## 那么x和y必须进行动态修正和调整，使得非均衡状态尽量恢复到均衡状态。
    # 显著性测试(ECM)
    sig_1=test_significance(xdata,ydata,residuals1)
    sig_2=test_significance(ydata,xdata,residuals2)
        
        
    print("\nThe following is the regression result of the Error Correction model when {} is the independent and {} is the dependent variable".format(s1,s2))
    if print_summary:
        print(sig_1.summary())
    critical_value=abs(t.ppf(sig_value_ci+0.5*(1-sig_value_ci),len(residuals1)))
    if abs(sig_1.tvalues['et-1']) > critical_value:
        print('【1】通过显著性测试!\nt_value={}, 临界值={}\n'.format(sig_1.tvalues['et-1'], critical_value))
        # print("\nThe t-statistic value of the lagged residual coefficient in the error correction model is {} against a critical value of {} and the null hypothesis of the coefficient not being significant is rejected. Hence, cointegration is significant".format(sig_1.tvalues['et-1'],critical_value))
    else:
        print('【1】未通过显著性测试!\nt_value={}, 临界值={}\n'.format(sig_1.tvalues['et-1'], critical_value))
        # print("\nThe t-statistic value of the lagged residual coefficient in the error correction model is {} against a critical value of {} and the null hypothesis of the coefficient not being significant is accepted. Hence, cointegration is not significant".format(sig_1.tvalues['et-1'],critical_value))
        flag1+=1        
       
    print("\nThe following is the regression result of the Error Correction model when {} is the independent and {} is the dependent variable".format(s2,s1))
    if print_summary:
        print(sig_2.summary())
    critical_value=abs(t.ppf(sig_value_ci+0.5*(1-sig_value_ci),len(residuals2)))
    if abs(sig_2.tvalues['et-1']) > critical_value:
        print('【2】通过显著性测试!\nt_value={}, 临界值={}\n'.format(sig_1.tvalues['et-1'], critical_value))
        # print("\nThe t-statistic value of the lagged residual coefficient in the error correction model is {} against a critical value of {} and the null hypothesis of the coefficient not being significant is rejected. Hence, cointegration is significant".format(sig_2.tvalues['et-1'],critical_value))   
    else:
        print('【2】未通过显著性测试!\nt_value={}, 临界值={}\n'.format(sig_1.tvalues['et-1'], critical_value))
        # print("\nThe t-statistic value of the lagged residual coefficient in the error correction model is {} against a critical value of {} and the null hypothesis of the coefficient not being significant is accepted. Hence, cointegration is not significant".format(sig_2.tvalues['et-1'],critical_value))
        flag1+=1

    if flag1 == 2:
        print('【FAIL】二者都未通过显著性测试!')
        return -2
    
    if abs(sig_1.tvalues['et-1']) < abs(sig_2.tvalues['et-1']):
        print('\n对于这个协整问题，自变量x是{}，因变量y是{}'.format(s1,s2))
        # print("\nFor the cointegration problem, the independent variable in regression between the asset classes is {} and the dependent variable is {}".format(s1,s2))
        return 2
    else:
        print('\n对于这个协整问题，自变量x是{}，因变量y是{}'.format(s2,s1))
        # print("\nFor the cointegration problem, the independent variable in regression between the asset classes is {} and the dependent variable is {}".format(s2,s1))
        return 1

def robustness(xdata,ydata,long_xdata,long_ydata,ci):
    
    # Finding Cointegration Weights of Short Period
    coef,intercept,resid=regression(xdata,ydata)
    # Finding Cointegration Weights of Long Period 
    long_coef,long_intercept,long_resid=regression(long_xdata,long_ydata)
    
    # Testing the R-squared of the cointegration weight
    ecm_object=test_significance(xdata,ydata,resid)
    print("\nThe R2 score of the error correction model is {}. This means the lagged spread explains {}% of the total variance. ".format(ecm_object.rsquared,ecm_object.rsquared*100))

    t_statistic,p_value=ttest_ind(resid,long_resid)
    
    if abs(t_statistic) < t.ppf(ci,len(xdata)):
        print ("\nThe t-statistic is {} and the spread over 2 periods are similar according to t-statistic test".format(t_statistic))
        print ("通过稳健性测试!")
    else:
        print ("\nThe t-statistic is {} and the spread over 2 periods are not similar according to t-statistic test".format(t_statistic))
        print ("【FAIL】未通过稳健性测试! As co-integration is not significant, consider the use of Kalman Filters")

