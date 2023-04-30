import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


with open('./output_data.csv', 'r') as file:
    data = pd.read_csv(file)

y = np.mean(data['output_own_sales'].to_numpy().reshape((54, 365)), axis=0)
x1 = np.mean(data['output_own_price'].to_numpy().reshape((54, 365)), axis=0)
x2 = np.mean(data['output_comp_price'].to_numpy().reshape((54, 365)), axis=0)

#Model own demand using Linear Regression, 
#independet variable - difference between avg daily own prices and competitors price
diff = x1 - x2
x_train, x_test, y_train, y_test = train_test_split(diff.reshape(-1,1),y.reshape(-1,1), test_size=0.1, random_state=42)
reg = LinearRegression().fit(x_train, y_train)
reg.score(x_train, y_train)
weight = reg.coef_
b = reg.intercept_


def get_lags(new_name, old_name):
    data[new_name] = data[old_name]
    for n in range(0, len(data), 365):
        data.loc[n:n+7,new_name] = data.loc[n:n+365, old_name].shift(7)
    return data.loc[:,new_name]


data['output_date'] = pd.to_datetime(data['output_date'])

#Create discrete variables
data['month'] = data.output_date.dt.month
data['day_week'] = data.output_date.dt.day_of_week


#Create continuous variables
get_lags('lag_7_share', 'output_own_share')
get_lags('lag_7_comp_price', 'output_comp_price')
get_lags('lag_7_output_x', 'output_X')
get_lags('lag_7_sales', 'output_own_sales')
get_lags('lag_7_price', 'output_own_price')


predictors = data[['mkt_id', 'month', 'day_week','output_own_cost', 
                    'lag_7_share', 'lag_7_comp_price', 'lag_7_output_x', 'lag_7_sales', 'lag_7_price']]

whole_data = data.copy()

#Process data
def process_training_data():
    def drop_vals():
        data.drop(data[predictors.lag_7_share.isnull() == True].index, inplace=True)
        predictors.dropna(inplace=True)
    drop_vals()

    def get_time_split():
        X_train = predictors[(predictors['month'] != 11) & (predictors['month'] != 12)].to_numpy()
        X_val = predictors[predictors['month'] == 11].to_numpy()
        X_test =  predictors[predictors['month'] == 12].to_numpy()
        Y_train = data[(data['month'] != 11) & (data['month'] != 12)]['output_own_price'].to_numpy().reshape(-1,1)
        Y_val = data[data['month'] == 11]['output_own_price'].to_numpy().reshape(-1,1)
        Y_test = data[data['month'] == 12]['output_own_price'].to_numpy().reshape(-1,1)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
        
    def get_comp_price_costs():
        comp_price_val = data[data['month'] == 11]['output_comp_price'].to_numpy().reshape(-1,1) 
        comp_price_test = data[data['month'] == 12]['output_comp_price'].to_numpy().reshape(-1,1) 
        costs_val = predictors[predictors['month'] == 11]['output_own_cost'].to_numpy()
        costs_test = predictors[predictors['month'] == 12]['output_own_cost'].to_numpy()
        return comp_price_val, costs_val,  comp_price_test, costs_test
    
    return get_time_split(), get_comp_price_costs()
    

X_train, X_val, X_test, Y_train, Y_val, Y_test = process_training_data()[0]

#Train model
def objective_function(p1,p2, cost):
    return (p1-cost)*(b[0] + weight[0][0]*(p1-p2))

def train_model(hp1, hp2, hp3):
    reg = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=hp1), random_state=42, n_estimators=hp2, learning_rate=hp3)
    reg.fit(X_train, Y_train)
    return reg

def get_profit(prediction, moment, profit=0):
    if moment == 'test': comp_price, cost = process_training_data()[1][2:]
    else: comp_price, cost = process_training_data()[1][:2]
    for n, p in enumerate(prediction):
        profit += objective_function(p, comp_price[n][0], cost[n])
    return profit

def optimization(hp1, hp2, hp3):
    reg = train_model(hp1, hp2, hp3)
    validation = reg.predict(X_val)
    return get_profit(validation, 'validation')

#Get ideal hyperparameter for the task in hands
h_p = 0
max_profit = 0

def check_values(profit, hp1, hp2, hp3, max_profit, h_p):
    if profit > max_profit:
        h_p = (hp1, hp2, hp3)
        max_profit = profit
    return h_p, max_profit

for hp1 in range(1, 3, 1):
    for hp2 in range(20, 201, 20):
            for hp3 in np.arange(0.7,0.9,0.02):
                profit = optimization(hp1, hp2, hp3)
                h_p, max_profit = check_values(profit, hp1, hp2, hp3, max_profit, h_p)
                #print('max_depth: {} \nn_estimators: {} \nlearning_rate: {} \nprofit: {} \n'.format(hp1, hp2, hp3, profit))


#Test results 
def test(x,y):
    reg = train_model(h_p[0], h_p[1], h_p[2])
    test = reg.predict(x)
    return round(get_profit(test, 'test'), 2)

def expected_gains():
    past_price = data[data['month'] == 12]['output_own_price'].to_numpy()
    real_price_profit = round(get_profit(past_price, 'test'), 2)
    predicted_price_profit = test(X_test, Y_test)
    return round(((predicted_price_profit/real_price_profit)-1) * 100, 2)

print('Increase in profit: {} % '.format(expected_gains()))


#Getting new inputs
lst_new_frames = []
future = pd.date_range('2020-01-01', '2020-01-31')
future_df = pd.DataFrame(future, columns=['output_date'])
future_df['is_future'] = True
data['is_future'] = False
for i in range(0, whole_data.shape[0], 365):
    chunk = whole_data.iloc[i:i+365]
    
    p_f = pd.concat([chunk, future_df]).reset_index()
    p_f.iloc[365:396]['day_week'] = list(p_f.iloc[0:31]['day_week'])
    p_f['mkt_id'] = p_f.mkt_id.fillna(p_f.iloc[0]['mkt_id'])
    p_f['month'] = p_f.month.fillna(p_f.iloc[0]['month'])
    for feature in [('output_own_cost', 'output_own_cost'), 
                        ('output_own_share', 'lag_share'), ('output_comp_price', 'lag_comp_price'), 
                        ('output_X', 'lag_output_x'), ('output_own_sales', 'lag_sales'), ('output_own_price', 'lag_price')]:
        p_f[feature[1]] = p_f[feature[0]].shift(31)
       
    p_f = p_f[p_f.is_future == True]
    p_f = p_f[['output_date', 'mkt_id', 'month', 'day_week', 'output_own_cost', 'lag_share', 'lag_comp_price',
                'lag_output_x', 'lag_sales', 'lag_price']]
    lst_new_frames.append(p_f)


#Predict prices for all markets Jan 2020
def predict_price(inputs, reg):
    price_lst = []
    for market in inputs:
        input = market.iloc[:,1:].to_numpy()
        price = reg.predict(input)

        price_lst.append(price.round(decimals=2))
    markets = list(data['mkt_id'].unique())
    arr_time = []
    for i in future:
        arr_time.append(str(i).split()[0])
    
    price = pd.DataFrame(data=np.array(price_lst).T, columns=markets)
    price.index = [arr_time]
    return price


predict_price(lst_new_frames, train_model(h_p[0], h_p[1], h_p[2])).to_csv('./prices.csv')