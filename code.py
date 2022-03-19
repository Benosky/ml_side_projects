# Import relevant libraries

import pandas as pd
import pandasql as ps
import numpy as np
# import datetime
import matplotlib.pyplot as plt
import threading
from sklearn.model_selection import train_test_split

import datetime as dt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor


#--------------------------------------------------------

# MERGING THE DATA INTO A SINGLE FILE

#--------------------------------------------------------


# Create a function for joining country level data

def country_df(customers_df, orders_df, deliveries_df, resp):
    """
    args:
        customers_df -> country customers DataFrame
        orders_df -> country orders DataFrame
        deliveries_df -> country deliveries DataFrame
        
    return:
        country_df -> A DataFrame containing the join of all data for a country
    """
    # Ensure all the column lables are in upper case to maintain uniformity
    customers_df.columns = [x.upper() for x in customers_df.columns.values]
    orders_df.columns = [x.upper() for x in orders_df.columns.values]
    deliveries_df.columns = [x.upper() for x in deliveries_df.columns.values]
    
    # Query for extracting and joining the data
    data_query = """
                    -- I separated this joins to ensure that the query runs a little faster while capturing
                    -- data for customers with no order and deliveries data
                    WITH orders_deliveries AS(
                            SELECT 
                                *
                            -- FROM customers_df
                            FROM orders_df  -- I did a left join to capture data of customers who may have not made any orders
                                -- ON customers_df."Customer Id" = orders_df."Customer ID"
                            INNER JOIN deliveries_df -- Same as above
                                ON orders_df."ORDER ID" = SUBSTRING(deliveries_df."ORDER_ID", 4, 8)
                        )
                        SELECT 
                            *
                        FROM customers_df
                        LEFT OUTER JOIN orders_deliveries
                            ON customers_df."CUSTOMER ID" = orders_deliveries."CUSTOMER ID"
                    """    
    # Extract, join and load the full country data to a single DataFrame
    country_df = ps.sqldf(data_query, locals())
    
    # Drop columns with no labels
    country_df = country_df.loc[:, ~country_df.columns.str.contains("UNNAMED")]
    
    # Drop the duplicate columns resulting from the joining of the various tables
    country_df = country_df.loc[:,~country_df.columns.duplicated()]
    resp.append(country_df) 



#--------------------------------------------------------

# CLEANING THE DATA

#--------------------------------------------------------


# Create a function for handling numeric, string and datetime variables

def handle_numeric_string_date_types(df, num_variables, str_variables, dt_variables, resp):
    # Convert the expected numeric columns to numbers and replace the non-numerics with zero. The non-numeric values in the 
    # numeric columns are most '-'. So replacing them with zero will do little of no harm.
    df = df.copy()
    for x in num_variables:
        df[x] = pd.to_numeric(df[x], downcast='unsigned', errors='coerce')
    
    # Replace NaN in the numeric columns with 0
    df[num_variables] = df[num_variables].fillna(0)
    
    # Replace "-", 0 and ''  in the string columns with NaN
    df[str_variables] = df[str_variables].replace('-', np.nan, regex=True)
    df[str_variables] = df[str_variables].replace('', np.nan, regex=True)
    df[str_variables] = df[str_variables].replace('0', np.nan, regex=True)
    df[str_variables] = df[str_variables].replace(0, np.nan, regex=True)
        
    # Drop columns that have all NaN or zeros values to strip the table of irrelevant or unmeasured variables
    df = df.dropna(axis=1, how='all') # For NaN
    df = df.loc[:, df.any()]          # For zeros
        
    # Since we had earlier converted numeric columns to numeric, we are left we ensuring that string and datetime columns 
    # are what they are expected to be.
    
    ## Convert the string datetime values to datetime format and replace every invalid parsing with NaT
    df[dt_variables] = df[dt_variables].apply(pd.to_datetime, errors='coerce')
        
    # Fill all invalid parsing with the date before them to handle any missing values
    df[dt_variables] = df[dt_variables].ffill()
    
    # Convert all the remaining string columns to string format.
    new_string_variables = [x for x in str_variables if x in df.columns]
    df[new_string_variables] = df[new_string_variables].applymap(str)
    resp.append(df)


# Create a function to handle duplicate rows and columns

def handle_duplicate(df, resp):
    # From the exploratory data analysis we observed that there are duplicate rows. 
    # So we need to deduplicate the rows or any duplicate column

    # Deduplicate the columns
    df = df.loc[:,~df.columns.duplicated()]

    # Deduplicate the rows; keep the first of the duplicates
    df = df.drop_duplicates()
    resp.append(df)



# Create a function to handle the outliers

def outlier_handler(df, resp):
    """
    A low-pass filter to detect and replace outliers in numerics variables.
    arguments:
        df - Pandas dataframe
    return:
        df: Pandas series. Series containing the column that has been adjusted for outliers
    """
    buffer = 100    
    series_name = df.name
    df = df.copy()
    df = pd.to_numeric(df)
    df = df.fillna(0)
    
    # Get the mode value of the series
    mode_df = df[df > 0] #Dropped the zero values to enable us get non-zero mode
    mode_value = mode_df.mode()
    
    # Convert the series to a DataFrame
    df = pd.DataFrame(df)
    
    # Detect the outliers
    # Create a mode column to help with comparison later
    df['Mode'] = mode_value.to_list()[0]
    df['Low_Pass_Filter_Outlier'] = ((df[series_name]) > (df['Mode']*10))
    
    #Replace the outlier values with the mode * (buffer * 0.5) 
    df[series_name] = np.where(df['Low_Pass_Filter_Outlier'] == True, df['Mode']*(buffer/2), df[series_name])
    resp.append(df[series_name])





# Load the Customers, Deliveries and Orders data
customers_ng = pd.read_csv("Nigeria Customers.csv")
customers_ke = pd.read_csv("Kenya Customers.csv")
deliveries_ng = pd.read_csv("Nigeria Deliveries.csv")
deliveries_ke = pd.read_csv("Kenya Deliveries.csv")
orders_ng = pd.read_csv("Nigeria Orders.csv")
orders_ke = pd.read_csv("Kenya Orders.csv")


# Wrap the above functions in a multithread function and get the various countries data

respCountry_dfN = []
tGetCdfN = threading.Thread(target=country_df, args=[customers_ng, orders_ng, deliveries_ng, respCountry_dfN]) 
tGetCdfN.start()
tGetCdfN.join()

# Get Nigeria data
nigeria_df = respCountry_dfN[0]

respCountry_dfK = []
tGetCdfK = threading.Thread(target=country_df, args=[customers_ke, orders_ke, deliveries_ke, respCountry_dfK]) 
tGetCdfK.start()
tGetCdfK.join()

# Get Kenya data
kenya_df = respCountry_dfK[0]


# Merge the countries DataFrames into one
full_result = pd.concat([kenya_df, nigeria_df], ignore_index=False, sort=False)

# Write the merged data to a csv file
full_result.to_csv("Merged_file.csv", header=True)


# Define the expected or assumed data types of the various columns based on the exploration of the data and column title
data_schema = { 'CUSTOMER ID': int, 
          'LAST USED PLATFORM': str, 
          'IS BLOCKED': int, 
          'CREATED AT': 'datetime',
          'LANGUAGE': str, 
          'OUTSTANDING AMOUNT': int,
          'LOYALTY POINTS': int,
          'NUMBER OF EMPLOYEES': int, 
          'UPLOAD RESTUARANT LOCATION': str, 
          'ORDER ID': int,
          'ORDER STATUS': str, 
          'CATEGORY NAME': str, 
          'SKU': str, 
          'CUSTOMIZATION GROUP': str,
          'CUSTOMIZATION OPTION': str, 
          'QUANTITY': int, 
          'UNIT PRICE': float, 
          'COST PRICE': float,
          'TOTAL COST PRICE': float, 
          'TOTAL PRICE': float, 
          'ORDER TOTAL': float, 
          'SUB TOTAL': float, 
          'TAX': str,
          'DELIVERY CHARGE': str , 
          'TIP': str, 
          'DISCOUNT': str, 
          'REMAINING BALANCE': int,
          'PAYMENT METHOD': str, 
          'ADDITIONAL CHARGE': str, 
          'TAXABLE AMOUNT': str,
          'TRANSACTION ID': int, 
          'CURRENCY SYMBOL': str, 
          'TRANSACTION STATUS': str, 
          'PROMO CODE': str,
          'MERCHANT ID': int,
          'STORE NAME': str,
          'PICKUP ADDRESS': str, 
          'DESCRIPTION': str,
          'DISTANCE (IN KM)': float, 
          'ORDER TIME': 'datetime', 
          'PICKUP TIME': 'datetime', 
          'DELIVERY TIME': 'datetime',
          'RATINGS': int, 
          'REVIEWS': str, 
          'MERCHANT EARNING': float, 
          'COMMISSION AMOUNT': float,
          'COMMISSION PAYOUT STATUS': str, 
          'ORDER PREPARATION TIME': int, 
          'DEBT AMOUNT': str,
          'REDEEMED LOYALTY POINTS': int, 
          'CONSUMED LOYALTY POINTS': int,
          'CANCELLATION REASON': str, 
          'FLAT DISCOUNT': float, 
          'CHECKOUT TEMPLATE NAME': str,
          'CHECKOUT TEMPLATE VALUE': int, 
          'TASK_ID': int, 
          'ORDER_ID': str, 
          'RELATIONSHIP': float,
          'TEAM_NAME': str, 
          'TASK_TYPE': str, 
          'NOTES': str, 
          'AGENT_ID': int, 
          'AGENT_NAME': str,
          'DISTANCE(M)': float, 
          'TOTAL_TIME_TAKEN(MIN)': float, 
          'PICK_UP_FROM': str, 
          'START_BEFORE': 'datetime',
          'COMPLETE_BEFORE': 'datetime', 
          'COMPLETION_TIME': 'datetime', 
          'TASK_STATUS': str, 
          'REF_IMAGES': str,
          'RATING': int, 
          'REVIEW': str, 
          'LATITUDE': float, 
          'LONGITUDE': float, 
          'TAGS': str, 
          'PROMO_APPLIED': float,
          'CUSTOM_TEMPLATE_ID': str, 
          'TASK_DETAILS_QTY': int, 
          'TASK_DETAILS_AMOUNT': str,
          'SPECIAL_INSTRUCTIONS': str, 
          'TIP:1': str,
          'DELIVERY_CHARGES': str, 
          'DISCOUNT:1': str,
          'SUBTOTAL': float, 
          'PAYMENT_TYPE': str,
          'TASK_CATEGORY': str, 
          'EARNING': float, 
          'PRICING': str
}

# Get the names of columns that are expected to have numeric values
numeric_variables = [x for x in data_schema if data_schema[x] in [int, float]]

# Get the names of columns that are expected to have string values
string_variables = [x for x in data_schema if data_schema[x] == str]

## Get the datetime columns names
datatime_variables = [x for x in data_schema if data_schema[x] == 'datetime']


# Clean the numeric, string and datetime variables

respHNSD = []
tGetCNSD = threading.Thread(target=handle_numeric_string_date_types, args=[full_result, numeric_variables, string_variables, datatime_variables, respHNSD]) 
tGetCNSD.start()
tGetCNSD.join()

full_result = respHNSD[0]

# Handle duplicate rows and columns

respHDRC = []
tGetCRC = threading.Thread(target=handle_duplicate, args=[full_result, respHDRC]) 
tGetCRC.start()
tGetCRC.join()

full_result = respHDRC[0]

# Handle outliers

variable_with_outlier = 'NUMBER OF EMPLOYEES'

respHOl = []
tGetCOL = threading.Thread(target=outlier_handler, args=[full_result[variable_with_outlier], respHOl]) 
tGetCOL.start()
tGetCOL.join()




cleaned_df = full_result.copy()

cleaned_df[variable_with_outlier] = respHOl[0]



#---------------------------------------------------------------------------------

#Use relevant ML Model(s) to predict Customer Retention

#---------------------------------------------------------------------------------


# Load the features data from the dataset
new_dataset = cleaned_df.copy()

# Cast the numeric variables to numeric just to be sure it works with the pandas groupby function
number_var = ['DISTANCE (IN KM)','TOTAL_TIME_TAKEN(MIN)', 'RATING']

new_dataset[number_var] = new_dataset[number_var].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Group the data frame by 'CUSTOMER ID' and extract a number of stats from each group
new_dataset = new_dataset.groupby('CUSTOMER ID').agg(
                {
                    'IS BLOCKED':'first',
                    'LOYALTY POINTS':'first',
                    'NUMBER OF EMPLOYEES':'first',
                    'ORDER ID': "count",
                    'CATEGORY NAME': lambda x: x.mode(), # Most prevalent category name
                    'QUANTITY': 'mean',
                    'PAYMENT METHOD': lambda x: x.mode(), # Most prevalent payment method
                    'DISTANCE (IN KM)': 'mean', 
                    'REDEEMED LOYALTY POINTS': sum,
                    'CONSUMED LOYALTY POINTS': sum,
                    'AGENT_NAME': lambda x: x.mode(), # Most prevalent agent name
                    'DISTANCE(M)': 'mean',
                    'TOTAL_TIME_TAKEN(MIN)' : 'mean',
                    'PICK_UP_FROM': lambda x: x.mode(), # Most prevalent PICK_UP_FROM
                    'TASK_STATUS': lambda x: x.mode(), # Most prevalent TASK_STATUS
                    'RATING': 'mean',
                    'SUBTOTAL': 'mean'
                    
                }
            )


# Drop rows with 4 or more cell with either NaN, NaT or None
new_dataset_2 = new_dataset.copy().dropna(thresh=17)


# Data Preprocessing

# Define the nominal categorical variables
nom_cat = ['CATEGORY NAME', 'PAYMENT METHOD', 'AGENT_NAME', 'PICK_UP_FROM', 'TASK_STATUS', 'RATING']

#We adopt the pd.get_dummies to transform the nominal categorical features, while inserting the drop_first=True arguement
#to handle the multicollinearity that may result
encoded_nomcat = pd.get_dummies(new_dataset_2[nom_cat].applymap(str), drop_first=True)

#Drop the former norminal categorical features from new_dataset and concatenate the encoded norminal categorical features
new_dataset_2 = new_dataset_2.drop(nom_cat, axis=1, inplace=False)
new_dataset_2 = pd.concat([new_dataset_2, encoded_nomcat], axis=1)

# Separate the predicted/target feature from the predicting features
X =  new_dataset_2.drop(['IS BLOCKED'], axis=1) # Explanatory variables
y = new_dataset_2['IS BLOCKED'] # Explained variable: if y = 1, the customer is churned; if y = 0, the customer is retained

# Split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


# Machine Learning Model Training and Evaluation

# We will use RandomForestClassifier ML model

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500, random_state=0) 
classifier.fit(X_train, y_train) 
predictions = classifier.predict(X_test)


# Evaluate model
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test,predictions )) 
print("Customer Retention Predict Model Accuracy is ", round(accuracy_score(y_test, predictions )*100,3),"%")




#---------------------------------------------------------------------------------

#Use relevant ML Model(s) to Classify Customers

#---------------------------------------------------------------------------------


# Load data

# Define the relevant features
class_features = ['CUSTOMER ID', 'SUBTOTAL', 'ORDER ID', 'ORDER TIME']

classify_dataset = cleaned_df.copy()
classify_dataset = classify_dataset[class_features]

# Drop rows with zero 'ORDER ID'
classify_dataset = classify_dataset[classify_dataset['ORDER ID'] > 0]

# Drop rows with zero 'SUBTOTAL'
classify_dataset = classify_dataset[classify_dataset['SUBTOTAL'] > 0]

# Be sure the data are in their expected types
classify_dataset['CUSTOMER ID'] = pd.to_numeric(classify_dataset['CUSTOMER ID'], downcast='unsigned', errors='coerce')
classify_dataset['SUBTOTAL'] = pd.to_numeric(classify_dataset['SUBTOTAL'], downcast='unsigned', errors='coerce')
classify_dataset['ORDER ID'] = pd.to_numeric(classify_dataset['ORDER ID'], downcast='unsigned', errors='coerce')
classify_dataset['ORDER TIME'] = classify_dataset['ORDER TIME'].apply(pd.to_datetime, errors='coerce')


# Features engineering

# Define an end date for use in computing the recency
end_date = max(classify_dataset['ORDER TIME']) + dt.timedelta(days=1)

# Calculate some RFM variables - that is, recency, frequency, and monetary value of each customer.

rfm_data = classify_dataset.groupby('CUSTOMER ID').agg(
                    recency=('ORDER TIME', lambda x: (end_date - x.max()).days),
                    frequency=('ORDER ID', 'count'),
                    monetary=('SUBTOTAL', 'sum')
                    ).reset_index()


# Preprocess the data

# Create a function to preprocess the data to normalize the skewed RFM variables
def preprocess_rfm(df, resp):
    """Preprocess data for KMeans clustering"""
    df_log = np.log1p(df)
    scaler = StandardScaler()
    scaler.fit(df_log)
    norm = scaler.transform(df_log)
    resp.append(norm)



# Find the optimum number of cluster - k


# Create a function to get the optimum number of cluster - k

def find_k(df, increment, decrement, resp):
    """Find the optimum k clusters"""
    
    respPrepDt = []
    tGetPdt = threading.Thread(target=preprocess_rfm, args=[df, respPrepDt]) 
    tGetPdt.start()
    tGetPdt.join()
    
    norm = respPrepDt[0]
    sse = {}
    
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, random_state=1)
        kmeans.fit(norm)
        sse[k] = kmeans.inertia_
    
    kn = KneeLocator(
                 x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing'
                 )
    k = kn.knee + increment - decrement
    resp.append(k)


# Running the KMeans clustering model

# Create a function to run the KMeans clustering model

def apply_kmeans(df, resp, increment, decrement):
    """Run KMeans clustering, including the preprocessing of the data
    and the automatic selection of the optimum k. 
    """
    respPrepDt = []
    tGetPdt = threading.Thread(target=preprocess_rfm, args=[df, respPrepDt]) 
    tGetPdt.start()
    tGetPdt.join()
    
    norm = respPrepDt[0]
    
    respFnK = []
    tGetFK = threading.Thread(target=find_k, args=[df, increment, decrement, respFnK]) 
    tGetFK.start()
    tGetFK.join()
    
    k = respFnK[0] #find_k(df, increment, decrement)
    kmeans = KMeans(n_clusters=k, 
                    random_state=1)
    kmeans.fit(norm)
    resp.append(df.assign(cluster=kmeans.labels_))


# Cluster the customer

respApKm = []
tGetAK = threading.Thread(target=apply_kmeans, args=[rfm_data, respApKm, 0, 2]) 
tGetAK.start()
tGetAK.join()

# clusters_decrement = apply_kmeans(rfm_data, decrement=2)
clusters_decrement = respApKm[0]
clusters_decrement.groupby('cluster').agg(
    recency=('recency','mean'),
    frequency=('frequency','mean'),
    monetary=('monetary','mean'),
    cluster_size=('CUSTOMER ID','count')
).round(1).sort_values(by='recency')


classes = {3:'bronze', 0:'silver',2:'gold',1:'platinum'}
clusters_decrement['class'] = clusters_decrement['cluster'].map(classes)
print("------------------------Customers Classification----------------------")
print(clusters_decrement)


#---------------------------------------------------------------------------------

#Use relevant ML Model(s) to predict Product Recommendations

#---------------------------------------------------------------------------------


#Create a function to get top five scoring recs that are not the original product

def get_recommendations(df, item, resp):
    # Create an item matrix by pivoting our dataset
    pivot_df = pd.pivot_table(df.copy(), index = 'ORDER ID', columns = 'SKU',values = 'QUANTITY',aggfunc = 'sum')
    pivot_df.reset_index(inplace=True)

    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.drop('ORDER ID', axis=1)
    
    
    # Transform pivot table into a co-occurrence matrix
    co_matrix = pivot_df.T.dot(pivot_df)
    np.fill_diagonal(co_matrix.values, 0)

    #  Transform the co-occurrence matrix into a matrix of cosine similarities
    cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
    cos_score_df.index = co_matrix.index
    cos_score_df.columns = np.array(co_matrix.index)
    
    product_recs = cos_score_df[cos_score_df.index!=item][item].sort_values(ascending = False)[0:5].index
    resp.append(product_recs.tolist())


# Load Data

df_baskets = cleaned_df.copy()

df_baskets = df_baskets[['ORDER ID', 'SKU', 'QUANTITY' ]]

# Drop rows with 'None' 'SKU'
df_baskets = df_baskets[df_baskets['SKU'] != "None"]

# Drop rows with 'nan' 'SKU'
df_baskets = df_baskets[df_baskets['SKU'] != "nan"]

# # Be sure the data are in their expected types
df_baskets['ORDER ID'] = pd.to_numeric(df_baskets['ORDER ID'], downcast='unsigned', errors='coerce')
df_baskets['QUANTITY'] = pd.to_numeric(df_baskets['QUANTITY'], downcast='unsigned', errors='coerce')
df_baskets['SKU'] = df_baskets['SKU'].astype(str)


respGRec = []
tGetRec = threading.Thread(target=get_recommendations, args=[df_baskets, 'KKGRO239', respGRec]) 
tGetRec.start()
tGetRec.join()


print("Recommendations: ", respGRec[0])



#---------------------------------------------------------------------------------

#Use relevant ML Model(s) Optimize Revenue

#---------------------------------------------------------------------------------


"""
Factors that influence Revenue include:
    "Total Quantity"
    "Number of customers"
    "Gross margin%"
    "Average Order Frequency"
    "Average Order Value"
    """

# Load, engineer and prepare the required data

rev_data = cleaned_df.copy()
rev_data["ORDER TIME"] = rev_data["ORDER TIME"].apply(pd.to_datetime, errors='coerce')
rev_data = rev_data.set_index("ORDER TIME")#.resample('1D')
rev_data = rev_data.resample('1D').agg(
                    {
                        'QUANTITY': 'sum',
                        'CUSTOMER ID': 'nunique',
                        'TOTAL PRICE': 'sum',
                        'TOTAL COST PRICE': 'sum',
                        'ORDER ID': 'nunique'
                    }
                    )
new_columns = ['total_quantity', 'number_of_customers', 'sales_prics', 'cost_prics', 'unique_orders'  ]
rev_data.columns = new_columns
index = rev_data.index
index.name = "Date"

rev_data["Gross_margin%"] = ((rev_data["sales_prics"] - rev_data["cost_prics"])/rev_data["sales_prics"])*100.0
rev_data["Average_Order_Frequency"] = rev_data["unique_orders"]/rev_data["number_of_customers"] 
rev_data["Average_Order_Value"] = rev_data["sales_prics"]/rev_data["unique_orders"] 
rev_data["net_revenue"] = (rev_data["number_of_customers"] * rev_data["Average_Order_Frequency"] * 
                           rev_data["Average_Order_Value"] * rev_data["Gross_margin%"])


# Predict optimum revenue

## Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Separate the predicted/target feature from the predicting features
X =  rev_data.drop(['net_revenue'], axis=1) # Features
y = rev_data['net_revenue'] # Label

# Split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)

from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf_rev = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth=8,)
# Train the model on training data
rf_rev.fit(X_train, y_train);

# Make prediction for the test data
rev_predictions = rf_rev.predict(X_test)


# Performance metrics
errors = abs(rev_predictions - y_test)
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

