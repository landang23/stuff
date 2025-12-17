# Databricks notebook source
%pip install Lifetimes

# COMMAND ---------- 

import pandas as pd
import lifetimes
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_probability_alive_matrix, plot_frequency_recency_matrix
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases, plot_period_transactions,plot_history_alive
from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

# DBTITLE 1,Load Segmentation Data
# Get catalog and schema from job parameters
catalog_name = (dbutils.widgets.get("catalog_name") 
                if "catalog_name" in dbutils.widgets.getAll() 
                else "dev_customer_segmentation")
schema_name = (dbutils.widgets.get("schema_name") 
               if "schema_name" in dbutils.widgets.getAll() 
               else "segmentation")

# COMMAND ----------

# Load individual customer segments 
df = spark.table(f"{catalog_name}.{schema_name}.transactions").toPandas()


# COMMAND ----------

today_date = dt.datetime(2025, 9, 30)

df = df.groupby('customer_id').agg({'total_amount':'sum',
                              'transaction_date': [lambda date: (date.max()-date.min()).days,
                                              lambda date: (today_date.date() - date.min()).days],
                              'transaction_id': lambda Invoice: Invoice.nunique()})

# COMMAND ----------

df.columns.droplevel(0)
df.columns = ['monetary', 'recency', 'T', 'frequency']


# COMMAND ----------

df['monetary'] = df['monetary'] / df['frequency'] #average spend per visit
df = df[df['frequency'] > 1] #making sure the customer has purchased with us atleast once
df['recency'] = df['recency'] / 7  #converting the recency in weeks
df['T'] = df['T'] / 7 #converting the total timespan in weeks

# COMMAND ----------

bgf = BetaGeoFitter()
bgf.fit(df['frequency'], df['recency'], df['T'])
bgf.summary

# COMMAND ----------

df = df[df['monetary'] > 0]

# COMMAND ----------

plot_period_transactions(bgf)

# COMMAND ----------

df['expected_purch_6month'] = bgf.predict(4*6, df['frequency'], df['recency'], df['T'])


# COMMAND ----------

df.sort_values(by='expected_purch_6month', ascending= False).head()


# COMMAND ----------

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(df['frequency'], df['monetary'])
ggf.summary


# COMMAND ----------

ggf.conditional_expected_average_profit(df['frequency'], df['monetary']).sort_values(ascending=False).head(10)

# COMMAND ----------

df['expected_average_profit'] = ggf.conditional_expected_average_profit(df['frequency'], df['monetary'])
cltv = ggf.customer_lifetime_value(bgf, df['frequency'], df['recency'], df['T'], df['monetary'], time=6, freq='W')

# COMMAND ----------

cltv = cltv.reset_index()

# COMMAND ----------

cltv_final = df.merge(cltv, on='customer_id', how='left')

# COMMAND ----------

cltv_final.sort_values(by='clv', ascending=False).head()

# COMMAND ----------

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[['clv']])
cltv_final['scaled_cltv'] = scaler.transform(cltv_final[['clv']])
cltv_final.sort_values(by='scaled_cltv', ascending=False).head()

# COMMAND ----------

cltv_final.sort_values(by='scaled_cltv', ascending=False)

# COMMAND ----------

cltv_final_s = spark.createDataFrame(cltv_final)
cltv_final_s.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.customer_cltv")

# COMMAND ----------

