# Databricks notebook source
# MAGIC %md
# MAGIC # Data Analysis: Netflix w/ pandas

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Data prep & cleaning

# COMMAND ----------

# DBTITLE 1,Imports & data loading
# Imports & data loading
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  

df = pd.read_csv("/Volumes/lgeorge/demos/demo_vol/netflix_titles.csv")

# COMMAND ----------

# Show first few rows
df.head()

# COMMAND ----------

# Get number of rows from id column
df['show_id'].count()

# COMMAND ----------

# Check to see if any of the columns have nulls
for i in df.columns:
  print(i, '-', df[i].isna().sum())

# COMMAND ----------

# Remove rows that have an Null in the Rating column
df = df[~df['rating'].isna()]

# COMMAND ----------

df['show_id'].count()

# COMMAND ----------

# Check first year for each type
print(df['release_year'][df['type'] == 'TV Show'].min())
print(df['release_year'][df['type'] == 'Movie'].min())


# COMMAND ----------

# Remove rows where the release date is older than 1966
df = df[df['release_year'] > 1966]

df['release_year'].min()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Key Questions:
# MAGIC <ol>
# MAGIC   <li>How many movies and tv shows were released each year? Is there a trend in the releases?</li>
# MAGIC   <li>For tv shows, how many shows released since 2015 have more than 3 seasons</li>
# MAGIC   <li>Is there a general trend or relationship of movie length over time</li>
# MAGIC   <li>What categories are Reality TV movies and shows most often associated with</li>
# MAGIC   <li>Which rating is most commonly associated with Crime movies</li>
# MAGIC </ol>

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. How many movies & tv shows were released each year?

# COMMAND ----------

yearly_releases = pd.DataFrame(df.groupby(['release_year', 'type'])['show_id'].count()).reset_index()
#yearly_releases.set_index('release_year', inplace=True)
yearly_releases = yearly_releases.pivot(index='release_year', columns='type', values='show_id')

# COMMAND ----------

yearly_releases.head()

# COMMAND ----------

yearly_releases.plot()

# COMMAND ----------

new_releases = yearly_releases[(yearly_releases.index > 2010) & (yearly_releases.index < 2021)]

# COMMAND ----------

new_releases.plot()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. For tv shows, how many shows released since 2015 have more than 3 seasons

# COMMAND ----------

new_tv_shows_df = df[(df.type == 'TV Show') & (df.release_year >= 2015)]
new_tv_shows_df[['season_number','suffix']] = new_tv_shows_df['duration'].str.extract('(\w+)\s(\w+)', expand=True)
new_tv_shows_df['season_number'] = pd.to_numeric(new_tv_shows_df['season_number'])

# COMMAND ----------

len(new_tv_shows_df[new_tv_shows_df['season_number'] > 3])

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Is there a general trend or relationship of average movie length over time

# COMMAND ----------

df.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. What categories are Reality TV movies and shows most often associated with

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Which rating is most commonly associated with Crime movies

# COMMAND ----------


