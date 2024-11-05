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

# Display an updated count of records, now that there are no records with a null value in the rating column
df['show_id'].count()

# COMMAND ----------

# Check first year for each type
print("First Year - TV Show:", df['release_year'][df['type'] == 'TV Show'].min())
print("First Year - Movie:", df['release_year'][df['type'] == 'Movie'].min())

# Check last year for each type
print("Last Year - TV Show:", df['release_year'][df['type'] == 'TV Show'].max())
print("First Year - Movie:", df['release_year'][df['type'] == 'Movie'].max())



# COMMAND ----------

# Remove rows where the release date is older than 1966 and newer than 2021
df = df[(df['release_year'] >= 1990) & (df['release_year'] < 2021)]

df['release_year'].min(),df['release_year'].max()

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
# MAGIC   <li>Is there a general trend or relationship of TV Show season count over time</li>
# MAGIC   <li>What categories are Reality TV movies and shows most often associated with</li>
# MAGIC   <li>Which rating is most commonly associated with Crime movies</li>
# MAGIC </ol>

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. How many movies & tv shows were released each year?

# COMMAND ----------

yearly_releases = pd.DataFrame(df.groupby(['release_year', 'type'])['show_id'].count()).reset_index()
yearly_releases = yearly_releases.pivot(index='release_year', columns='type', values='show_id')

# COMMAND ----------

yearly_releases.head()

# COMMAND ----------

yearly_releases.plot()

# COMMAND ----------

new_releases = yearly_releases[yearly_releases.index > 2010]

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

# Show the number of TV shows since 2015 that has more than 3 seasons
season_3_count = len(new_tv_shows_df[new_tv_shows_df['season_number'] > 3])
print(f"Number of TV Shows since 2015 with more than three seasons: {season_3_count}")

# COMMAND ----------

# Show the top 10 longest running TV shows by season count
long_running_tv_shows_df = new_tv_shows_df[new_tv_shows_df['season_number'] > 3]
long_running_tv_shows_df[['title', 'season_number']].sort_values(by='season_number', ascending=False).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Is there a general trend or relationship of average movie length over time

# COMMAND ----------

df.head(3)

# COMMAND ----------

df[['duration_mins', '_']] = df['duration'].str.split(" ", expand=True)

# COMMAND ----------

duration_df = df[(~df['duration_mins'].isna()) & (df['type'] == "Movie")]
duration_df['duration_mins'] = duration_df['duration_mins'].astype(int)

# COMMAND ----------

duration_df[duration_df['duration_mins'] < 50]

# COMMAND ----------

duration_df.dtypes

# COMMAND ----------

duration_df.plot(x="release_year", y="duration_mins", kind="scatter")

# COMMAND ----------

duration_df.groupby("release_year").mean().plot()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Is there a general trend or relationship of TV Show season count over time 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. What categories are Reality TV movies and shows most often associated with

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Which rating is most commonly associated with Crime movies

# COMMAND ----------


