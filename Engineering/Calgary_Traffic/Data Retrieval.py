# Databricks notebook source
import pandas as pd
from datetime import datetime

# COMMAND ----------

now = str(datetime.now()).replace(".",":").replace(" ", "_")

# COMMAND ----------

incidents = pd.read_csv("https://data.calgary.ca/resource/4jah-h97u.csv").to_csv(f"/Volumes/lgeorge/building/yyc_traffic/incidents/{now}.csv")

# COMMAND ----------

travel_times = pd.read_json("https://data.calgary.ca/resource/aeb8-fh2w.json").to_json(f"/Volumes/lgeorge/building/yyc_traffic/traffic/{now}.json")
