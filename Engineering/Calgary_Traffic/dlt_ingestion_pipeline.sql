-- Databricks notebook source
-- Bronze
CREATE OR REFRESH STREAMING TABLE incidents_bronze
COMMENT "Bronze layer traffic incidents data coming from /Volumes/lgeorge/building/yyc_traffic/incidents/. Data is in CSV format"
AS
SELECT
*
FROM STREAM read_files("/Volumes/lgeorge/building/yyc_traffic/incidents/", format=>'csv');

-- COMMAND ----------

-- Silver
CREATE OR REFRESH STREAMING TABLE incidents_clean (
  CONSTRAINT `No Null Values` EXPECT (incident_info != 'NO TRAFFIC INCIDENTS') ON VIOLATION DROP ROW
)
COMMENT "Removed any records with NULL values and parsed out the file_path value into a new column"
AS
SELECT
  incident_info,
  description,
  start_dt,
  modified_dt,
  quadrant,
  count,
  longitude,
  latitude,
  YEAR(start_dt) year,
  MONTH(start_dt) month,
  DAY(start_dt) day,
  DAYOFWEEK(start_dt) dow,
  HOUR(start_dt) hour,
  `_rescued_data`:`_file_path`
FROM STREAM(LIVE.incidents_bronze)

-- COMMAND ----------

CREATE OR REFRESH STREAMING TABLE incidents_gold
COMMENT "Refined incident data containing description, location, and details around date and time. Duplicates are removed"
SELECT DISTINCT
  incident_info,
  description,
  quadrant,
  longitude,
  latitude,
  year,
  month,
  day,
  dow,
  CASE
    WHEN hour BETWEEN 6 AND 9 THEN 1
    WHEN hour BETWEEN 16 AND 18 THEN 1
    ELSE 0
  END AS is_rush_hour
FROM STREAM(LIVE.incidents_clean)
