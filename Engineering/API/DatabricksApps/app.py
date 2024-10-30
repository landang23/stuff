import pandas as pd
from flask import Flask
import logging
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog


# Ensure environment variable is set correctly
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

w = WorkspaceClient()


log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

flask_app = Flask(__name__)

@flask_app.route('/')
def hello_world():
    data = spark.sql("SELECT * FROM samples.nyctaxi.trips")
    data_df = data.toPandas()
    return f'<h1>Hello, World!</h1> {data_df.to_html(index=False)}'

if __name__ == '__main__':
    flask_app.run(debug=True)
