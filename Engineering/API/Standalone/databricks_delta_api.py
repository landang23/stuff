# Imports
from flask import Flask, jsonify, request
from databricks import sql
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Config setup
load_dotenv()

connection =  sql.connect(
                server_hostname = os.getenv("SERVER_HOSTNAME"),
                http_path       = os.getenv("HTTP_PATH"),
                access_token    = os.getenv("API_KEY")
                )


# List tables #
@app.route('/list-tables', methods=['GET'])
def list_tables():
  """List tables within a specified catalog and schema"""

  catalog = request.args.get('catalog')
  schema = request.args.get('schema')

  if not catalog or not schema:
     return 'Please specify a catalog and schema. Ie. ?catalog=samples&schema=nyctaxi'
  else:
    with connection.cursor() as cursor:
      
      cursor.tables(catalog_name=catalog, schema_name=schema)
      result = cursor.fetchall()

    cursor.close()
    return jsonify(result)


# List columns in table #
@app.route('/list-columns', methods=['GET'])
def list_columns():
  """List columns within a specified catalog, schema, and table"""

  catalog = request.args.get('catalog')
  schema = request.args.get('schema')
  table = request.args.get('table')

  if not catalog or not schema or not table:
    return 'Please specify a catalog, schema, and table. Ie. ?catalog=samples&schema=nyctaxi&table=trips'
  
  else:  
    with connection.cursor() as cursor:
      cursor.columns(catalog_name=catalog, schema_name=schema, table_name=table)
      result = cursor.fetchall()

      response = {
        "catalog_name": catalog,
        "schema_name": schema,
        "table_name": table,
        "data":[]
      }

      for row in result:
        data = {
          "column_name": row.COLUMN_NAME,
          "data_type": row.TYPE_NAME
        }

        response['data'].append(data)

    cursor.close()
    return jsonify(response)


# Get data in table #
@app.route('/get-data', methods=['GET'])
def get_data():
  """Gets a specified number of rows from a selected table. If no limit is specified, it defaults to 5"""

  catalog = request.args.get('catalog')
  schema = request.args.get('schema')
  table = request.args.get('table')
  limit = request.args.get('limit') or 5

  if not catalog or not schema or not table:
    return 'Please specify a catalog, schema, and table. Ie. ?catalog=samples&schema=nyctaxi&table=trips&limit=5'
  elif int(limit) > 100:
    return 'Please specify a limit less than 100 rows and try again.'
  else:
    with connection.cursor() as cursor:
      cursor.execute(f"SELECT * FROM {catalog}.{schema}.{table} LIMIT {limit}")
      result = cursor.fetchall()

      return result
    
# API Homepage
@app.route('/', methods=['GET'])
def home():

    text = """
    <h3>API Endpoint Overview:</h3>
    <ul>
        <li>/list-tables - List tables inside of a catalog and schema. Ex: ?catalog=samples&schema=nyctaxi</li>
        <li>/list-columns - List columns within a specified catalog, schema, and table. Ex: ?catalog=samples&schema=nyctaxi&table=trips</li>
        <li>/get-data - Gets a specified number of rows from a selected table. If no limit is specified, it defaults to 5. Ex: ?catalog=samples&schema=nyctaxi&table=trips&limit=5</li>
    </ul>
    """

    return text


# Start App
if __name__ == '__main__':
   app.run(port=8080, debug=True)
