import json
from flask import Flask, jsonify, request
from databricks import sql

app = Flask(__name__)

# Load credentials
with open('creds.json', 'r') as file:
    credentials = json.loads(file.read())

connection =  sql.connect(server_hostname = credentials['HOSTNAME'],
                 http_path       = credentials['HTTP_PATH'],
                 access_token    = credentials['TOKEN'])


# List tables
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

      # response = { "data":[]}
      # for row in result:

      #   data= {
      #     "table_name": row.TABLE_NAME,
      #     "table_type": row.TABLE_TYPE,
      #     "parent_schema": row.TABLE_SCHEM,
      #     "parent_catalog": row.TABLE_CAT   
      #   }

      #   response["data"].append(data)
        
    cursor.close()
    return jsonify(result)


# List columns in table
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


# Get data in table
@app.route('/get-data', methods=['GET'])
def get_data():
  """Gets a specified number of rows from a selected table. If no limit is specified, it defaults to 5"""

  catalog = request.args.get('catalog')
  schema = request.args.get('schema')
  table = request.args.get('table')
  limit = request.args.get('limit') or 5

  if not catalog or not schema or not table:
    return 'Please specify a catalog, schema, and table. Ie. ?catalog=samples&schema=nyctaxi&table=trips'
  
  else:
    with connection.cursor() as cursor:
      cursor.execute(f"SELECT * FROM {catalog}.{schema}.{table} LIMIT {limit}")
      result = cursor.fetchall()

      return result

if __name__ == '__main__':
   app.run(port=5000)
