# Product Review Summarizer

A Streamlit application that summarizes product reviews from a Databricks Delta table using Unity Catalog and AI-powered summarization.

## Features

- **Product Review Summarization**: Enter a product ID to get a comprehensive summary of all reviews
- **Databricks Integration**: Reads from Delta tables in Unity Catalog
- **AI-Powered Summaries**: Uses Databricks `ai_query` function with SQL to generate intelligent summaries
- **Key Areas Highlighted**: Summaries include:
  - Overall sentiment
  - Key strengths
  - Key weaknesses/concerns
  - Common themes
  - Product features
  - Recommendations

## Prerequisites

- Python 3.8 or higher
- Databricks workspace with:
  - Unity Catalog enabled
  - Delta table with product reviews (columns: `product_id`, `review_text`, `review_id`)
  - Access to `ai_query` function (requires Databricks SQL Pro or Serverless)
  - SQL warehouse endpoint

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Option 1: Streamlit Secrets (Recommended)

Create a `.streamlit/secrets.toml` file in your project directory:

```toml
[databricks]
server_hostname = "your-workspace.cloud.databricks.com"
http_path = "/sql/1.0/warehouses/your-warehouse-id"
access_token = "your-personal-access-token"
```

### Option 2: Environment Variables

Set the following environment variables:

```bash
export DATABRICKS_SERVER_HOSTNAME="your-workspace.cloud.databricks.com"
export DATABRICKS_HTTP_PATH="/sql/1.0/warehouses/your-warehouse-id"
export DATABRICKS_ACCESS_TOKEN="your-personal-access-token"
```

### Getting Databricks Credentials

1. **Server Hostname**: Your Databricks workspace URL (e.g., `adb-1234567890123456.7.azuredatabricks.net`)

2. **HTTP Path**: 
   - Go to your Databricks workspace
   - Navigate to SQL Warehouses
   - Click on your warehouse
   - Copy the "Serverless HTTP path" or "HTTP path"

3. **Access Token**:
   - Go to User Settings → Access Tokens
   - Generate a new token
   - Copy and save it securely

## Data Schema

Your Delta table should have the following structure:

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Unique product identifier |
| `review_text` | string | The review text content |
| `review_id` | string/int | Unique review identifier (optional but recommended) |

Example table creation:

```sql
CREATE TABLE IF NOT EXISTS main.default.reviews (
    product_id STRING,
    review_text STRING,
    review_id STRING
) USING DELTA;
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Configure your table settings in app.py:
   - Catalog (default: `users`)
   - Schema (default: `landan_george`)
   - Table name (default: `reviews`)

3. Enter a product ID in the main input field

4. Click "Generate Summary" to:
   - Fetch all reviews for the product
   - Generate an AI-powered summary
   - Display the summary with key insights

## How It Works

1. **Data Retrieval**: The app queries the Delta table to fetch all reviews for the specified product ID
2. **AI Summarization**: Uses Databricks `ai_query` function with a SQL query to generate a comprehensive summary
3. **Display**: Shows the summary along with the option to view raw reviews

## Troubleshooting

### Connection Issues
- Verify your Databricks credentials are correct
- Ensure your SQL warehouse is running
- Check that your access token has not expired

### No Reviews Found
- Verify the product ID exists in your table
- Check the table name, catalog, and schema in the sidebar
- Ensure the table has the correct column names

### AI Query Errors
- Ensure your Databricks workspace has access to the `ai_query` function
- Check that you're using a supported model (default: `databricks-claude-opus-4-5`)
- Verify your workspace has AI/ML capabilities enabled

## License

This project is provided as-is for demonstration purposes.
