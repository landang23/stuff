import streamlit as st
from databricks import sql
from databricks.sql.client import Connection
import os
from typing import List, Dict, Optional
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Product Review Summarizer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'connection' not in st.session_state:
    st.session_state.connection = None


@st.cache_resource
def get_databricks_connection() -> Optional[Connection]:
    """
    Create and cache Databricks SQL connection.
    Reads connection parameters from environment variables or Streamlit secrets.
    """
    try:
        # Try to get from Streamlit secrets first
        if 'databricks' in st.secrets:
            server_hostname = st.secrets['databricks']['server_hostname']
            http_path = st.secrets['databricks']['http_path']
            access_token = st.secrets['databricks']['access_token']
        else:
            # Fall back to environment variables
            server_hostname = os.getenv('DATABRICKS_SERVER_HOSTNAME')
            http_path = os.getenv('DATABRICKS_HTTP_PATH')
            access_token = os.getenv('DATABRICKS_ACCESS_TOKEN')
        
        if not all([server_hostname, http_path, access_token]):
            return None
        
        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token
        )
        return connection
    except Exception as e:
        st.error(f"Error connecting to Databricks: {str(e)}")
        return None


def fetch_reviews_for_product(connection: Connection, catalog: str, schema: str, table: str, product_id: str) -> pd.DataFrame:
    """
    Fetch all reviews for a given product ID from the Delta table.
    
    Args:
        connection: Databricks SQL connection
        catalog: Unity Catalog catalog name
        schema: Schema name
        table: Table name
        product_id: Product ID to filter by
    
    Returns:
        DataFrame with reviews
    """
    # Escape single quotes to prevent SQL injection
    product_id_escaped = product_id.replace("'", "''")
    
    query = f"""
    SELECT 
        product_id,
        review_text,
        review_id
    FROM {catalog}.{schema}.{table}
    WHERE product_id = '{product_id_escaped}'
    ORDER BY review_id
    """
    
    try:
        df = pd.read_sql(query, connection)
        return df
    except Exception as e:
        st.error(f"Error fetching reviews: {str(e)}")
        return pd.DataFrame()


def generate_summary_with_ai(connection: Connection, catalog: str, schema: str, table: str, product_id: str) -> str:
    """
    Use Databricks ai_query function to generate a summary of all reviews for a product.
    This function aggregates reviews in SQL and then uses ai_query to generate the summary.
    
    Args:
        connection: Databricks SQL connection
        catalog: Unity Catalog catalog name
        schema: Schema name
        table: Table name
        product_id: Product ID to summarize
    
    Returns:
        Summary text
    """
    # First, verify reviews exist
    reviews_df = fetch_reviews_for_product(connection, catalog, schema, table, product_id)
    
    if reviews_df.empty:
        return "No reviews found for this product ID."
    
    # Use SQL to aggregate reviews and pass to ai_query
    # This approach is more efficient and handles large datasets better
    product_id_escaped = product_id.replace("'", "''")
    
    ai_query_sql = f"""
    WITH numbered_reviews AS (
        SELECT 
            review_text,
            ROW_NUMBER() OVER (ORDER BY COALESCE(review_id, '0')) AS review_num
        FROM {catalog}.{schema}.{table}
        WHERE product_id = '{product_id_escaped}'
    ),
    aggregated_reviews AS (
        SELECT 
            CONCAT_WS(
                '\\n\\n---\\n\\n',
                COLLECT_LIST(
                    CONCAT('Review ', CAST(review_num AS STRING), ': ', review_text)
                )
            ) AS all_reviews_text
        FROM numbered_reviews
    )
    SELECT ai_query(
        'databricks-claude-opus-4-5',
        CONCAT(
            'You are a product review analyst. Summarize the following product reviews, highlighting key areas such as:\\n',
            '- Overall sentiment (positive, negative, neutral)\\n',
            '- Key strengths mentioned\\n',
            '- Key weaknesses or concerns\\n',
            '- Common themes across reviews\\n',
            '- Product features frequently mentioned\\n',
            '- Recommendations or suggestions\\n\\n',
            'Product Reviews:\\n',
            all_reviews_text,
            '\\n\\nProvide a comprehensive summary that captures the essence of all reviews.'
        )
    ) AS summary
    FROM aggregated_reviews
    """
    
    try:
        # Execute the ai_query
        cursor = connection.cursor()
        cursor.execute(ai_query_sql)
        result = cursor.fetchone()
        cursor.close()
        
        if result and len(result) > 0:
            return result[0]
        else:
            return "Error: Could not generate summary."
    except Exception as e:
        # Fallback: fetch reviews in Python and build prompt
        try:
            all_reviews = reviews_df['review_text'].tolist()
            reviews_text = "\n\n---\n\n".join([f"Review {i+1}: {review}" for i, review in enumerate(all_reviews)])
            
            # Truncate if too long (SQL string limits)
            max_length = 15000
            if len(reviews_text) > max_length:
                reviews_text = reviews_text[:max_length] + "... (truncated due to length)"
            
            # Escape single quotes for SQL
            reviews_text_escaped = reviews_text.replace("'", "''")
            
            prompt = f"""You are a product review analyst. Summarize the following product reviews, highlighting key areas such as:
- Overall sentiment (positive, negative, neutral)
- Key strengths mentioned
- Key weaknesses or concerns
- Common themes across reviews
- Product features frequently mentioned
- Recommendations or suggestions

Product Reviews:
{reviews_text_escaped}

Provide a comprehensive summary that captures the essence of all reviews."""
            
            prompt_escaped = prompt.replace("'", "''")
            
            fallback_query = f"""
            SELECT ai_query(
                'databricks-claude-opus-4-5',
                '{prompt_escaped}'
            ) AS summary
            """
            
            cursor = connection.cursor()
            cursor.execute(fallback_query)
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else f"Error generating summary: {str(e)}"
        except Exception as e2:
            return f"Error generating summary: {str(e2)}"


def main():
    st.title("📊 Product Review Summarizer")
    st.markdown("Enter a product ID to generate a comprehensive summary of all reviews from the Delta table.")
    
    # Table configuration
    connection = get_databricks_connection()
    catalog = "users"
    schema = "landan_george"
    table = "reviews"

    # Main content area
    if not connection:
        st.warning("Please configure your Databricks connection in the sidebar to continue.")
        return
    
    # Product ID input
    col1, col2 = st.columns([3, 1])
    with col1:
        product_id = st.text_input(
            "Product ID",
            placeholder="Enter product ID...",
            help="Enter the product ID to summarize reviews for"
        )
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        generate_button = st.button("Generate Summary", type="primary", use_container_width=True)
    
    # Generate summary
    if generate_button and product_id:
        if not product_id.strip():
            st.warning("Please enter a valid product ID.")
        else:
            with st.spinner("Fetching reviews and generating summary..."):
                # Show review count first
                reviews_df = fetch_reviews_for_product(connection, catalog, schema, table, product_id)
                
                if reviews_df.empty:
                    st.warning(f"No reviews found for Product ID: {product_id}")
                else:
                    st.info(f"Found {len(reviews_df)} review(s) for Product ID: {product_id}")
                    
                    # Generate summary
                    summary = generate_summary_with_ai(connection, catalog, schema, table, product_id)
                    
                    # Display summary
                    st.subheader("📝 Summary")
                    st.markdown(summary)
                    
                    # Option to view raw reviews
                    with st.expander("View Raw Reviews"):
                        st.dataframe(reviews_df, use_container_width=True)
    
    elif generate_button:
        st.warning("Please enter a product ID before generating a summary.")


if __name__ == "__main__":
    main()
