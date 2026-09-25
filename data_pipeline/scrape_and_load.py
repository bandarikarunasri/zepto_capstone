import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import os

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"
GBP_INR_RATE = 105.50

def scrape_books_data(page_count=5):
    collected = []
    print(f"Scraping {page_count} pages...")
    for p in range(1, page_count+1):
        page_url = f"{BASE_URL}catalogue/page-{p}.html"
        res = requests.get(page_url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        for item in soup.select("article.product_pod"):
            book_title = item.h3.a['title'].strip()
            price_raw = item.select_one("p.price_color").text
            star_word = item.select_one("p.star-rating")['class'][1]
            stock_info = item.select_one("p.instock.availability").text.strip()

            detail_href = item.h3.a['href'].replace('../../../', '')
            detail_page = requests.get(CATALOGUE_URL + detail_href, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
            detail_soup = BeautifulSoup(detail_page.text, 'html.parser')
            book_category = detail_soup.select("ul.breadcrumb li a")[2].text.strip()

            collected.append({
                "book_title": book_title,
                "raw_price": price_raw,
                "star_word": star_word,
                "stock_text": stock_info,
                "cat_name": book_category
            })
    return pd.DataFrame(collected)

def process_dataframe(df):
    star_to_num = {"One":1, "Two":2, "Three":3, "Four":4, "Five":5}
    df['price_gbp'] = df['raw_price'].str.replace(r'[^0-9.]','', regex=True).astype(float)
    df['star_rating'] = df['star_word'].map(star_to_num)
    df['stock_flag'] = df['stock_text'].str.contains('In stock', case=False).astype(int)
    df['price_gbp'] = df['price_gbp'].fillna(df['price_gbp'].median())
    df['star_rating'] = df['star_rating'].fillna(df['star_rating'].median())
    df.dropna(subset=['book_title'], inplace=True)
    df['price_inr'] = (df['price_gbp'] * GBP_INR_RATE).round(2)
    return df

def load_into_db(df):
    db_file = os.path.join(os.path.dirname(__file__), "books.db")
    if os.path.exists(db_file):
        os.remove(db_file)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE book_categories (
        category_id INTEGER PRIMARY KEY AUTOINCREMENT,
        category_name TEXT UNIQUE NOT NULL
    )""")
    cur.execute("""CREATE TABLE book_inventory (
        book_id INTEGER PRIMARY KEY AUTOINCREMENT,
        book_title TEXT NOT NULL,
        price_gbp REAL,
        price_inr REAL,
        star_rating INTEGER,
        stock_status INTEGER,
        category_id INTEGER,
        FOREIGN KEY(category_id) REFERENCES book_categories(category_id)
    )""")

    for cat in df['cat_name'].unique():
        cur.execute("INSERT INTO book_categories (category_name) VALUES (?)", (cat,))
    cat_lookup = {name: cid for cid, name in cur.execute("SELECT category_id, category_name FROM book_categories")}

    for _, r in df.iterrows():
        cur.execute("""INSERT INTO book_inventory
        (book_title, price_gbp, price_inr, star_rating, stock_status, category_id)
        VALUES (?,?,?,?,?,?)""",
        (r['book_title'], r['price_gbp'], r['price_inr'], int(r['star_rating']), int(r['stock_flag']), cat_lookup[r['cat_name']]))

    conn.commit()
    return conn

def execute_queries(conn):
    base = os.path.dirname(__file__)
    queries = {
        "q1_where": "SELECT book_title, star_rating FROM book_inventory WHERE star_rating >= 4",
        "q2_order_by_limit": "SELECT book_title, price_inr FROM book_inventory ORDER BY price_inr DESC LIMIT 10",
        "q3_distinct": "SELECT DISTINCT category_name FROM book_categories",
        "q4_between": "SELECT book_title, price_gbp FROM book_inventory WHERE price_gbp BETWEEN 20 AND 30",
        "q5_in_clause": "SELECT book_title, category_id FROM book_inventory WHERE category_id IN (1,2,3)",
        "q6_join": "SELECT b.book_title, b.star_rating, c.category_name FROM book_inventory b JOIN book_categories c ON b.category_id = c.category_id ORDER BY b.star_rating DESC LIMIT 10"
    }
    for fname, sql in queries.items():
        df_q = pd.read_sql_query(sql, conn)
        print(f"\n--- {fname} ---")
        print(df_q.head(5))
        df_q.to_csv(os.path.join(base, f"{fname}_output.csv"), index=False)

    # pd.merge proof for JOIN
    inv_df = pd.read_sql_query("SELECT * FROM book_inventory", conn)
    cat_df = pd.read_sql_query("SELECT * FROM book_categories", conn)
    merged_df = pd.merge(inv_df, cat_df, on="category_id", how="inner").sort_values("star_rating", ascending=False).head(10)
    print("\n pd.merge JOIN verification done")
    print(merged_df[['book_title','star_rating','category_name']].head())

if __name__ == "__main__":
    raw_df = scrape_books_data(5)
    clean_df = process_dataframe(raw_df)
    print(f"Total: {len(clean_df)} books, {clean_df['cat_name'].nunique()} categories")
    connection = load_into_db(clean_df)
    execute_queries(connection)
    connection.close()
    print("\nDONE - books.db created successfully")