# Book Data Collector & Loader

I built this for Zepto capstone to collect book info and store it properly.

**Flow I followed:**

Scraping part - I took https://books.toscrape.com/ and pulled 5 pages which gives 100 books. For each book I opened detail page to get category name, because listing page doesn't show it.

Cleaning part - Price was like £38.19 so I removed £ symbol. Rating was in words like Three, Four so I changed to numbers 3,4. Stock text I converted to 1/0.

INR logic - Assignment said fixed rate so I used 105.50. Formula: price_gbp * 105.50 = price_inr.

DB part - Used sqlite3. Created 2 tables. First table holds categories, second holds books with foreign key link. On every run I delete old db so no duplicate.

Queries part - Wrote 6 queries as per task:
- q1 filter high rated books
- q2 sort costly books and take top 10
- q3 unique list
- q4 range filter
- q5 id filter
- q6 join both tables, also did same with pandas merge to cross check.

**Run it:**