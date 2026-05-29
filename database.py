import sqlite3

conn = sqlite3.connect('users.db')

cursor = conn.cursor()


# =========================
# USERS TABLE
# =========================

cursor.execute('''
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    birthdate TEXT,
    password TEXT,
    avoid_days INTEGER DEFAULT 7
)
''')


# ADD birthdate COLUMN IF NOT EXISTS

try:
    cursor.execute("ALTER TABLE users ADD COLUMN birthdate TEXT")
except sqlite3.OperationalError:
    pass

# ADD avoid_days COLUMN IF NOT EXISTS

try:
    cursor.execute("ALTER TABLE users ADD COLUMN avoid_days INTEGER DEFAULT 7")
except sqlite3.OperationalError:
    pass


# =========================
# CLOTHES TABLE
# =========================

cursor.execute('''
CREATE TABLE IF NOT EXISTS clothes(

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_email TEXT,
    image_name TEXT,
    category TEXT,
    color TEXT,
    style TEXT,
    suitable_weather TEXT
)
''')

# ADD last_worn COLUMN IF NOT EXISTS

try:
    cursor.execute("ALTER TABLE clothes ADD COLUMN last_worn TEXT")
except sqlite3.OperationalError:
    pass


# =========================
# OUTFITS TABLE
# =========================

cursor.execute('''
CREATE TABLE IF NOT EXISTS outfits (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    user_email TEXT,
    city TEXT,
    temperature REAL,
    condition TEXT,
    weather_type TEXT,
    avoid_until TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')


# ADD avoid_until COLUMN IF NOT EXISTS

try:
    cursor.execute("ALTER TABLE outfits ADD COLUMN avoid_until TEXT")
except sqlite3.OperationalError:
    pass


# =========================
# OUTFIT ITEMS TABLE
# =========================

cursor.execute('''
CREATE TABLE IF NOT EXISTS outfit_items (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    outfit_id INTEGER,
    clothing_id INTEGER,
    item_type TEXT,

    FOREIGN KEY(outfit_id) REFERENCES outfits(id),
    FOREIGN KEY(clothing_id) REFERENCES clothes(id)
)
''')


# =========================
# SAVE CHANGES
# =========================

conn.commit()

conn.close()

print("Database Created Successfully!")