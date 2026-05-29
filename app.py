from flask import Flask, render_template, request, session, redirect, flash, jsonify
import sqlite3
import os
import json
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model
import requests
import random
import re
from datetime import datetime, timedelta
from uuid import uuid4


# =========================
# BASIC CONFIG
# =========================

API_KEY = "862d82173b624c7f84071615260105"

app = Flask(__name__)
app.secret_key = "smartcloset_secret"

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================
# MODEL LOADING
# =========================

clothing_model = load_model("models/clothing_model.h5")

with open("models/clothing_class_indices.json", "r") as f:
    clothing_class_indices = json.load(f)

clothing_classes = {v: k for k, v in clothing_class_indices.items()}

style_model = load_model("models/style_model.h5")

with open("models/style_class_indices.json", "r") as f:
    style_class_indices = json.load(f)

style_classes = {v: k for k, v in style_class_indices.items()}


# =========================
# HELPERS
# =========================

def get_db_connection():
    return sqlite3.connect("users.db")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def make_unique_filename(original_filename):
    safe_name = secure_filename(original_filename)
    name, ext = os.path.splitext(safe_name)
    unique_id = uuid4().hex[:10]
    return f"{name}_{unique_id}{ext}"


# =========================
# HOME / AUTH
# =========================

@app.route("/")
def home():
    if "user" in session:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT image_name, category, color, style
        FROM clothes
        WHERE user_email = ?
        ORDER BY id DESC
        LIMIT 6
        """, (session["user"],))

        saved_clothes = cursor.fetchall()
        conn.close()

        return render_template("dashboard.html", saved_clothes=saved_clothes)

    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        birthdate = request.form["birthdate"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        email_pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
        password_pattern = r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&^#])[A-Za-z\d@$!%*?&^#]{8,}$"

        if not re.match(email_pattern, email):
            flash("Email must be like name@email.com")
            return redirect("/register")

        if not re.match(password_pattern, password):
            flash("Password must contain capital letter, number, symbol, and be 8+ characters")
            return redirect("/register")

        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect("/register")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            conn.close()
            flash("Email already exists!")
            return redirect("/register")

        cursor.execute("""
        INSERT INTO users(username, email, birthdate, password, avoid_days)
        VALUES (?, ?, ?, ?, ?)
        """, (username, email, birthdate, password, 7))

        conn.commit()
        conn.close()

        flash("User registered successfully! Please login.")
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT * FROM users
        WHERE email = ? AND password = ?
        """, (email, password))

        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = email
            return redirect("/")

        flash("Wrong email or password!")
        return redirect("/login")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


# =========================
# AI PREDICTIONS
# =========================

def predict_clothing(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = clothing_model.predict(img_array)
    class_index = int(np.argmax(prediction))

    return clothing_classes[class_index]


def predict_style(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = style_model.predict(img_array)
    class_index = int(np.argmax(prediction))

    return style_classes[class_index]


color_palette = {
    "black": [25, 25, 25],
    "white": [240, 240, 240],
    "grey": [125, 125, 125],
    "blue": [45, 80, 140],
    "navy blue": [25, 45, 80],
    "red": [185, 45, 45],
    "green": [55, 130, 75],
    "brown": [120, 75, 45],
    "beige": [215, 200, 165],
    "yellow": [230, 210, 70],
    "orange": [230, 135, 45],
    "pink": [225, 145, 175],
    "purple": [125, 70, 155],
    "metallic": [180, 180, 180]
}


def preprocess_color_image(image_path):
    img = Image.open(image_path).convert("RGBA")
    img_np = np.array(img)
    alpha = img_np[:, :, 3]

    coords = np.argwhere(alpha > 20)

    if len(coords) == 0:
        return np.array(Image.open(image_path).convert("RGB"))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    cropped = img_np[y_min:y_max + 1, x_min:x_max + 1]

    rgb = cropped[:, :, :3]
    alpha_crop = cropped[:, :, 3] / 255.0
    white_bg = np.ones_like(rgb) * 255

    final = (rgb * alpha_crop[:, :, None] + white_bg * (1 - alpha_crop[:, :, None])).astype(np.uint8)
    return final


def closest_palette_color(rgb):
    rgb = np.array(rgb)
    best_color = None
    min_dist = float("inf")

    for name, value in color_palette.items():
        value = np.array(value)
        dist = np.linalg.norm(rgb - value)

        if dist < min_dist:
            min_dist = dist
            best_color = name

    return best_color


def get_top_colors_from_image(img, k=5):
    h, w, _ = img.shape

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.20)
    y2 = int(h * 0.80)

    center_img = img[y1:y2, x1:x2]
    pixels = center_img.reshape((-1, 3))

    mask = ~(
        ((pixels[:, 0] > 220) & (pixels[:, 1] > 220) & (pixels[:, 2] > 220)) |
        ((abs(pixels[:, 0] - pixels[:, 1]) < 10) &
         (abs(pixels[:, 1] - pixels[:, 2]) < 10) &
         (pixels[:, 0] > 170))
    )

    pixels = pixels[mask]

    if len(pixels) < 100:
        pixels = center_img.reshape((-1, 3))

    k = min(k, len(pixels))

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    colors = kmeans.cluster_centers_[order].astype(int)
    percentages = counts[order] / counts.sum()

    return colors, percentages


def predict_color(image_path):
    processed_img = preprocess_color_image(image_path)
    colors, percentages = get_top_colors_from_image(processed_img, k=5)

    results = []

    for rgb, pct in zip(colors, percentages):
        color_name = closest_palette_color(rgb)
        results.append((color_name, rgb, pct))

    for color_name, rgb, pct in results:
        if pct >= 0.18:
            return color_name

    return results[0][0]


# =========================
# WEATHER LOGIC
# =========================

def predict_suitable_weather(category):
    """
    This value is stored in the database and used by the recommendation system.
    Category name is used only to classify the item type, then we convert it into
    a weather suitability tag.
    """
    category = normalize_text(category)

    cold_items = [
        "jackets", "jacket", "coat", "coats",
        "sweaters", "sweater",
        "hoodies", "hoodie",
        "sweatshirts", "sweatshirt",
        "boots", "boot"
    ]

    hot_items = [
        "tshirts", "tshirt", "t-shirts", "t-shirt",
        "tops", "top",
        "tank top", "tank tops",
        "shorts",
        "sandals", "sandal"
    ]

    mild_items = [
        "shirts", "shirt", "blouse", "blouses",
        "pants", "jeans", "trousers",
        "skirts", "skirt",
        "dresses", "dress",
        "leggings", "track pants"
    ]

    all_weather_items = [
        "shoes", "shoe",
        "sneakers", "heels", "flats",
        "accessories", "accessory"
    ]

    if category in cold_items:
        return "Cold Weather"

    if category in hot_items:
        return "Hot Weather"

    if category in mild_items:
        return "Mild Weather"

    if category in all_weather_items:
        return "All Weather"

    return "All Weather"

def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception:
        return None, None

    if "current" not in data:
        return None, None

    temperature = data["current"].get("temp_c")
    condition = data["current"].get("condition", {}).get("text", "")

    return temperature, condition


def get_weather_type(temp, condition=""):

    condition = condition.lower()

    if (
        "snow" in condition or
        "sleet" in condition or
        "blizzard" in condition or
        temp <= 15
    ):
        return "Cold Weather"

    elif (
        "rain" in condition or
        "drizzle" in condition or
        "thunder" in condition or
        "storm" in condition
    ):
        return "Mild Weather"

    elif temp >= 25:
        return "Hot Weather"

    else:
        return "Mild Weather"


@app.route("/current_weather")
def current_weather():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({"error": "Location not found"})

    location = f"{lat},{lon}"
    temperature, condition = get_weather(location)

    if temperature is None:
        return jsonify({"error": "Unable to load weather"})

    weather_type = get_weather_type(temperature, condition)
    return jsonify({
        "temperature": temperature,
        "condition": condition,
        "weather_type": weather_type
    })


@app.route("/weather", methods=["GET", "POST"])
def weather():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    temperature = None
    condition = None
    city = None

    if request.method == "POST":
        city = request.form["city"]
        temperature, condition = get_weather(city)

        if temperature is None:
            flash("Weather information could not be loaded.")

    return render_template("weather.html", temperature=temperature, condition=condition, city=city)


# =========================
# CLOSET
# =========================

def needs_alert(last_worn):

    if not last_worn:
        return False

    try:
        last_date = datetime.strptime(
            last_worn,
            "%Y-%m-%d %H:%M:%S"
        )
    except:
        return False

    return datetime.now() - last_date > timedelta(days=90)

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    if request.method == "POST":
        image = request.files.get("image")

        if not image or image.filename == "":
            flash("No image selected!")
            return redirect("/upload")

        filename = make_unique_filename(image.filename)

        user_folder = os.path.join(app.config["UPLOAD_FOLDER"], session["user"])
        os.makedirs(user_folder, exist_ok=True)

        image_path = os.path.join(user_folder, filename)
        image.save(image_path)

        category = predict_clothing(image_path)
        style = predict_style(image_path)
        color = predict_color(image_path)
        suitable_weather = predict_suitable_weather(category)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO clothes(user_email,
            image_name,
            category,
            color,
            style,
            suitable_weather,
            last_worn)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session["user"],
            filename,
            category,
            color,
            style,
            suitable_weather,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        conn.close()

        return redirect("/upload")

    return render_template("upload.html")


@app.route("/closet")
def closet():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT * FROM clothes
    WHERE user_email = ?
    ORDER BY id DESC
    """, (session["user"],))

    clothes = cursor.fetchall()
    conn.close()

    return render_template(
        "closet.html",
        clothes=clothes,
        needs_alert=needs_alert
    )

@app.route("/update/<int:item_id>", methods=["POST"])
def update_item(item_id):
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    category = request.form["category"].strip()
    color = request.form["color"].strip()
    style = request.form["style"].strip()
    suitable_weather = request.form.get("suitable_weather", predict_suitable_weather(category)).strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE clothes
    SET category = ?, color = ?, style = ?, suitable_weather = ?
    WHERE id = ? AND user_email = ?
    """, (category, color, style, suitable_weather, item_id, session["user"]))

    conn.commit()
    conn.close()

    flash("Item updated successfully!")
    return redirect("/closet")


@app.route("/delete/<int:item_id>")
def delete_item(item_id):
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT image_name FROM clothes
    WHERE id = ? AND user_email = ?
    """, (item_id, session["user"]))

    item = cursor.fetchone()

    if item:
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], session["user"], item[0])

        if os.path.exists(image_path):
            os.remove(image_path)

        cursor.execute("""
        DELETE FROM outfit_items
        WHERE clothing_id = ?
        """, (item_id,))

        cursor.execute("""
        DELETE FROM clothes
        WHERE id = ? AND user_email = ?
        """, (item_id, session["user"]))

        conn.commit()
        flash("Item deleted successfully!")

    conn.close()
    return redirect("/closet")


# =========================
# RECOMMENDATION LOGIC
# =========================

def get_item_type(category):
    """
    Category is used only to understand the outfit structure:
    top / bottom / dress / jacket / shoes / accessory.
    Weather decision does NOT depend on this function.
    """
    category = normalize_text(category)

    if category in [
        "tshirts", "tshirt", "t-shirts", "t-shirt",
        "shirts", "shirt",
        "tops", "top",
        "tank top", "tank tops",
        "blouse", "blouses",
        "sweaters", "sweater",
        "hoodies", "hoodie",
        "sweatshirts", "sweatshirt"
    ]:
        return "top"

    if category in [
        "pants", "jeans", "trousers",
        "shorts",
        "skirts", "skirt",
        "leggings", "track pants"
    ]:
        return "bottom"

    if category in ["dress", "dresses"]:
        return "dress"

    if category in ["jackets", "jacket", "coat", "coats"]:
        return "jacket"

    if category in [
        "shoes", "shoe",
        "sneakers", "boots", "boot",
        "sandals", "sandal",
        "heels", "flats"
    ]:
        return "shoes"

    if category in ["accessories", "accessory"]:
        return "accessory"

    return "other"


def item_weather(item):
    """
    clothes table order used in this app:
    id, user_email, image_name, category, color, style, suitable_weather
    """
    if len(item) > 6 and item[6]:
        return item[6]

    # fallback for old records that may not have suitable_weather filled
    return predict_suitable_weather(item[3])


def is_item_suitable_for_weather(item, weather_type):
    """
    Main weather filter.
    The recommendation depends on suitable_weather, not on category name.
    """
    suitable_weather = normalize_text(item_weather(item))
    weather_type = normalize_text(weather_type)

    if suitable_weather == "all weather":
        return True

    return suitable_weather == weather_type


def get_blocked_outfit_sets(user_email):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute('''
    SELECT id FROM outfits
    WHERE user_email = ?
    AND avoid_until IS NOT NULL
    AND datetime(avoid_until) > datetime('now')
    ''', (user_email,))

    active_outfits = cursor.fetchall()
    blocked_sets = []

    for outfit in active_outfits:
        cursor.execute('''
        SELECT clothing_id FROM outfit_items
        WHERE outfit_id = ?
        ''', (outfit[0],))

        items = cursor.fetchall()
        blocked_sets.append(sorted([item[0] for item in items]))

    conn.close()
    return blocked_sets


def recommend_outfit(user_email, weather_type, selected_style, old_outfit_ids=None, change_type=None, attempts=0):
    if old_outfit_ids is None:
        old_outfit_ids = []

    if attempts > 30:
        return []

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute('''
    SELECT * FROM clothes
    WHERE user_email = ?
    AND lower(trim(style)) = lower(trim(?))
    ''', (user_email, selected_style))

    clothes = cursor.fetchall()
    conn.close()

    blocked_sets = get_blocked_outfit_sets(user_email)

    # Weather filtering happens here.
    # Category name is NOT used to decide hot/cold/mild suitability.
    suitable_clothes = [
        item for item in clothes
        if is_item_suitable_for_weather(item, weather_type)
    ]

    old_items = []
    for old_id in old_outfit_ids:
        for item in suitable_clothes:
            if str(item[0]) == str(old_id):
                old_items.append(item)

    tops = []
    bottoms = []
    dresses = []
    jackets = []
    shoes = []
    accessories = []
    others = []

    for item in suitable_clothes:
        item_type = get_item_type(item[3])

        if item_type == "top":
            tops.append(item)
        elif item_type == "bottom":
            bottoms.append(item)
        elif item_type == "dress":
            dresses.append(item)
        elif item_type == "jacket":
            jackets.append(item)
        elif item_type == "shoes":
            shoes.append(item)
        elif item_type == "accessory":
            accessories.append(item)
        else:
            others.append(item)

    outfit = []
    used_item_ids = set()

    def choose_one(items, item_type):
        available_items = [
            item for item in items
            if item[0] not in used_item_ids
        ]

        if not available_items:
            return None

        if change_type == item_type:
            current_ids = [str(item[0]) for item in old_items]
            changed_items = [
                item for item in available_items
                if str(item[0]) not in current_ids
            ]

            if changed_items:
                selected = random.choice(changed_items)
                used_item_ids.add(selected[0])
                return selected

        for old_item in old_items:
            if get_item_type(old_item[3]) == item_type and old_item[0] not in used_item_ids:
                used_item_ids.add(old_item[0])
                return old_item

        selected = random.choice(available_items)
        used_item_ids.add(selected[0])
        return selected

    def choose_many(items, item_type, min_count=1, max_count=2):
        available_items = [
            item for item in items
            if item[0] not in used_item_ids
        ]

        if not available_items:
            return []

        if change_type == item_type:
            current_ids = [str(item[0]) for item in old_items]
            available_items = [
                item for item in available_items
                if str(item[0]) not in current_ids
            ]

            if not available_items:
                return []

        count = min(random.randint(min_count, max_count), len(available_items))
        selected_items = random.sample(available_items, count)

        for item in selected_items:
            used_item_ids.add(item[0])

        return selected_items

    # Outfit structure only. Weather filtering already happened above.
    use_dress = False
    if dresses and random.random() < 0.25:
        use_dress = True

    if use_dress:
        dress = choose_one(dresses, "dress")
        if dress:
            outfit.append(dress)
    else:
        top = choose_one(tops, "top")
        bottom = choose_one(bottoms, "bottom")

        if top:
            outfit.append(top)

        if bottom:
            outfit.append(bottom)

    jacket = choose_one(jackets, "jacket")
    if jacket:
        outfit.append(jacket)

    shoe = choose_one(shoes, "shoes")
    if shoe:
        outfit.append(shoe)

    selected_accessories = choose_many(accessories, "accessory", 1, 2)
    outfit.extend(selected_accessories)

    if len(outfit) < 2 and others:
        extra = choose_one(others, "other")
        if extra:
            outfit.append(extra)

    current_outfit_ids = sorted([item[0] for item in outfit])

    if current_outfit_ids in blocked_sets:
        return recommend_outfit(
            user_email,
            weather_type,
            selected_style,
            old_outfit_ids,
            change_type,
            attempts + 1
        )

    return outfit


@app.route("/recommend", methods=["GET", "POST"])
def recommend():

    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    outfits = []
    city = None
    temperature = None
    condition = None
    weather_type = None
    selected_style = None

    if request.method == "POST":

        city = request.form["city"]
        selected_style = request.form["style"]

        temperature, condition = get_weather(city)

        if temperature is None:
            flash("Weather information could not be loaded. Please choose another city.")
            return redirect("/recommend")

        weather_type = get_weather_type(temperature, condition)

        target_outfit_index = request.form.get("target_outfit_index")
        remove_item_id = request.form.get("remove_item_id")
        change_item_id = request.form.get("change_item_id")
        change_type = request.form.get("change_type")

        if change_type:
            change_type = get_item_type(change_type)

        if target_outfit_index is not None:

            target_outfit_index = int(target_outfit_index)

            conn = get_db_connection()
            cursor = conn.cursor()

            all_outfits = []
            index = 0

            while True:
                ids = request.form.getlist(f"outfit_{index}_ids")

                if not ids:
                    break

                outfit = []

                for item_id in ids:
                    cursor.execute(
                        "SELECT * FROM clothes WHERE id = ? AND user_email = ?",
                        (item_id, session["user"])
                    )

                    item = cursor.fetchone()

                    if item:
                        outfit.append(item)

                all_outfits.append(outfit)
                index += 1

            if remove_item_id:
                all_outfits[target_outfit_index] = [
                    item for item in all_outfits[target_outfit_index]
                    if str(item[0]) != str(remove_item_id)
                ]

            elif change_item_id and change_type:


                cursor.execute(
                    """
                    SELECT * FROM clothes
                    WHERE user_email = ?
                    AND lower(trim(style)) = lower(trim(?))
                    AND lower(trim(suitable_weather)) IN (lower(trim(?)), 'all weather')
                    ORDER BY RANDOM()
                    """,
                    (session["user"], selected_style, weather_type)
                )

                candidates = cursor.fetchall()

                same_type_candidates = [
                    item for item in candidates
                    if get_item_type(item[3]) == change_type
                ]

                new_item = None

                for item in same_type_candidates:
                    if str(item[0]) != str(change_item_id):
                        new_item = item
                        break

                if not new_item and same_type_candidates:
                    new_item = same_type_candidates[0]

                if new_item:
                    new_outfit = []

                    for item in all_outfits[target_outfit_index]:
                        if str(item[0]) == str(change_item_id):
                            new_outfit.append(new_item)
                        else:
                            new_outfit.append(item)

                    all_outfits[target_outfit_index] = new_outfit

            conn.close()

            return render_template(
                "recommend.html",
                outfits=all_outfits,
                city=city,
                temperature=temperature,
                condition=condition,
                weather_type=weather_type,
                selected_style=selected_style
            )

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
        SELECT COUNT(*)
        FROM clothes
        WHERE user_email = ?
        AND lower(trim(style)) = lower(trim(?))
        """, (session["user"], selected_style))

        items_count = cursor.fetchone()[0]
        conn.close()

        if items_count < 5:
            flash("Add more clothes with this style to get better outfits.")

        if items_count < 10:
            outfits_number = 3
        elif items_count < 20:
            outfits_number = 4
        else:
            outfits_number = 5

        attempts = 0

        while len(outfits) < outfits_number and attempts < 40:

            outfit = recommend_outfit(
                session["user"],
                weather_type,
                selected_style
            )

            outfit_ids = sorted([item[0] for item in outfit])

            existing_outfits_ids = [
                sorted([item[0] for item in old_outfit])
                for old_outfit in outfits
            ]

            if outfit and outfit_ids not in existing_outfits_ids:
                outfits.append(outfit)

            attempts += 1

        if not outfits:
            flash("No suitable outfit found. Add more clothes to your closet.")

    return render_template(
        "recommend.html",
        outfits=outfits,
        city=city,
        temperature=temperature,
        condition=condition,
        weather_type=weather_type,
        selected_style=selected_style
    )


# =========================
# SAVE / DELETE OUTFITS
# =========================

@app.route("/wore_outfit", methods=["POST"])
def wore_outfit():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    city = request.form["city"]
    temperature = request.form["temperature"]
    condition = request.form["condition"]
    weather_type = request.form["weather_type"]
    outfit_items = request.form.getlist("outfit_items")

    if not outfit_items:
        flash("No outfit items selected.")
        return redirect("/recommend")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT avoid_days
    FROM users
    WHERE email = ?
    """, (session["user"],))

    result = cursor.fetchone()

    try:
        avoid_days = int(result[0]) if result and result[0] is not None else 7
    except ValueError:
        avoid_days = 7

    avoid_until = datetime.now() + timedelta(days=avoid_days)

    cursor.execute("""
    INSERT INTO outfits(user_email, city, temperature, condition, weather_type, avoid_until)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        session["user"],
        city,
        temperature,
        condition,
        weather_type,
        avoid_until.strftime("%Y-%m-%d %H:%M:%S")
    ))

    outfit_id = cursor.lastrowid

    for clothing_id in outfit_items:
        cursor.execute("""
        INSERT INTO outfit_items(outfit_id, clothing_id, item_type)
        VALUES (?, ?, ?)
        """, (outfit_id, clothing_id, "saved"))

    conn.commit()
    conn.close()

    flash("Outfit saved successfully!")
    return redirect("/recommend")


@app.route("/saved_outfits")
def saved_outfits():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT outfits.id,
           outfits.city,
           outfits.temperature,
           outfits.condition,
           outfits.weather_type,
           outfits.created_at,
           clothes.image_name,
           clothes.category,
           clothes.color,
           clothes.style
    FROM outfits
    JOIN outfit_items ON outfits.id = outfit_items.outfit_id
    JOIN clothes ON clothes.id = outfit_items.clothing_id
    WHERE outfits.user_email = ?
    ORDER BY outfits.id DESC
    """, (session["user"],))

    rows = cursor.fetchall()
    conn.close()

    saved_outfits_data = {}

    for row in rows:
        outfit_id = row[0]

        if outfit_id not in saved_outfits_data:
            saved_outfits_data[outfit_id] = {
                "city": row[1],
                "temperature": row[2],
                "condition": row[3],
                "weather_type": row[4],
                "created_at": row[5],
                "items": []
            }

        saved_outfits_data[outfit_id]["items"].append({
            "image_name": row[6],
            "category": row[7],
            "color": row[8],
            "style": row[9]
        })

    return render_template("saved_outfits.html", saved_outfits=saved_outfits_data)


@app.route("/delete_saved_outfit/<int:outfit_id>")
def delete_saved_outfit(outfit_id):
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id FROM outfits
    WHERE id = ? AND user_email = ?
    """, (outfit_id, session["user"]))

    outfit = cursor.fetchone()

    if outfit:
        cursor.execute("DELETE FROM outfit_items WHERE outfit_id = ?", (outfit_id,))
        cursor.execute("DELETE FROM outfits WHERE id = ? AND user_email = ?", (outfit_id, session["user"]))
        conn.commit()
        flash("Outfit deleted successfully!")
    else:
        flash("Outfit not found.")

    conn.close()
    return redirect("/saved_outfits")


# =========================
# PROFILE
# =========================

@app.route("/profile")
def profile():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT username, email, birthdate, avoid_days
    FROM users
    WHERE email = ?
    """, (session["user"],))

    user = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM clothes WHERE user_email = ?", (session["user"],))
    clothes_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM outfits WHERE user_email = ?", (session["user"],))
    outfits_count = cursor.fetchone()[0]

    conn.close()

    return render_template("profile.html", user=user, clothes_count=clothes_count, outfits_count=outfits_count)


@app.route("/update_profile", methods=["POST"])
def update_profile():
    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    username = request.form["username"].strip()
    birthdate = request.form["birthdate"]

    try:
        avoid_days = int(request.form["avoid_days"])
    except ValueError:
        avoid_days = 7

    if avoid_days < 0:
        avoid_days = 7

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE users
    SET username = ?, birthdate = ?, avoid_days = ?
    WHERE email = ?
    """, (username, birthdate, avoid_days, session["user"]))

    conn.commit()
    conn.close()

    flash("Profile updated successfully!")
    return redirect("/profile")

@app.route("/change_password", methods=["GET", "POST"])
def change_password():

    if "user" not in session:
        flash("Please login first!")
        return redirect("/login")

    if request.method == "POST":

        old_password = request.form["old_password"]
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        password_pattern = r"^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&^#])[A-Za-z\d@$!%*?&^#]{8,}$"

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT password FROM users WHERE email = ?",
            (session["user"],)
        )

        user = cursor.fetchone()

        if not user or user[0] != old_password:

            conn.close()

            return render_template(
                "change_password.html",
                old_error="Old password is incorrect!"
            )

        if not re.match(password_pattern, new_password):

            conn.close()

            return render_template(
                "change_password.html",
                new_error="Password must contain capital letter, number, symbol, and be 8+ characters"
            )

        if new_password != confirm_password:

            conn.close()

            return render_template(
                "change_password.html",
                match_error="Passwords do not match!"
            )

        cursor.execute(
            "UPDATE users SET password = ? WHERE email = ?",
            (new_password, session["user"])
        )

        conn.commit()
        conn.close()

        flash("Password changed successfully!")
        return redirect("/profile")

    return render_template("change_password.html")


# =========================
# RUN APP
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)