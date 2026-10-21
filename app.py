from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import re
import uuid
import tensorflow as tf

app = Flask(__name__)

MODEL_FILE = "models/plant_model_v3.keras"
CLASSES_FILE = "class_names_v3.json"
IMG_SIZE = 224
MIN_CONFIDENCE = 0.50   # isse kam yaqeen par "Pata nahi" dikhao

model = tf.keras.models.load_model(MODEL_FILE)
class_names = json.load(open(CLASSES_FILE))

with open("plant_disease.json", "r", encoding="utf-8") as file:
    plant_disease = json.load(file)


def key(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def pretty(name):
    return name.replace("_", " ").title()


# ---------- wajah aur ilaaj (plant_disease.json) ----------
CROP_FIX = {"corn_maize": "corn", "cherry_including_sour": "cherry",
            "pepper_bell": "bell_pepper", "orange": "citrus"}
ALIAS = {
    "apple_rust": "apple_cedar_apple_rust",
    "corn_gray_leaf_spot": "corn_cercospora_leaf_spot_gray_leaf_spot",
    "corn_rust": "corn_common_rust",
    "grape_leaf_spot": "grape_leaf_blight_isariopsis_leaf_spot",
    "citrus_greening": "citrus_haunglongbing_citrus_greening",
    "tomato_spider_mites": "tomato_spider_mites_two_spotted_spider_mite",
}
info_by_key = {}
for d in plant_disease:
    crop, _, disease = d["name"].partition("___")
    c = CROP_FIX.get(key(crop), key(crop))
    dk = key(disease)
    if dk.startswith(c + "_"):
        dk = dk[len(c) + 1:]
    info_by_key[c + "_" + dk] = d


def info_for(name):
    k = ALIAS.get(name, name)
    if k in info_by_key:
        return info_by_key[k]
    return {"name": pretty(name),
            "cause": "Is bimari ki tafseel abhi shamil nahi hai.",
            "cure": "Kisi zarai maahir se mashwara karein."}


# ---------- fasal ke hisaab se classes ----------
CROPS = sorted([
    "apple", "banana", "basil", "bean", "bell_pepper", "blueberry", "broccoli",
    "cabbage", "carrot", "cauliflower", "celery", "cherry", "citrus", "coffee",
    "corn", "cotton", "cucumber", "eggplant", "garlic", "ginger", "grape",
    "grapevine", "lettuce", "maple", "peach", "plum", "potato", "raspberry",
    "rice", "soybean", "squash", "strawberry", "sugarcane", "tobacco", "tomato",
    "wheat", "zucchini",
], key=len, reverse=True)
CROP_ALIAS = {"grapevine": "grape"}
OVERRIDE = {"flag_smut": "wheat", "leaf_curl": "cotton", "wilt": "cotton",
            "leaf_smut": "rice", "army_worm": "other"}


def crop_of(name):
    if name in OVERRIDE:
        return OVERRIDE[name]
    wrapped = "_" + name + "_"
    for c in CROPS:
        if "_" + c + "_" in wrapped:
            return CROP_ALIAS.get(c, c)
    return "other"


CLASS_CROP = [crop_of(n) for n in class_names]
CROP_LIST = sorted(set(CLASS_CROP) - {"other"})
CROP_IDX = {c: np.array([i for i, x in enumerate(CLASS_CROP) if x == c]) for c in CROP_LIST}

missing = [n for n in class_names if ALIAS.get(n, n) not in info_by_key]
print(len(class_names), "classes |", len(missing), "ki tafseel json mein nahi:", missing)


# ---------- pages ----------
@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html', crops=CROP_LIST, selected="auto")


def extract_features(path):
    raw = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return np.array([tf.image.resize(raw, (IMG_SIZE, IMG_SIZE)).numpy()])

def unsure(cause):
    return {
        "name": "Pata nahi (model ko yaqeen nahi)",
        "cause": cause,
        "cure": "Aik saaf patta, close-up mein, dobara upload karein, ya kisi zarai maahir se poochein.",
    }


def model_predict(path, selected="auto"):
    probs = model.predict(extract_features(path), verbose=0)[0]

    def show(ids):
        return ", ".join(f"{pretty(class_names[i])} {probs[i] * 100:.0f}%" for i in ids)

    overall = probs.argsort()[::-1][:3]
    print("Top 3 (sab fasalein):", show(overall))
    top = int(overall[0])

    if selected in CROP_IDX:
        idx = CROP_IDX[selected]
        share = float(probs[idx].sum())
        order = idx[np.argsort(probs[idx])[::-1][:3]]
        best = int(order[0])
        sub = float(probs[best]) / max(share, 1e-9)
        print(f"Chuni hui fasal {selected}:", show(order), f"| fasal ka hissa {share * 100:.0f}%")

        if CLASS_CROP[top] != selected and float(probs[top]) >= 0.5:
            return unsure(f"Model ko ye photo {pretty(class_names[top])} jaisi lagti hai ({probs[top] * 100:.0f}%), "
                          f"{pretty(selected)} jaisi nahi. Chuni hui fasal check karein.")
        if share < 0.25:
            return unsure(f"Ye photo chuni hui fasal ({pretty(selected)}) jaisi kam lagti hai. "
                          f"Sab se qareeb andaze: {show(overall)}.")
        if sub < MIN_CONFIDENCE:
            return unsure(f"{pretty(selected)} ke andar bhi model sirf {sub * 100:.0f}% sure hai. "
                          f"Sab se qareeb andaze: {show(order)}.")
        result = dict(info_for(class_names[best]))
        result["confidence"] = round(sub * 100, 1)
        return result

    best_prob = float(probs[top])
    if best_prob < MIN_CONFIDENCE:
        return unsure(f"Model sirf {best_prob * 100:.0f}% sure hai. Sab se qareeb andaze: {show(overall)}.")
    result = dict(info_for(class_names[top]))
    result["confidence"] = round(best_prob * 100, 1)
    return result


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        selected = request.form.get("crop", "auto")
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        image.save(f'{temp_name}_{image.filename}')
        print(f'{temp_name}_{image.filename}')
        prediction = model_predict(f'./{temp_name}_{image.filename}', selected)
        return render_template('home.html', result=True,
                               imagepath=f'/{temp_name}_{image.filename}',
                               prediction=prediction,
                               crops=CROP_LIST, selected=selected)
    else:
        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)