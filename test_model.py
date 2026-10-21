import sys
import glob
import os
import json
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/plant_model.keras")
print("input:", model.input_shape, "| output:", model.output_shape)
last = model.layers[-1]
print("last layer:", last.__class__.__name__, "| activation:", getattr(last, "activation", None))

with open("plant_disease.json", "r") as f:
    names = [d["name"] for d in json.load(f)]

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    files = glob.glob("uploadimages/*")
    path = max(files, key=os.path.getmtime)
print("image:", path)

img = tf.keras.utils.load_img(path, target_size=(160, 160))
arr = tf.keras.utils.img_to_array(img)

variants = {
    "raw 0-255": arr,
    "divide by 255": arr / 255.0,
    "scale -1 to 1": arr / 127.5 - 1.0,
}

for title, x in variants.items():
    p = model.predict(np.array([x]), verbose=0)[0]
    print("\n==", title, "| sum of outputs:", round(float(p.sum()), 3))
    for i in p.argsort()[-3:][::-1]:
        print(f"   {names[i]}  ->  {p[i] * 100:.1f}%")