import os, sys
import numpy as np
import tensorflow as tf
from app import model, class_names, CLASS_CROP, CROP_IDX

folder = sys.argv[1] if len(sys.argv) > 1 else "google_test"
IMG = (".jpg", ".jpeg", ".png")
n = t1 = t3 = tc = 0
print(f"{'photo':34s}{'asli':26s}{'sab fasalein':26s}{'fasal chun kar':26s}")
for f in sorted(os.listdir(folder)):
    if not f.lower().endswith(IMG) or "__" not in f:
        continue
    true = f.split("__")[0]
    if true not in class_names:
        print("SKIP (naam class_names mein nahi):", f)
        continue
    ti = class_names.index(true)
    img = tf.keras.utils.load_img(os.path.join(folder, f), target_size=(224, 224))
    p = model.predict(np.array([tf.keras.utils.img_to_array(img)]), verbose=0)[0]
    order = p.argsort()[::-1]
    idx = CROP_IDX.get(CLASS_CROP[ti])
    ci = int(idx[np.argmax(p[idx])]) if idx is not None else int(order[0])
    ok1, ok3, okc = order[0] == ti, ti in order[:3], ci == ti
    n += 1; t1 += ok1; t3 += ok3; tc += okc
    print(f"{f[:33]:34s}{true[:25]:26s}{class_names[order[0]][:20]:20s}{'ok' if ok1 else ('top3' if ok3 else 'X'):>5s}   {class_names[ci][:20]:20s}{'ok' if okc else 'X'}")

if n:
    print(f"\n{n} photos | sab fasalein: top1 {t1/n:.0%}, top3 {t3/n:.0%} | fasal chun kar (website jaisa): {tc/n:.0%}")