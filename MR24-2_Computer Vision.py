import cv2
import numpy as np
import datetime

def edge_detect(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(g, 100, 200)
    return e

def find_shapes(img):
    out = img.copy()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 400: 
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        v = len(approx)
        nm = "?"
        if v == 3: nm = "Triangle"
        elif v == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            nm = "Square" if 0.95 <= ar <= 1.05 else "Rect"
        elif v == 5: nm = "Pentagon"
        elif v == 6: nm = "Hexagon"
        else: nm = "Circle"
        cv2.drawContours(out, [approx], -1, (0,255,0), 2)
        M = cv2.moments(c)
        if M["m00"]:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(out, nm, (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return out

def blur_img(img):
    return cv2.GaussianBlur(img, (21,21), 0)

def gray_img(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def sepia_img(img):
    k = np.array([[0.272, 0.534, 0.131],
                  [0.349, 0.686, 0.168],
                  [0.393, 0.769, 0.189]], dtype="float32")
    s = cv2.transform(img, k)
    return np.clip(s, 0, 255).astype(np.uint8)

def save_face(img, name):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn = f"face_{name}_{ts}.png"
    cv2.imwrite(fn, img)
    print("saved as", fn)

def filter_face(face):
    cv2.imshow("face", face)
    print("b=blur, g=gray, s=sepia, n=skip")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('b'):
            f = blur_img(face)
            cv2.imshow("filtered", f)
            save_face(f, "blur")
            break
        elif k == ord('g'):
            f = gray_img(face)
            cv2.imshow("filtered", f)
            save_face(f, "gray")
            break
        elif k == ord('s'):
            f = sepia_img(face)
            cv2.imshow("filtered", f)
            save_face(f, "sepia")
            break
        elif k in (ord('n'), 27):
            break
    cv2.destroyWindow("face")
    if cv2.getWindowProperty("filtered", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("filtered")

def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    edges_on = False
    shapes_on = False
    faces_on = False
    face_lock = False

    while True:
        ret, frm = cap.read()
        if not ret: break
        disp = frm.copy()

        if edges_on:
            e = edge_detect(frm)
            disp = cv2.addWeighted(disp, 0.8, cv2.cvtColor(e, cv2.COLOR_GRAY2BGR), 0.5, 0)
        if shapes_on:
            disp = find_shapes(disp)
        if faces_on:
            g = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(g, 1.1, 5, minSize=(60,60))
            for (x,y,w,h) in faces:
                cv2.rectangle(disp, (x,y), (x+w, y+h), (0,0,255), 2)
            if len(faces) and not face_lock:
                (fx, fy, fw, fh) = faces[0]
                pad = int(0.1 * fw)
                x1 = max(fx-pad, 0)
                y1 = max(fy-pad, 0)
                x2 = min(fx+fw+pad, frm.shape[1])
                y2 = min(fy+fh+pad, frm.shape[0])
                roi = frm[y1:y2, x1:x2]
                face_lock = True
                filter_face(roi)
            elif not len(faces):
                face_lock = False

        cv2.putText(disp, "e=edge s=shape f=face q=quit", (10,20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow("camera", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        elif k == ord('e'): edges_on = not edges_on
        elif k == ord('s'): shapes_on = not shapes_on
        elif k == ord('f'): faces_on = not faces_on

    cap.release()
    cv2.destroyAllWindows()

main()
