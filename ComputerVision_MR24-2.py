import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_mean(img, ksize):
    offset = ksize - 1

    np_img = np.array(img)

    for i in range(np_img.shape[0] - offset):
        for j in range(np_img.shape[1] - offset):
            arr = np.array(np_img[i:(i+ksize), j:(j+ksize)]).flatten()
            mean = np.mean(arr)
            np_img[i+(ksize//2), j+(ksize//2)] = mean
    
    return np_img

def image_preprocess():
    img = cv2.imread('image_preprocess/floyd.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    chess = cv2.imread('image_preprocess/chess.jpg', cv2.IMREAD_GRAYSCALE)

    _, bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    _, inv_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    _, trunc = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)
    _, tozero = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)
    _, inv_tozero = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)
    _, otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(chess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV', 'OTSU', 'ADAPTIVE']
    images = [gray, bin, inv_bin, trunc, tozero, inv_tozero, otsu, adaptive]

    plt.figure(1, figsize=(8, 8))

    for i, (curr_img, curr_titles) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, (i+1))
        plt.imshow(curr_img,'gray')
        plt.title(curr_titles)
        plt.axis(False)

    plt.show()

    mean = cv2.blur(gray, (11, 11))
    gaussian = cv2.GaussianBlur(gray, (11, 11), 5.0)
    median = cv2.medianBlur(gray, 11)
    bilateral = cv2.bilateralFilter(gray, 5, 150, 150)
    manual = manual_mean(gray, 11)

    titles = ['Original Image','MEAN','GAUSSIAN','MEDIAN','BILATERAL', 'MANUAL']
    images = [gray, mean, gaussian, median, bilateral, manual]
    
    plt.figure(1, figsize=(8, 8))

    for i, (curr_img, curr_titles) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, (i+1))
        plt.imshow(curr_img, 'gray')
        plt.title(curr_titles)
        plt.axis(False)
    
    plt.show()

def edge_detection():
    img = cv2.imread('image_preprocess/floyd.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    w = img.shape[1]

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_uint = np.uint8(np.absolute(laplacian))

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_xy = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    canny = cv2.Canny(gray, 100, 200)

    images = [gray, laplacian, laplacian_uint, sobel_x, sobel_y, sobel_xy, canny]
    titles = ['ori', 'laplacian', 'lap uint', 'sobel x', 'sobel y', 'sobel xy', 'canny']

    plt.figure(figsize=(6, 6))
    for i, (curr_img, curr_titles) in enumerate(zip(images, titles)):
        plt.subplot(3, 3, (i+1))
        plt.imshow(curr_img,'gray')
        plt.title(curr_titles)
        plt.axis(False)
    plt.show()

    kernel = np.array([
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    ])

    ksize = 3

    gray = cv2.GaussianBlur(gray, (3, 3), 5.0)
    gray_sobel = gray.copy()
    for i in range(h - ksize - 1):
        for j in range(w - ksize - 1):
            img_matrix = gray[i:(i+ksize), j:(j+ksize)].flatten()
            res = np.convolve(img_matrix, kernel, 'valid')
            gray_sobel[i+(ksize//2), j+(ksize//2)] = res[0]

    plt.imshow(gray_sobel, 'gray')
    plt.axis(False)
    plt.show()

def shape_detection():
    img = cv2.imread('image_preprocess/chess.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        shape = "unidentified"
        sides = len(approx)

        if sides == 3:
            shape = "Triangle"
        elif sides == 4:
            # Check square vs rectangle
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif sides == 5:
            shape = "Pentagon"
        elif sides > 5:
            shape = "Circle"

        cv2.putText(output, shape, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    images = [gray, edges, output]
    titles = ['Gray', 'Canny Edges', 'Detected Shapes']

    plt.figure(figsize=(10, 5))
    for i, (curr_img, curr_title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i+1)
        if len(curr_img.shape) == 2:
            plt.imshow(curr_img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
        plt.title(curr_title)
        plt.axis(False)

    plt.show()

def pattern_recognition():
    img = cv2.imread("image_preprocess/people.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.show()

def detect_face():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]

            filtered = cv2.GaussianBlur(face_roi, (35, 35), 30)
            frame[y:y+h, x:x+w] = filtered

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    while(True):
        print("1. Image Preprocessing")
        print("2. Edge Detection")
        print("3. shape Detection")
        print("4. Pattern Recognition")
        print("5. Detect Face")
        print("333. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            image_preprocess()
        elif choice ==2:
            edge_detection()
        elif choice ==3:
            shape_detection()
        elif choice ==4:
            pattern_recognition()
        elif choice ==5:
            detect_face()
        elif choice == 333:
            break
        else:
            print("Invalid choice. Please try again.")

main()