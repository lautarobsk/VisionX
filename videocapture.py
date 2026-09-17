import cv2

cap = cv2.VideoCapture("McP1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 200:400]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    cv2.imshow("Frame", frame)
    cv2.imshow("ROI", roi)
    cv2.imshow("Umbralizada", thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

