import time
import cv2

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
print("Kamera açıldı mı:", cap.isOpened())

try:
    deadline = time.monotonic() + 10
    first_frame = True

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            if time.monotonic() >= deadline:
                print("10 saniye boyunca görüntü alınamadı.")
                break
            time.sleep(0.1)
            continue

        if first_frame:
            print("İlk görüntü alındı:", frame.shape)
            first_frame = False

        deadline = time.monotonic() + 10
        print(
            f"min={frame.min()}, "
            f"max={frame.max()}, "
            f"ortalama={frame.mean():.2f}"
        )
        cv2.imshow("Kamera testi", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()