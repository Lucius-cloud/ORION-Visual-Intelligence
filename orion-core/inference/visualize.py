import cv2

def show_result(image_path, label, confidence):
    img = cv2.imread(image_path)
    text = f"{label}: {confidence:.2f}"
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow("ORION", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
