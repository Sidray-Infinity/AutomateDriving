from get_images import grab_screen, HEIGHT, WIDTH, REGION
from object_detection_script import detect_objects
import cv2
import time

if __name__ == "__main__":

    for i in range(5, 1, -1):
        print(i)
        time.sleep(1)

    while True:
        image = grab_screen(REGION)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        #print(type(image), image.shape)
        image = detect_objects(image)
        cv2.imshow('object detection', cv2.resize(image, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
