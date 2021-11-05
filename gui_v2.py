import numpy as np
import cv2
import datetime

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from timeit import default_timer as timer

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('emotion_model_gpu_50.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral",
                5: "Sad", 6: "Surprised"}
emoji_dist = {0: "emojis/angry.png", 1: "emojis/disgusted.png", 2: "emojis/fearful.png", 3: "emojis/happy.png",
              4: "emojis/neutral.png", 5: "emojis/sad.png", 6: "emojis/surprised.png"}


def main():
    # create display window
    cv2.namedWindow("webcam", cv2.WINDOW_NORMAL)

    # initialize webcam capture object
    cap = cv2.VideoCapture(0)

    # retrieve properties of the capture object
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    fps_sleep = int(1000 / cap_fps)
    print('* Capture width:', cap_width)
    print('* Capture height:', cap_height)
    print('* Capture FPS:', cap_fps, 'ideal wait time between frames:', fps_sleep, 'ms')

    # initialize time and frame count variables
    last_time = datetime.datetime.now()
    frames = 0

    # main loop: retrieves and displays a frame from the camera
    while True:
        start = timer()
        # blocks until the entire frame is read
        success, img = cap.read()
        print("capture time" + str(start - timer()))
        img = cv2.resize(img, (600, 500))
        bounding_box = cv2.CascadeClassifier('venv/lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        print("bounding box time" + str(start - timer()))

        for (x, y, w, h) in num_faces:
            cv2.rectangle(img, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            print("crop time" + str(start - timer()))
            prediction = emotion_model.predict(cropped_img)
            print("detection time" + str(start - timer()))
            maxindex = int(np.argmax(prediction))

            expression_image = cv2.imread(emoji_dist[maxindex], -1)
            x_offset = x
            y_offset = 0 if (y - 75) < 0 else (y - 75)
            y1, y2 = y_offset, y_offset + expression_image.shape[0]
            x1, x2 = x_offset, x_offset + expression_image.shape[1]

            alpha_s = expression_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_s * expression_image[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])

            expression_name = emotion_dict[maxindex]
            cv2.putText(img, expression_name, (x2 + 5, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        frames += 1

        # compute fps: current_time - last_time
        delta_time = datetime.datetime.now() - last_time
        elapsed_time = delta_time.total_seconds()
        cur_fps = np.around(frames / elapsed_time, 1)

        # draw FPS text and display image
        cv2.putText(img, 'FPS: ' + str(cur_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("webcam", img)

        # wait 1ms for ESC to be pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    # release resources
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
