import numpy as np
import cv2
import dlib

class FaceDetector(object):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.detector = dlib.get_frontal_face_detector()

    def detect_face(self, image):
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        return self.detector(gray, 1)


class LandmarkExtractor(object):
    def __init__(self, landmark_model_file="shape_predictor_68_face_landmarks.dat"):
        super(LandmarkExtractor, self).__init__()
        self.predictor = dlib.shape_predictor(landmark_model_file)

    def reshape_for_polyline(self, array):
        return np.array(array, np.int32).reshape((-1, 1, 2))

    def detect_landmarks(self, image, faces):
        landmarks = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for face in faces:
            landmarks.append(self.predictor(gray, face).parts())
        return landmarks

    def landmark_drawer(self, faces, image, landmarks):
        black_image = np.zeros(image.shape, np.uint8)

        # perform if there is a face detected
        if len(faces) == 0:
            print("No landmark found")
            return image
        else:
            landmarks = [[p.x, p.y] for p in landmarks]
            jaw = self.reshape_for_polyline(landmarks[0:17])
            left_eyebrow = self.reshape_for_polyline(landmarks[22:27])
            right_eyebrow = self.reshape_for_polyline(landmarks[17:22])
            nose_bridge = self.reshape_for_polyline(landmarks[27:31])
            lower_nose = self.reshape_for_polyline(landmarks[30:35])
            left_eye = self.reshape_for_polyline(landmarks[42:48])
            right_eye = self.reshape_for_polyline(landmarks[36:42])
            outer_lip = self.reshape_for_polyline(landmarks[48:60])
            inner_lip = self.reshape_for_polyline(landmarks[60:68])

            color = (255, 255, 255)
            thickness = 3

            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)

            return cv2.add(image, black_image)

