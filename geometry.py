import cv2

import numpy as np

limb_sequence = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
                 [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
                 [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]


class BodyPose:

    def __init__(self):
        self.nose = None
        self.neck = None

        self.left_eye = None
        self.right_eye = None

        self.left_ear = None
        self.right_ear = None

        self.left_shoulder = None
        self.left_elbow = None
        self.left_wrist = None

        self.right_shoulder = None
        self.right_elbow = None
        self.right_wrist = None

        self.left_hip = None
        self.left_knee = None
        self.left_ankle = None

        self.right_hip = None
        self.right_knee = None
        self.right_ankle = None

    @classmethod
    def from_18_pose_body_parts(cls, pose_body_parts):
        assert len(pose_body_parts) == 18
        bp = cls()
        bp.nose = pose_body_parts[0]
        bp.neck = pose_body_parts[1]
        bp.left_eye = pose_body_parts[15]
        bp.right_eye = pose_body_parts[14]
        bp.left_ear = pose_body_parts[17]
        bp.right_ear = pose_body_parts[16]
        bp.left_shoulder = pose_body_parts[5]
        bp.left_elbow = pose_body_parts[6]
        bp.left_wrist = pose_body_parts[7]
        bp.right_shoulder = pose_body_parts[2]
        bp.right_elbow = pose_body_parts[3]
        bp.right_wrist = pose_body_parts[4]
        bp.left_hip = pose_body_parts[11]
        bp.left_knee = pose_body_parts[12]
        bp.left_ankle = pose_body_parts[13]
        bp.right_hip = pose_body_parts[8]
        bp.right_knee = pose_body_parts[9]
        bp.right_ankle = pose_body_parts[10]

        # self.eyes = (self.left_eye, self.right_eye)
        # self.ears = (self.left_ear, self.right_ear)
        # self.left_arm = (pose_body_parts[5], pose_body_parts[6], pose_body_parts[7])
        # self.right_arm = (pose_body_parts[2], pose_body_parts[3], pose_body_parts[4])
        # self.left_leg = (pose_body_parts[11], pose_body_parts[12], pose_body_parts[13])
        # self.right_leg = (pose_body_parts[8], pose_body_parts[9], pose_body_parts[10])
        return bp

    @classmethod
    def from_connected_body_parts(cls, connected_body_parts):
        """
        :param connected_body_parts: array of body limbs, array size should be 17,
        [0]:    right acromial,     neck to right shoulder
        [1]:    left acromial,      neck to left shoulder
        [2]:    right arm,          right shoulder to right elbow
        [3]:    right forearm,      right elbow to right wrist
        [4]:    left arm,           left shoulder to left elbow
        [5]:    left forearm,       left elbow to left wrist
        [6]:    right torso,        neck to right hip
        [7]:    right thigh,        right hip to right knee
        [8]:    right leg,          right knee to right ankle
        [9]:    left torso,         neck to left hip
        [10]:   left thigh,         left hip to left knee
        [11]:   left leg,           left knee to left ankle
        [12]:   neck to nose
        [13]:   nose to right eye
        [14]:   right eye to right ear
        [15]:   nose to left eye
        [16]:   left eye to left ear
        :return: instance of BodyPose
        """

        def _parse_neck():
            all_neck = [connected_body_parts[i][0] for i in [0, 1, 6, 9, 12] if len(connected_body_parts[i]) > 0]
            np_neck = np.array(all_neck)
            return np.mean(np_neck, axis=0) if len(np_neck) > 0 else None

        def _parse_2_times_overlapped(index1, index2):
            points = []
            if len(connected_body_parts[index1]) > 0:
                points.append(connected_body_parts[index1][1])
            if len(connected_body_parts[index2]) > 0:
                points.append(connected_body_parts[index2][0])
            return np.mean(np.array(points), axis=0) if len(points) > 0 else None

        def _parse_nose():
            points = []
            if len(connected_body_parts[12]) > 0:
                points.append(connected_body_parts[12][1])
            if len(connected_body_parts[13]) > 0:
                points.append(connected_body_parts[13][0])
            if len(connected_body_parts[15]) > 0:
                points.append(connected_body_parts[15][0])
            return np.mean(np.array(points), axis=0) if len(points) > 0 else None

        bp = cls()
        bp.neck = _parse_neck()
        bp.right_shoulder = _parse_2_times_overlapped(0, 2)

        bp.left_shoulder = _parse_2_times_overlapped(1, 4)

        bp.right_elbow = _parse_2_times_overlapped(2, 3)

        if len(connected_body_parts[3]) > 0:
            bp.right_wrist = connected_body_parts[3][1]

        bp.left_elbow = _parse_2_times_overlapped(4, 5)

        if len(connected_body_parts[5]) > 0:
            bp.left_wrist = connected_body_parts[5][1]

        bp.right_hip = _parse_2_times_overlapped(6, 7)
        bp.right_knee = _parse_2_times_overlapped(7, 8)

        if len(connected_body_parts[8]) > 0:
            bp.right_ankle = connected_body_parts[8][1]

        bp.left_hip = _parse_2_times_overlapped(9, 10)
        bp.left_knee = _parse_2_times_overlapped(10, 11)

        if len(connected_body_parts[11]) > 0:
            bp.left_ankle = connected_body_parts[11][1]

        bp.nose = _parse_nose()

        bp.right_eye = _parse_2_times_overlapped(13, 14)

        if len(connected_body_parts[14]) > 0:
            bp.right_ear = connected_body_parts[14][1]

        bp.left_eye = _parse_2_times_overlapped(15, 16)
        if len(connected_body_parts[16]) > 0:
            bp.left_ear = connected_body_parts[16][1]

        return bp

    def acromial_left(self, (neck, shoulder)):
        self.neck = neck
        self.left_shoulder = shoulder

    def forearm_left(self, (wrist, elbow)):
        self.left_wrist = wrist
        self.left_elbow = elbow

    def is_hands_up(self):
        left_hands_up = _front_arm_vertical(self.left_elbow, self.left_wrist) \
                        and _wrist_not_lower_than_nose(self.nose, self.left_wrist)
        if left_hands_up:
            return True

        right_hands_up = _front_arm_vertical(self.right_elbow, self.right_wrist) \
                         and _wrist_not_lower_than_nose(self.nose, self.right_wrist)
        if right_hands_up:
            return True

        return False

    def mbr(self):
        """
        :return: the minimum bounding box of all these body parts, points actually
        """
        all_parts = [self.nose, self.neck,
                     self.left_eye, self.right_eye,
                     self.left_ear, self.right_ear,
                     self.left_shoulder, self.left_elbow, self.left_wrist,
                     self.right_shoulder, self.right_elbow, self.right_wrist,
                     self.left_hip, self.left_knee, self.left_ankle,
                     self.right_hip, self.right_knee, self.right_ankle
                     ]
        return max_rectangle(all_parts)


def _front_arm_vertical(elbow, wrist, threshold=10):
    wrist_higher_than_elbow = wrist[1] < elbow[1]
    almost_vertical = abs(wrist[0] - elbow[0]) <= threshold
    return wrist_higher_than_elbow and almost_vertical


def _wrist_not_lower_than_nose(nose, wrist, threshold=10):
    almost_same_height = abs(wrist[1] - nose[1]) <= threshold
    return almost_same_height or wrist[1] < nose[1]


def hands_up(pose_body_parts):
    if len(pose_body_parts) is not 18:
        return False
    nose = pose_body_parts[0]
    right_elbow = pose_body_parts[3]
    right_wrist = pose_body_parts[4]

    left_elbow = pose_body_parts[6]
    left_wrist = pose_body_parts[7]

    left_hands_up = _front_arm_vertical(left_elbow, left_wrist) and _wrist_not_lower_than_nose(nose, left_wrist)
    if left_hands_up:
        return True

    right_hands_up = _front_arm_vertical(right_elbow, right_wrist) and _wrist_not_lower_than_nose(nose, right_wrist)
    if right_hands_up:
        return True

    return False


def max_rectangle(pose_body_parts):
    np_array = np.array(pose_body_parts)
    all_x = np_array[:, 0]
    all_y = np_array[:, 1]
    max_x = np.amax(all_x)
    max_y = np.amax(all_y)

    min_x = np.amin(all_x)
    min_y = np.amin(all_y)

    return min_x, min_y, max_x, max_y


def convert_2_int_tuple(arr):
    np_array = np.array(arr).astype(np.int)
    x = np_array[0]
    y = np_array[1]

    return y, x


def draw_body_pose_key_points(body_pose, canvas, colors):
    if body_pose.nose is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.nose), 4, colors[0], thickness=-1)
    if body_pose.right_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_eye), 4, colors[1], thickness=-1)
    if body_pose.right_ear is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_ear), 4, colors[2], thickness=-1)
    if body_pose.left_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_eye), 4, colors[3], thickness=-1)
    if body_pose.left_ear is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_ear), 4, colors[4], thickness=-1)
    if body_pose.neck is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.neck), 4, colors[5], thickness=-1)
    if body_pose.right_shoulder is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_shoulder), 4, colors[6], thickness=-1)
    if body_pose.right_elbow is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_elbow), 4, colors[7], thickness=-1)
    if body_pose.right_wrist is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_wrist), 4, colors[8], thickness=-1)
    if body_pose.left_shoulder is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_shoulder), 4, colors[9], thickness=-1)
    if body_pose.left_elbow is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_elbow), 4, colors[10], thickness=-1)
    if body_pose.left_wrist is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_wrist), 4, colors[11], thickness=-1)
    if body_pose.right_hip is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_hip), 4, colors[12], thickness=-1)
    if body_pose.right_knee is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_knee), 4, colors[13], thickness=-1)
    if body_pose.right_ankle is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_ankle), 4, colors[14], thickness=-1)
    if body_pose.left_hip is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_hip), 4, colors[15], thickness=-1)
    if body_pose.left_knee is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_knee), 4, colors[16], thickness=-1)
    if body_pose.left_ankle is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_ankle), 4, colors[17], thickness=-1)
    return canvas
