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
            all_neck = [connected_body_parts[i][0] for i in [0, 1, 6, 9, 12] if
                        len(connected_body_parts[i]) > 0 and connected_body_parts[i][0] is not None]
            np_neck = np.array(all_neck)
            return np.mean(np_neck, axis=0) if len(np_neck) > 0 else None

        def _parse_2_times_overlapped(index1, index2):
            points = []
            if len(connected_body_parts[index1]) > 0 and connected_body_parts[index1][1] is not None:
                points.append(connected_body_parts[index1][1])
            if len(connected_body_parts[index2]) > 0 and connected_body_parts[index2][0] is not None:
                points.append(connected_body_parts[index2][0])
            return np.mean(np.array(points), axis=0) if len(points) > 0 else None

        def _parse_nose():
            points = []
            if len(connected_body_parts[12]) > 0 and connected_body_parts[12][1] is not None:
                points.append(connected_body_parts[12][1])
            if len(connected_body_parts[13]) > 0 and connected_body_parts[13][0] is not None:
                points.append(connected_body_parts[13][0])
            if len(connected_body_parts[15]) > 0 and connected_body_parts[15][0] is not None:
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

    def _append_eye(self, points):
        if self.left_eye is not None:
            points.append([self.left_eye[1], self.left_eye[0]])
        if self.right_eye is not None:
            points.append([self.right_eye[1], self.right_eye[0]])
        return points

    def head_direction(self, img, offset2nose=50):
        if self.nose is not None:
            x2 = self.nose[1]
            y2 = self.nose[0]
            if self.right_ear is not None and self.left_ear is not None:
                x_r = self.right_ear[1]
                y_r = self.right_ear[0]
                x_l = self.left_ear[1]
                y_l = self.left_ear[0]
                points = [[x_r, y_r], [x_l, y_l]]

                x1, y1 = _centroid(points)

                points = [[x2, y2]]
                if self.left_eye is not None and self.right_eye is not None:
                    points.append([self.left_eye[1], self.left_eye[0]])
                    points.append([self.right_eye[1], self.right_eye[0]])

                x2, y2 = _centroid(points)
                x0, y0 = _x0_y0(x1, y1, x2, y2, offset2nose)

                cv2.line(img, (int(x2), int(y2)), (int(x0), int(y0)), (255, 0, 255), 2)
                # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            elif self.right_ear is not None:
                x1 = self.right_ear[1]
                y1 = self.right_ear[0]

                points = [[x2, y2]]
                if self.right_eye is not None:
                    points.append([self.right_eye[1], self.right_eye[0]])
                x2, y2 = _centroid(points)

                x0, y0 = _x0_y0(x1, y1, x2, y2, offset2nose)
                cv2.line(img, (int(x2), int(y2)), (int(x0), int(y0)), (255, 0, 255), 2)
                # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            elif self.left_ear is not None:
                x1 = self.left_ear[1]
                y1 = self.left_ear[0]

                points = [[x2, y2]]
                if self.left_eye is not None:
                    points.append([self.left_eye[1], self.left_eye[0]])

                x2, y2 = _centroid(points)
                x0, y0 = _x0_y0(x1, y1, x2, y2, offset2nose)
                cv2.line(img, (int(x2), int(y2)), (int(x0), int(y0)), (255, 0, 255), 2)
                # cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        return img

    def is_hands_up(self, threshold=45):
        return self.left_hand_up(threshold=threshold) or self.right_hand_up(threshold=threshold)

    # def is_stand(self):
    #     return self.right_leg_stand() or self.left_leg_stand()
    #
    # def right_leg_stand(self):
    #     if self.right_hip is not None and self.right_ankle is not None:
    #         if higher(self.right_hip, self.right_ankle):
    #             return angle_2_vertical(self.right_hip, self.right_ankle) < 5
    #     return False
    #
    # def left_leg_stand(self):
    #     if self.left_hip is not None and self.left_ankle is not None:
    #         if higher(self.left_hip, self.left_ankle):
    #             return angle_2_vertical(self.left_hip, self.left_ankle) < 5
    #     return False

    def head_key_points(self):
        if self.nose is not None and self.right_eye is not None and self.left_eye is not None and \
                self.right_ear is not None and self.left_ear is not None:
            return np.array([self.nose, self.right_eye, self.left_eye, self.right_ear, self.left_ear], dtype=np.double)
        else:
            return None

    def nose_2eyes(self):
        if self.nose is not None and self.right_eye is not None and self.left_eye is not None:
            return np.array([self.nose, self.right_eye, self.left_eye], dtype=np.double)
        else:
            return None

    def mbr(self):
        """
        :return: the minimum bounding box of all these body parts, points actually
        """
        all_parts = []
        if self.nose is not None:
            all_parts.append(self.nose)
        if self.neck is not None:
            all_parts.append(self.neck)
        if self.left_eye is not None:
            all_parts.append(self.left_eye)
        if self.right_eye is not None:
            all_parts.append(self.right_eye)
        if self.left_ear is not None:
            all_parts.append(self.left_ear)
        if self.right_ear is not None:
            all_parts.append(self.right_ear)
        if self.left_shoulder is not None:
            all_parts.append(self.left_shoulder)
        if self.left_elbow is not None:
            all_parts.append(self.left_elbow)
        if self.left_wrist is not None:
            all_parts.append(self.left_wrist)
        if self.right_shoulder is not None:
            all_parts.append(self.right_shoulder)
        if self.right_elbow is not None:
            all_parts.append(self.right_elbow)
        if self.right_wrist is not None:
            all_parts.append(self.right_wrist)
        if self.left_hip is not None:
            all_parts.append(self.left_hip)
        if self.left_knee is not None:
            all_parts.append(self.left_knee)
        if self.left_ankle is not None:
            all_parts.append(self.left_ankle)
        if self.right_hip is not None:
            all_parts.append(self.right_hip)
        if self.right_knee is not None:
            all_parts.append(self.right_knee)
        if self.right_ankle is not None:
            all_parts.append(self.right_ankle)

        return max_rectangle(all_parts)

    def left_hand_up(self, threshold=45):
        if self.left_wrist is not None and self.left_elbow is not None:
            print('left wrist: {}'.format(self.left_wrist))
            print('left elbow: {}'.format(self.left_elbow))
            print('==========={}========='.format('self.left_wrist is not None and self.left_elbow is not None'))

            return higher(self.left_wrist, self.left_elbow) and angle_2_vertical(self.left_wrist,
                                                                                 self.left_elbow) < threshold
        elif self.left_wrist is not None and self.left_shoulder is not None:
            print('==========={}========='.format('self.left_wrist is not None and self.left_shoulder is not None'))
            return higher(self.left_wrist, self.left_shoulder) and angle_2_vertical(self.left_wrist,
                                                                                    self.left_shoulder) < threshold
        elif self.left_elbow is not None and self.left_shoulder is not None:
            print('==========={}========='.format('self.left_elbow is not None and self.left_shoulder is not None'))
            return higher(self.left_elbow, self.left_shoulder) and angle_2_vertical(self.left_elbow,
                                                                                    self.left_shoulder) < threshold
        elif self.left_elbow is not None and self.nose is not None:
            print('==========={}========='.format('self.left_elbow is not None and self.nose is not None'))
            return higher(self.left_elbow, self.nose)
        elif self.left_wrist is not None and self.nose is not None:
            print('==========={}========='.format('self.left_wrist is not None and self.nose is not None'))
            return higher(self.left_wrist, self.nose)
        else:
            print('==========={}========='.format('All is None'))
            return False

    def right_hand_up(self, threshold=45):
        if self.right_wrist is not None and self.right_elbow is not None:
            print('==========={}========='.format('self.right_wrist is not None and self.right_elbow is not None'))
            print('right_wrist : {}'.format(self.right_wrist))
            print('right_elbow : {}'.format(self.right_elbow))
            return higher(self.right_wrist, self.right_elbow) and angle_2_vertical(self.right_wrist,
                                                                                   self.right_elbow) < threshold
        elif self.right_wrist is not None and self.right_shoulder is not None:
            print('==========={}========='.format('self.right_wrist is not None and self.right_shoulder is not None'))
            return higher(self.right_wrist, self.right_shoulder) and angle_2_vertical(self.right_wrist,
                                                                                      self.right_shoulder) < threshold
        elif self.right_elbow is not None and self.right_shoulder is not None:
            print('==========={}========='.format('self.right_elbow is not None and self.right_shoulder is not None'))
            return higher(self.right_elbow, self.right_shoulder) and angle_2_vertical(self.right_elbow,
                                                                                      self.right_shoulder) < threshold
        elif self.right_elbow is not None and self.nose is not None:
            print('==========={}========='.format('self.right_elbow is not None and self.nose is not None'))
            return higher(self.right_elbow, self.nose)
        elif self.right_wrist is not None and self.nose is not None:
            print('==========={}========='.format('self.right_wrist is not None and self.nose is not None'))
            return higher(self.right_wrist, self.nose)
        else:
            print('==========={}========='.format('All is None'))
            return False


def _x0_y0(x1, y1, x2, y2, offset=50):
    sita = np.arctan2(y2 - y1, x2 - x1)
    y0 = np.sin(sita) * offset + y2
    x0 = np.cos(sita) * offset + x2
    return x0, y0


def _centroid(points):
    """
    :param points: points list, or ndarray
    :return: centroid point
    """
    points = np.array(points)
    size, rank = points.shape
    if size == 1:
        point = points.tolist()
        return [point[0], point[1]]

    if rank < 2:
        raise ValueError('Points should be 2D point list or 3D point list')

    xs = points[:, 0]
    ys = points[:, 1]

    if len(points[0]) == 3:
        zs = points[:, 2]
        return [np.mean(xs), np.mean(ys), np.mean(zs)]
    else:
        return [np.mean(xs), np.mean(ys)]


def angle_2_vertical(pnt1, pnt2):
    """
    :param pnt1: point A (x, y)
    :param pnt2: point B (x, y)
    :return: the angle between the line which cross point A and point B and the vertical line cross point A
    """
    dist = np.linalg.norm(pnt1 - pnt2)
    if np.isclose(dist, 0):
        return 0
    x_dist = abs(pnt1[1] - pnt2[1])
    return np.arcsin(float(x_dist) / float(dist)) / np.pi * 180


def higher(pnt1, pnt2):
    """
    :param pnt1: point A (x, y)
    :param pnt2: point B (x, y)
    :return: if point A is higher than point B in image
    """
    if np.isclose(pnt1[0], pnt2[0], rtol=0.05, atol=0.05):
        return False

    return pnt1[0] < pnt2[0]


def max_rectangle(pose_body_parts):
    np_array = np.array(pose_body_parts)
    print('-xxxxx-: {}'.format(np_array))
    all_y = np_array[:, 0]
    all_x = np_array[:, 1]
    max_x = np.amax(all_x)
    max_y = np.amax(all_y)

    min_x = np.amin(all_x)
    min_y = np.amin(all_y)

    return min_x, min_y, max_x, max_y


def convert_2_int_tuple(arr):
    np_array = np.array(arr).astype(np.int)
    y = np_array[0]
    x = np_array[1]

    return x, y


def draw_body_pose_key_points(body_pose, canvas, colors):
    if body_pose.nose is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.nose), 2, colors[0], thickness=-1)
    if body_pose.right_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_eye), 2, colors[1], thickness=-1)
    if body_pose.right_ear is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_ear), 2, colors[2], thickness=-1)
    if body_pose.left_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_eye), 2, colors[3], thickness=-1)
    if body_pose.left_ear is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_ear), 2, colors[4], thickness=-1)
    if body_pose.neck is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.neck), 2, colors[5], thickness=-1)
    if body_pose.right_shoulder is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_shoulder), 2, colors[6], thickness=-1)
    if body_pose.right_elbow is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_elbow), 2, colors[7], thickness=-1)
    if body_pose.right_wrist is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_wrist), 2, colors[8], thickness=-1)
    if body_pose.left_shoulder is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_shoulder), 2, colors[9], thickness=-1)
    if body_pose.left_elbow is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_elbow), 2, colors[10], thickness=-1)
    if body_pose.left_wrist is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_wrist), 2, colors[11], thickness=-1)
    if body_pose.right_hip is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_hip), 2, colors[12], thickness=-1)
    if body_pose.right_knee is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_knee), 2, colors[13], thickness=-1)
    if body_pose.right_ankle is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_ankle), 2, colors[14], thickness=-1)
    if body_pose.left_hip is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_hip), 2, colors[15], thickness=-1)
    if body_pose.left_knee is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_knee), 2, colors[16], thickness=-1)
    if body_pose.left_ankle is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_ankle), 2, colors[17], thickness=-1)
    return canvas


def draw_nose_2eyes(canvas, body_pose, colors=None):
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    if body_pose.nose is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.nose), 2, colors[0], thickness=-1)
    if body_pose.right_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.right_eye), 2, colors[1], thickness=-1)
    if body_pose.left_eye is not None:
        cv2.circle(canvas, convert_2_int_tuple(body_pose.left_eye), 2, colors[2], thickness=-1)
    return canvas


def draw_head_pose(img, points):
    cv2.line(img, (points[5][1], points[5][0]), (points[6][1], points[6][0]), (0, 255, 0), thickness=2)
    cv2.line(img, (points[6][1], points[6][0]), (points[7][1], points[7][0]), (255, 0, 0), thickness=2)
    cv2.line(img, (points[2][1], points[2][0]), (points[6][1], points[6][0]), (0, 0, 255), thickness=2)
