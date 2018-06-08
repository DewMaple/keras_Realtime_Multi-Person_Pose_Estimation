import numpy as np


class BodySkeleton:

    def __init__(self):
        """
        nose and neck is a tuple of 2D coordination (x, y) in image, other parts like eyes are dicts with keys right and
        left, which hold two body joints, such as eyes, eyes.right is right eye, eyes.left is left eye, if negative,
        the value is None
        """
        self.nose = None
        self.neck = None

        self.eyes = {'right': None, 'left': None}
        self.ears = {'right': None, 'left': None}
        self.shoulders = {'right': None, 'left': None}
        self.elbows = {'right': None, 'left': None}
        self.wrists = {'right': None, 'left': None}
        self.hips = {'right': None, 'left': None}
        self.knees = {'right': None, 'left': None}
        self.ankles = {'right': None, 'left': None}

    def _add_attributes_validation(self):
        attributes = list(vars(self).keys())
        for attr in attributes:
            setattr(self, '{}_is_valid'.format(attr), self.is_valid(getattr(self, attr)))
            if attr not in ['nose', 'neck']:
                setattr(self, '{}_right'.format(attr), getattr(self, attr)['right'])
                setattr(self, '{}_left'.format(attr), getattr(self, attr)['left'])
                setattr(self, '{}_right_is_valid'.format(attr), self.is_valid(getattr(self, attr)['right']))
                setattr(self, '{}_left_is_valid'.format(attr), self.is_valid(getattr(self, attr)['left']))

    @classmethod
    def from_18_joints(cls, joints):
        """
        Build BodySkeleton instance by 18 pose body parts
        detail to see OpenPose, https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        :param joints: 18 pose body parts
        :return: BodySkeleton instance
        """
        assert len(joints) == 18
        body_skeleton = cls()
        body_skeleton.nose = joints[0]
        body_skeleton.neck = joints[1]

        body_skeleton.eyes['right'] = joints[14]
        body_skeleton.eyes['left'] = joints[15]

        body_skeleton.ears['right'] = joints[17]
        body_skeleton.ears['left'] = joints[16]

        body_skeleton.shoulders['right'] = joints[2]
        body_skeleton.shoulders['left'] = joints[5]

        body_skeleton.elbows['right'] = joints[3]
        body_skeleton.elbows['left'] = joints[6]

        body_skeleton.wrists['right'] = joints[4]
        body_skeleton.wrists['left'] = joints[7]

        body_skeleton.hips['right'] = joints[8]
        body_skeleton.hips['left'] = joints[11]

        body_skeleton.knees['right'] = joints[9]
        body_skeleton.knees['left'] = joints[12]

        body_skeleton.ankles['right'] = joints[10]
        body_skeleton.ankles['left'] = joints[13]
        body_skeleton._add_attributes_validation()
        return body_skeleton

    @classmethod
    def from_limbs(cls, limbs):
        """
        :param limbs: array of body limbs, array size should be 17, connected joints
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
        :return: BodySkeleton instance
        """

        def _parse_neck():
            all_neck = [limbs[i][0] for i in [0, 1, 6, 9, 12] if
                        len(limbs[i]) > 0 and limbs[i][0] is not None]
            np_neck = np.array(all_neck)
            return np.mean(np_neck, axis=0) if len(np_neck) > 0 else None

        def _parse_2_times_overlapped(index1, index2):
            points = []
            if len(limbs[index1]) > 0 and limbs[index1][1] is not None:
                points.append(limbs[index1][1])
            if len(limbs[index2]) > 0 and limbs[index2][0] is not None:
                points.append(limbs[index2][0])
            return np.mean(np.array(points), axis=0) if len(points) > 0 else None

        def _parse_nose():
            points = []
            if len(limbs[12]) > 0 and limbs[12][1] is not None:
                points.append(limbs[12][1])
            if len(limbs[13]) > 0 and limbs[13][0] is not None:
                points.append(limbs[13][0])
            if len(limbs[15]) > 0 and limbs[15][0] is not None:
                points.append(limbs[15][0])
            return np.mean(np.array(points), axis=0) if len(points) > 0 else None

        body_skeleton = cls()
        body_skeleton.nose = _parse_nose()
        body_skeleton.neck = _parse_neck()

        body_skeleton.shoulders['right'] = _parse_2_times_overlapped(0, 2)
        body_skeleton.shoulders['left'] = _parse_2_times_overlapped(1, 4)

        body_skeleton.elbows['right'] = _parse_2_times_overlapped(2, 3)
        body_skeleton.elbows['left'] = _parse_2_times_overlapped(4, 5)

        if len(limbs[3]) > 0:
            body_skeleton.wrists['right'] = limbs[3][1]

        if len(limbs[5]) > 0:
            body_skeleton.wrists['left'] = limbs[5][1]

        body_skeleton.hips['right'] = _parse_2_times_overlapped(6, 7)
        body_skeleton.hips['left'] = _parse_2_times_overlapped(9, 10)

        body_skeleton.knees['right'] = _parse_2_times_overlapped(7, 8)
        body_skeleton.knees['left'] = _parse_2_times_overlapped(10, 11)

        if len(limbs[8]) > 0:
            body_skeleton.ankles['right'] = limbs[8][1]

        if len(limbs[11]) > 0:
            body_skeleton.ankles['left'] = limbs[11][1]

        body_skeleton.eyes['right'] = _parse_2_times_overlapped(13, 14)
        body_skeleton.eyes['left'] = _parse_2_times_overlapped(15, 16)

        if len(limbs[14]) > 0:
            body_skeleton.ears['right'] = limbs[14][1]

        if len(limbs[16]) > 0:
            body_skeleton.ears['left'] = limbs[16][1]

        body_skeleton._add_attributes_validation()
        return body_skeleton

    @staticmethod
    def is_valid(joint):
        if isinstance(joint, dict):
            return joint['right'] is not None and joint['left'] is not None
        return joint is not None

    def minimum_bounding_box(self):
        """
        :return: the minimum bounding box of all these body parts, points actually
        """
        all_parts = []
        attributes = list(vars(self).keys())
        for attr in attributes:
            if attr.endswith('_right_is_valid'):
                real_attr = attr.split('_')[0]
                valid = getattr(self, attr)
                if valid:
                    all_parts.append(getattr(self, '{}_right'.format(real_attr)))
            if attr.endswith('_left_is_valid'):
                real_attr = attr.split('_')[0]
                valid = getattr(self, attr)
                if valid:
                    all_parts.append(getattr(self, '{}_left'.format(real_attr)))
        if self.is_valid(self.nose):
            all_parts.append(self.nose)
        if self.is_valid(self.neck):
            all_parts.append(self.neck)

        min_x, min_y, max_x, max_y = max_rectangle(all_parts)

        return min_x, min_y, max_x, max_y


def higher(pnt1, pnt2):
    """
    :param pnt1: point A (x, y)
    :param pnt2: point B (x, y)
    :return: if point A is higher than point B in image
    """
    if np.isclose(pnt1[1], pnt2[1], rtol=0.05, atol=0.05):
        return False

    return pnt1[1] < pnt2[1]


def max_rectangle(pose_body_parts):
    np_array = np.array(pose_body_parts)
    all_x = np_array[:, 0]
    all_y = np_array[:, 1]
    max_x = np.amax(all_x)
    max_y = np.amax(all_y)

    min_x = np.amin(all_x)
    min_y = np.amin(all_y)
    return min_x, min_y, max_x, max_y


def angle_2_vertical(pnt1, pnt2):
    """
    :param pnt1: point A (x, y)
    :param pnt2: point B (x, y)
    :return: the angle between the line which cross point A and point B and the vertical line cross point A
    """
    dist = np.linalg.norm(pnt1 - pnt2)
    if np.isclose(dist, 0):
        return 0
    x_dist = abs(pnt1[0] - pnt2[0])
    return np.arcsin(float(x_dist) / float(dist)) / np.pi * 180
