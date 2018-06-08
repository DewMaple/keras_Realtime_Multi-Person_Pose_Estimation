import cv2
import numpy as np
from img_utils.colors import random_colors


def _2_int_tuple(arr):
    np_array = np.array(arr).astype(np.int)
    x = np_array[0]
    y = np_array[1]

    return x, y


def draw_person_joints(canvas, person, colors=None):
    if colors is None:
        colors = np.array(random_colors(18)) * 255
    attributes = list(vars(person).keys())
    count = 0
    for attr in attributes:
        if attr.endswith('_right_is_valid'):
            real_attr = attr.split('_')[0]
            val = getattr(person, '{}_right'.format(real_attr))
            valid = getattr(person, attr)
            if valid:
                cv2.circle(canvas, (int(val[0]), int(val[1])), 2, tuple(colors[count]), 2, lineType=cv2.LINE_AA)
                count += 1
        elif attr.endswith('_left_is_valid'):
            valid = getattr(person, attr)
            if valid:
                real_attr = attr.split('_')[0]
                val = getattr(person, '{}_left'.format(real_attr))
                cv2.circle(canvas, (int(val[0]), int(val[1])), 2, tuple(colors[count]), 2, lineType=cv2.LINE_AA)
                count += 1

    if person.nose_is_valid:
        val = person.nose
        cv2.circle(canvas, (int(val[0]), int(val[1])), 2, tuple(colors[count]), 2, lineType=cv2.LINE_AA)
        count += 1
    if person.neck_is_valid:
        val = person.neck
        cv2.circle(canvas, (int(val[0]), int(val[1])), 2, tuple(colors[count]), 2, lineType=cv2.LINE_AA)
        count += 1
    return canvas


def draw_person_limbs(canvas, person, colors=None):
    if colors is None:
        colors = np.array(random_colors(17)) * 255
    count = 0
    if person.nose_is_valid and person.eyes_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.nose), _2_int_tuple(person.eyes_right), colors[count], 2)
        count += 1
    if person.nose_is_valid and person.eyes_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.nose), _2_int_tuple(person.eyes_left), colors[count], 2)
        count += 1

    if person.eyes_right_is_valid and person.ears_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.eyes_right), _2_int_tuple(person.ears_right), colors[count], 2)
        count += 1

    if person.eyes_left_is_valid and person.ears_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.eyes_left), _2_int_tuple(person.ears_left), colors[count], 2)
        count += 1
    if person.nose_is_valid and person.neck_is_valid:
        cv2.line(canvas, _2_int_tuple(person.nose), _2_int_tuple(person.neck), colors[count], 2)
        count += 1
    if person.neck_is_valid and person.shoulders_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.neck), _2_int_tuple(person.shoulders_right), colors[count], 2)
        count += 1
    if person.neck_is_valid and person.shoulders_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.neck), _2_int_tuple(person.shoulders_left), colors[count], 2)
        count += 1

    if person.shoulders_right_is_valid and person.elbows_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.shoulders_right), _2_int_tuple(person.elbows_right), colors[count], 2)
        count += 1

    if person.shoulders_left_is_valid and person.elbows_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.shoulders_left), _2_int_tuple(person.elbows_left), colors[count], 2)
        count += 1

    if person.elbows_right_is_valid and person.wrists_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.elbows_right), _2_int_tuple(person.wrists_right), colors[count], 2)
        count += 1

    if person.elbows_left_is_valid and person.wrists_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.elbows_left), _2_int_tuple(person.wrists_left), colors[count], 2)
        count += 1

    if person.neck_is_valid and person.hips_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.neck), _2_int_tuple(person.hips_right), colors[count], 2)
        count += 1
    if person.neck_is_valid and person.hips_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.neck), _2_int_tuple(person.hips_left), colors[count], 2)
        count += 1
    if person.hips_right_is_valid and person.knees_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.hips_right), _2_int_tuple(person.knees_right), colors[count], 2)
        count += 1
    if person.hips_left_is_valid and person.knees_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.hips_left), _2_int_tuple(person.knees_left), colors[count], 2)
        count += 1

    if person.knees_right_is_valid and person.ankles_right_is_valid:
        cv2.line(canvas, _2_int_tuple(person.knees_right), _2_int_tuple(person.ankles_right), colors[count], 2)
        count += 1
    if person.knees_left_is_valid and person.ankles_left_is_valid:
        cv2.line(canvas, _2_int_tuple(person.knees_left), _2_int_tuple(person.ankles_left), colors[count], 2)
        count += 1

    return canvas


def draw_nose_2eyes(canvas, body_pose, colors=None):
    if colors is None:
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    if body_pose.nose is not None:
        cv2.circle(canvas, _2_int_tuple(body_pose.nose), 2, colors[0], thickness=-1)
    if body_pose.right_eye is not None:
        cv2.circle(canvas, _2_int_tuple(body_pose.right_eye), 2, colors[1], thickness=-1)
    if body_pose.left_eye is not None:
        cv2.circle(canvas, _2_int_tuple(body_pose.left_eye), 2, colors[2], thickness=-1)
    return canvas


def draw_head_pose(img, points):
    cv2.line(img, (points[5][1], points[5][0]), (points[6][1], points[6][0]), (0, 255, 0), thickness=2)
    cv2.line(img, (points[6][1], points[6][0]), (points[7][1], points[7][0]), (255, 0, 0), thickness=2)
    cv2.line(img, (points[2][1], points[2][0]), (points[6][1], points[6][0]), (0, 0, 255), thickness=2)
