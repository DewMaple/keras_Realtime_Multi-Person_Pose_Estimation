import os
import sys
import time

import cv2
from img_utils.colors import random_colors
from img_utils.files import filename, fname, images_in_dir

from person_detection import draw
from person_detection.estimator import MultiPersonPoseEstimator

estimator = MultiPersonPoseEstimator("model/keras/model.h5")


def predict(image):
    start = time.time()
    persons, bounding_boxes, heat_map = estimator.estimate(image)
    print('Predict finished, time spent: {}'.format(time.time() - start))
    return persons, bounding_boxes, heat_map


def visualization(image, persons, bounding_boxes, heat_map):
    colors = random_colors(len(persons))
    for i, (person, bbox) in enumerate(zip(persons, bounding_boxes)):
        image = draw.draw_person_joints(image, person)
        image = draw.draw_person_limbs(image, person)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), tuple(colors[i]), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(im, bboxes, output_dir, file_name):
    file_name = fname(file_name)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        f_name = os.path.join(output_dir, '{}_{}'.format(file_name, '{0:06d}.jpg'.format(i)))
        cv2.imwrite(f_name, im[y1:y2, x1:x2, :])


def batch(image_dir, output_dir):
    image_files = images_in_dir(image_dir)
    for image_file in image_files:
        process_one_image(image_file, output_dir)


def process_one_image(image_file, output_dir):
    start = time.time()
    im = cv2.imread(image_file)
    file_name = filename(image_file)

    persons, bounding_boxes, heat_map = predict(im)
    # visualization(im, persons, bounding_boxes, heat_map)
    save_results(im, bounding_boxes, output_dir, file_name)
    print('{} done, time spent: {}'.format(file_name, time.time() - start))


def main():
    inputs = args[1]
    outputs = args[2]
    if os.path.isdir(inputs):
        batch(inputs, outputs)
    else:
        process_one_image(inputs, outputs)


if __name__ == '__main__':
    args = sys.argv
    main()
