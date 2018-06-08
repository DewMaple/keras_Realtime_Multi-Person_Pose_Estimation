import numpy as np
from scipy.ndimage import gaussian_filter

from person_detection.body_skeleton import BodySkeleton
from config_reader import config_reader
from model import get_testing_model
from util import pad_right_down_corner, resize

LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
            [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
            [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]

# PAF_SEQ = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20],
#            [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50],
#            [53, 54], [51, 52], [55, 56], [37, 38], [45, 46]]
PAF_SEQ = [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25],
           [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [28, 29],
           [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27]]
JOINTS_SIZE = 18
params, model_params = config_reader()


class MultiPersonPoseEstimator:
    def __init__(self, weights_file):
        self.model = get_testing_model()
        self.model.load_weights(weights_file)
        self.im_height = 0
        self.im_width = 0

    def _extract_heat_map_and_paf(self, image):
        """
        :param image: target image
        :return: 18 layers heat map for body parts and 1 layer for background, paf,
        detail to see OpenPose, https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        """
        height, width = self.im_height, self.im_width

        multiplier = [x * model_params['boxsize'] / height for x in params['scale_search']]
        multiplier_len = len(multiplier)

        heat_map_average = np.zeros((height, width, 19))
        paf_average = np.zeros((height, width, 38))

        for scale in multiplier:
            image2test = resize(image, fx=scale, fy=scale)
            padded_image, pad = pad_right_down_corner(image2test, model_params['stride'], model_params['padValue'])
            input_img = np.transpose(np.float32(padded_image[:, :, :, np.newaxis]), (3, 0, 1, 2))
            results = self.model.predict(input_img)

            heat_map = np.squeeze(results[1])
            heat_map = resize(heat_map, fx=model_params['stride'], fy=model_params['stride'])
            heat_map = heat_map[:padded_image.shape[0] - pad[2], :padded_image.shape[1] - pad[3], :]
            heat_map = resize(heat_map, output_size=(width, height))
            heat_map_average = heat_map_average + heat_map / multiplier_len

            paf = np.squeeze(results[0])  # output 0 is PAFs
            paf = resize(paf, fx=model_params['stride'], fy=model_params['stride'])
            paf = paf[:padded_image.shape[0] - pad[2], :padded_image.shape[1] - pad[3], :]
            paf = resize(paf, output_size=(width, height))
            paf_average = paf_average + paf / multiplier_len

        return heat_map_average, paf_average

    def _extract_18_joints(self, heat_map):
        """
        filter and merge 19 layers heat map into one
        :param heat_map: 19 layers heat map
        :return:
        """

        all_peaks = []
        peak_counter = 0
        height = self.im_height
        thickness = 0
        for idx in range(JOINTS_SIZE):
            map_ori = heat_map[:, :, idx]
            _map = gaussian_filter(map_ori, sigma=3)

            map_top = np.zeros(_map.shape)
            map_top[1:, :] = _map[:-1, :]
            map_bottom = np.zeros(_map.shape)
            map_bottom[:-1, :] = _map[1:, :]
            map_left = np.zeros(_map.shape)
            map_left[:, 1:] = _map[:, :-1]
            map_right = np.zeros(_map.shape)
            map_right[:, :-1] = _map[:, 1:]

            peaks_binary = np.logical_and.reduce(
                (_map >= map_top, _map >= map_bottom, _map >= map_left, _map >= map_right, _map > params['thre1']))
            peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
            if thickness == 0:
                thickness = _thickness_of_meat(_map, peaks[0], int(height / 2))
            peaks_with_score = [x + (map_ori[x[1], x[0]],) for x in peaks]

            _ids = range(peak_counter, peak_counter + len(peaks))
            peaks_with_score_and_id = [peaks_with_score[i] + (_ids[i],) for i in range(len(_ids))]

            all_peaks.append(peaks_with_score_and_id)
            peak_counter += len(peaks)
        if thickness == 0:
            thickness = 20
        return all_peaks, thickness

    def _extract_17_bones(self, joints, paf):
        mid_num = 10
        special_k = []
        connection_all = []
        image_height = self.im_height
        for k, x in enumerate(PAF_SEQ):
            score_mid = paf[:, :, x]
            limb_start = joints[LIMB_SEQ[k][0] - 1]
            limb_end = joints[LIMB_SEQ[k][1] - 1]
            n_start = len(limb_start)
            n_end = len(limb_end)
            if n_start != 0 and n_end != 0:
                connection_candidate = []
                for i in range(n_start):
                    for j in range(n_end):
                        vec = np.subtract(limb_end[j][:2], limb_start[i][:2])
                        norm = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                        # failure case when 2 body parts overlaps
                        if norm == 0:
                            continue
                        vec = np.divide(vec, norm)

                        start_end = list(zip(np.linspace(limb_start[i][0], limb_end[j][0], num=mid_num),
                                             np.linspace(limb_start[i][1], limb_end[j][1], num=mid_num)))

                        vec_x = np.array([score_mid[int(round(se[1])), int(round(se[0])), 0] for se in start_end])
                        vec_y = np.array([score_mid[int(round(se[1])), int(round(se[0])), 1] for se in start_end])

                        score_middle_points = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])

                        score_with_dist_prior = sum(score_middle_points) / len(score_middle_points) + min(
                            0.5 * image_height / norm - 1, 0)
                        criterion1 = len(np.nonzero(score_middle_points > params['thre2'])[0]) > 0.8 * len(
                            score_middle_points)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append(
                                [i, j, score_with_dist_prior,
                                 score_with_dist_prior + limb_start[i][2] + limb_end[j][2]])

                connection_candidate = sorted(connection_candidate, key=lambda c: c[2], reverse=True)
                connection = np.zeros((0, 5))
                for cc in connection_candidate:
                    i, j, s = cc[0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:

                        connection = np.vstack([connection, [limb_start[i][3], limb_end[j][3], s, i, j]])
                        if len(connection) >= min(n_start, n_end):
                            break

                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])
        # =======================This is just a line to split this huge shit code into two parts========================
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in joints for item in sublist])

        for k in range(len(PAF_SEQ)):
            if k not in special_k:
                part_starts = connection_all[k][:, 0]
                part_ends = connection_all[k][:, 1]
                index_start, index_end = np.array(LIMB_SEQ[k]) - 1

                for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):  # 1:size(subset,1):
                        if subset[j][index_start] == part_starts[i] or subset[j][index_end] == part_ends[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][index_end] != part_ends[i]:
                            subset[j][index_end] = part_ends[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[part_ends[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:  # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][index_end] = part_ends[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[part_ends[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[index_start] = part_starts[i]
                        row[index_end] = part_ends[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])

        index2delete = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                index2delete.append(i)
        subset = np.delete(subset, index2delete, axis=0)

        return subset

    def _calculate_bounding_box(self, persons, thickness):
        boxes = []
        for person in persons:
            x1, y1, x2, y2 = person.minimum_bounding_box()
            x1 = max(x1 - thickness, 0)
            y1 = max(y1 - 1.5 * thickness, 0)
            x2 = min(x2 + thickness, self.im_width)
            y2 = min(y2 + thickness, self.im_height)
            boxes.append((x1, y1, x2, y2))
        return boxes

    def estimate(self, np_img):
        self.im_height, self.im_width = np_img.shape[:2]
        heat_map, paf = self._extract_heat_map_and_paf(np_img)
        joints, thickness = self._extract_18_joints(heat_map)
        connected_joints = self._extract_17_bones(joints, paf)
        persons = _build_body_skeletons(joints, connected_joints)
        bboxes = self._calculate_bounding_box(persons, thickness)
        return persons, bboxes, heat_map


def _thickness_of_meat(filtered_heat_map, nose, half_height):
    assert isinstance(filtered_heat_map, np.ndarray)
    binary = filtered_heat_map > params['thre1']
    thickness = -1
    for i in range(half_height):
        if not binary[nose[1] + i][nose[0]]:
            thickness = i
            break
    return thickness


def _build_body_skeletons(joints, connected_joints):
    persons = []
    candidate = np.array([item for sublist in joints for item in sublist])
    for n in range(len(connected_joints)):
        limbs = []
        for i in range(JOINTS_SIZE - 1):
            index = connected_joints[n][np.array(LIMB_SEQ[i]) - 1]
            if -1 in index:
                limbs.append([])
                continue
            xs = candidate[index.astype(int), 0]
            ys = candidate[index.astype(int), 1]
            limbs.append(((xs[0], ys[0]), (xs[1], ys[1])))
        persons.append(BodySkeleton.from_limbs(limbs))
    return persons
