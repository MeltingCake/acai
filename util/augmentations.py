import numpy as np


def rotate_4(strokes_gt, strokes_teacher, image, class_name):
    """
    Rotates input strokes creating 3 new classes 90, 180 and 270 degree rotations
    :param strokes_gt: ground truth for loss
    :param strokes_teacher: teacher forcing input sequence
    :param image: input image
    :param class_name: class
    :return: origional + 3 augmented images
    """
    try:
        class_name = class_name.decode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        pass

    rot_90 = np.array([[0., -1., 0., 0., 0.],
                       [1., 0., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1.]])
    flip_y = np.array([[-1., 0., 0., 0., 0.],
                       [0., -1., 0., 0., 0.],
                       [0., 0., 1., 0., 0.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 1.]])

    return [(strokes_gt, strokes_teacher, image, class_name),
            (strokes_gt @ rot_90, strokes_teacher @ rot_90, np.rot90(image, 1), class_name + "-rot90"),
            (strokes_gt @ flip_y, strokes_teacher @ flip_y, np.rot90(image, 2), class_name + "-rot180"),
            (strokes_gt @ flip_y @ rot_90, strokes_teacher @ flip_y @ rot_90, np.rot90(image, 3), class_name + "-rot270")]


def teacher_noise_4(strokes_gt, strokes_teacher, image, class_name):
    try:
        class_name = class_name.decode('utf-8')
    except (UnicodeDecodeError, AttributeError):
        pass

    augmentations = 3
    noise = np.random.normal(loc=0, scale=0.045, size=(augmentations, strokes_teacher.shape[0]-1, 2))

    augmented_teachers = np.repeat(strokes_teacher[np.newaxis, :], repeats=augmentations+1, axis=0)
    augmented_teachers[1:, 1:, :2] += noise
    return [(strokes_gt, augmented_teachers[0], image, class_name),
            (strokes_gt, augmented_teachers[1], image, class_name + "-noise-1"),
            (strokes_gt, augmented_teachers[2], image, class_name + "-noise-2"),
            (strokes_gt, augmented_teachers[3], image, class_name + "-noise-3")]
