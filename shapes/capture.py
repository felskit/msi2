from src.classifiers.geometric import GeometricClassifier
from src.classifiers.network import NetworkClassifier
from src.common.config import config
from src.common.extractor import ShapeExtractor
from src.data.types import ContourProcessingMode, FillMode, ShapeType
from src.utils.capture import CaptureWindow
from timeit import default_timer as timer
import cv2
import numpy as np


def create_time_trial_classifier(model, name, draw, color, fill_mode):
    return {
        'active': False,
        'color': color,
        'draw': draw,
        'frame_counter': 0,
        'model': model,
        'name': name,
        'timer': zero_timer,
        'fill_mode': fill_mode,
        'classify_count': 0,
        'classify_sum': 0,
        'classify_min': 999,
        'classify_max': -999
    }


def switch_drawn_classifier(active_classifiers, index):
    for idx, c in enumerate(active_classifiers):
        c['draw'] = idx == index


def reset_state(active_classifiers):
    for c in active_classifiers:
        c['timer'] = zero_timer
        c['frame_counter'] = 0
        c['classify_count'] = 0
        c['classify_sum'] = 0
        c['classify_min'] = 999
        c['classify_max'] = -999


def format_timer(time_from, time_to):
    elapsed = time_to - time_from
    return str(int(elapsed / 60)).zfill(2) \
       + ':' + str(int(elapsed) % 60).zfill(2) \
       + '.' + str(int((elapsed - int(elapsed)) * 1000))


time_start = None
zero_timer = '00:00.000'
new_timer = zero_timer
mistake_limit = 24
show_help = False
debug_active = False
time_trial_active = False
expected_shape = ShapeType.SQUARE

capture = CaptureWindow(0)
lower = np.array([0, 50, 50])
upper = np.array([15, 255, 255])
image_size = config["image_size"]
extractor = ShapeExtractor(lower, upper, image_size)
geometric_classifier = create_time_trial_classifier(
    model=GeometricClassifier(),
    name='geometric',
    draw=False,
    color=(127, 127, 0),
    fill_mode=FillMode.WHITE_ON_BLACK
)
network_classifier_1d_vec = create_time_trial_classifier(
    model=NetworkClassifier(model_dir="./model/shapes_model_1d_vec.h5", flatten=True),
    name='network-1d-vec',
    draw=False,
    color=(0, 127, 127),
    fill_mode=FillMode.BLACK_ON_WHITE
)
network_classifier_2d_img = create_time_trial_classifier(
    model=NetworkClassifier(model_dir="./model/shapes_model_2d_img.h5", flatten=False),
    name='network-2d-img',
    draw=True,
    color=(127, 0, 127),
    fill_mode=FillMode.BLACK_ON_WHITE
)
classifiers = [geometric_classifier, network_classifier_1d_vec, network_classifier_2d_img]

while capture.running:
    if time_trial_active and time_start is not None:
        new_timer = format_timer(time_from=time_start, time_to=timer())
    read, frame = capture.next_frame()
    if read:
        if show_help:
            capture.draw_help(classifiers[0]['color'], classifiers[1]['color'], classifiers[2]['color'])
        for i, classifier in enumerate(reversed(classifiers)):
            regions = extractor.get_regions(
                frame,
                fill_mode=classifier['fill_mode'],
                contour_processing_mode=ContourProcessingMode.MORPHOLOGICAL_CLOSING
            )
            capture.draw_timer(i, classifier['timer'], color=classifier['color'])
            if classifier['active']:
                classifier['timer'] = new_timer
            if regions is not None:
                for region in regions:
                    classify_start = timer()
                    show_preview = debug_active & classifier['draw']
                    result = classifier['model'].classify(region, verbose=show_preview)
                    classify_end = timer()
                    # TODO: possibly move this to debug_active (but then lose stats when inactive)
                    classify_current = classify_end - classify_start
                    classifier['classify_count'] += 1
                    classifier['classify_sum'] += classify_current
                    if classifier['classify_min'] > classify_current:
                        classifier['classify_min'] = classify_current
                    elif classifier['classify_max'] < classify_current:
                        classifier['classify_max'] = classify_current
                    if debug_active:
                        classify_mean = classifier['classify_sum'] / classifier['classify_count']
                        print(
                            classifier['name']
                            + ': min = ' + str(classifier['classify_min'])
                            + ', mean = ' + str(classify_mean)
                            + ', max = ' + str(classifier['classify_max'])
                        )
                    if result is not None and classifier['draw']:
                        capture.draw_recognized_region(region, result, color=classifier['color'])
                    if time_trial_active:
                        if result != expected_shape:
                            classifier['frame_counter'] += 1
                        else:
                            classifier['frame_counter'] = 0
                        if classifier['frame_counter'] > mistake_limit:
                            classifier['active'] = False
        capture.draw_shape(expected_shape)
        capture.show_frame()
        key_code = cv2.waitKey(1) & 0xff
        if key_code == ord('q'):
            capture.stop_capture()
        elif key_code == ord('1'):
            switch_drawn_classifier(classifiers, 0)
        elif key_code == ord('2'):
            switch_drawn_classifier(classifiers, 1)
        elif key_code == ord('3'):
            switch_drawn_classifier(classifiers, 2)
        elif key_code == ord('a'):
            time_start = timer()
            time_trial_active = True
            for classifier in classifiers:
                classifier['active'] = True
            reset_state(classifiers)
        elif key_code == ord('s'):
            time_start = None
            time_trial_active = False
            for classifier in classifiers:
                classifier['active'] = False
        elif key_code == ord('r'):
            time_start = timer()
            new_timer = zero_timer
            reset_state(classifiers)
        elif key_code == ord('z'):
            expected_shape = ShapeType.SQUARE
        elif key_code == ord('x'):
            expected_shape = ShapeType.STAR
        elif key_code == ord('c'):
            expected_shape = ShapeType.CIRCLE
        elif key_code == ord('v'):
            expected_shape = ShapeType.TRIANGLE
        elif key_code == ord('h'):
            show_help = not show_help
        elif key_code == ord('d'):
            debug_active = not debug_active
