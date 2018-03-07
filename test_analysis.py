import cv2
from services.analysis import Analyzer, LandmarkDetector


def analyze(img):
    features, points = LandmarkDetector().detect_facial_feature(img, visible=False)
    pore_img = features['pore_roi']
    wrinkle_img = features['wrinkle_roi']
    skin_img = features['skin_roi']

    # cv2.imshow('Pore RoI', pore_img)
    # cv2.imshow('Wrinkle RoI', wrinkle_img)
    # cv2.imshow('Skin RoI', skin_img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    set_visible = True
    analyzer = Analyzer()
    result_data = {}
    result_data = analyzer.analyze_erythema(skin_img, result_data=result_data, visible=set_visible)
    print(result_data)
    result_data = analyzer.analyze_pore(pore_img, result_data=result_data, visible=set_visible)
    print(result_data)
    result_data = analyzer.analyze_pigmentation(skin_img, result_data=result_data, visible=set_visible)
    result_data = analyzer.analyze_wrinkle(wrinkle_img, result_data=result_data, visible=set_visible)
    result_data = analyzer.calc_total_score(result_data)

    return result_data


image = cv2.imread('images/images.jpg')
print(analyze(image))
