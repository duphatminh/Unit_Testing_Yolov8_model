import unittest
from ultralytics import YOLO

class TestStringMethods(unittest.TestCase):

    def test_predict_model(self):
        model_fpath = 'yolov8s.pt'
        model = YOLO(model_fpath)

        img_fpath = 'person_1.jpg'
        results = model(img_fpath)

        self.assertIsNotNone(results)  # Check if the result is None

        list_results = list(results)
        self.assertIsNotNone(list_results[0])  # Check if the list of results is None

        first_result = list_results[0]

        print(f"Names in the models {first_result.names}")

        list_boxes = list(first_result.boxes)
        box = list_boxes[0]

        index_object = int(box.cls[0])

        self.assertEqual(index_object, 0)

        detected_classname = first_result.names[index_object]
        self.assertEqual(detected_classname, 'person')

        self.assertGreater(box.conf[0], 0.5)

        first_result.save(filename='person_output.jpg')

if __name__ == '__main__':
    unittest.main()