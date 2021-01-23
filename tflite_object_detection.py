# based on https://github.com/tensorflow/examples/blob/master/lite/examples/object_detection/raspberry_pi/detect_picamera.py
from imutils.video import VideoStream, FPS
from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
import re
from PIL import Image, ImageDraw, ImageFont
import numpy as np


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


def draw_image(image, results, labels, size):
    result_size = len(results)
    for idx, obj in enumerate(results):
        print(obj)
        # Prepare image for drawing
        draw = ImageDraw.Draw(image)

        # Prepare boundary box
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * size[0])
        xmax = int(xmax * size[0])
        ymin = int(ymin * size[1])
        ymax = int(ymax * size[1])

        # Draw rectangle to desired thickness
        for x in range(0, 4):
            draw.rectangle((ymin, xmin, ymax, xmax), outline=(255, 255, 0))

        # Annotate image with label and confidence score
        display_str = labels[obj['class_id']] + ": " + \
            str(round(obj['score']*100, 2)) + "%"
        draw.text((ymin, xmin), display_str, font=ImageFont.truetype(
            "/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 20))

        displayImage = np.asarray(image)
        cv2.imshow('Coral Live Object Detection', displayImage)


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return Interpreter(
        model_path=model_file,
        experimental_delegates=[
            load_delegate(EDGETPU_SHARED_LIB,
                          {'device': device[0]} if device else {})
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    parser.add_argument('--threshold', help='Score threshold for detected objects.',
                        required=False, type=float, default=0.4)
    parser.add_argument('--picamera', action='store_true', help="Use PiCamera for image capture",
                        default=False)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    _, input_height, input_width, _ = interpreter.get_input_details()[
        0]['shape']

    # Initialize video stream
    vs = VideoStream(usePiCamera=args.picamera, resolution=(640, 480))
    vs.start()
    time.sleep(1)

    fps = FPS().start()

    while True:
        try:
            # Read frame from video
            screenshot = vs.read()
            image = Image.fromarray(screenshot)
            image_pred = image.resize(
                (input_width, input_height), Image.ANTIALIAS)

            # Perfrom inference
            results = detect_objects(interpreter, image_pred, args.threshold)

            draw_image(image, results, labels, image.size)

            if(cv2.waitKey(5) & 0xFF == ord('q')):
                fps.stop()
                break

            fps.update()
        except KeyboardInterrupt:
            fps.stop()
            break

    print("Elapsed time: " + str(fps.elapsed()))
    print("Approx FPS: :" + str(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
    time.sleep(2)


if __name__ == '__main__':
    main()
