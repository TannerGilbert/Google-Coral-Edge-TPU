# based on https://github.com/google-coral/pycoral/blob/master/examples/classify_image.py
from imutils.video import VideoStream, FPS
import argparse
import time
import cv2
from PIL import Image
import numpy as np

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_image(image, classes, labels):
    image_np = np.asarray(image)
    if len(classes) > 0:
        c = classes[0]
        cv2.putText(image_np, labels.get(c.id, c.id), (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    cv2.imshow('Live Inference', image_np)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
        '--labels', help='File path of label file.', required=True)
    parser.add_argument('--picamera',
                        action='store_true',
                        help="Use PiCamera for image capture",
                        default=False)
    parser.add_argument(
        '-t', '--threshold', type=float, default=0.5,
        help='Classification score threshold')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    size = common.input_size(interpreter)

    # Initialize video stream
    vs = VideoStream(usePiCamera=args.picamera, resolution=(640, 480)).start()
    time.sleep(1)

    fps = FPS().start()

    while True:
        try:
            # Read frame from video
            screenshot = vs.read()
            image = Image.fromarray(screenshot)
            image_pred = image.resize(size, Image.ANTIALIAS)
            common.set_input(interpreter, image_pred)
            interpreter.invoke()
            classes = classify.get_classes(interpreter, 1, args.threshold)

            draw_image(image, classes, labels)

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
