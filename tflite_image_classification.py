from imutils.video import VideoStream, FPS
from tflite_runtime.interpreter import Interpreter, load_delegate
import argparse
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def draw_image(image, result):
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), result, font=ImageFont.truetype("/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 20))
    displayImage = np.asarray( image )
    cv2.imshow( 'Live Inference', displayImage )


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).
    Args:
        path: path to label file.
        encoding: label file encoding.
    Returns:
        Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
        '--label', help='File path of label file.', required=True)
    parser.add_argument( '--picamera',
                         action='store_true',
                         help="Use PiCamera for image capture",
                         default=False)
    parser.add_argument(
                        '-t', '--threshold', type=float, default=0.0,
                        help='Classification score threshold')
    args = parser.parse_args()

    # Prepare labels.
    labels = load_labels(args.label)
    
    # Get interpreter
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    # Initialize video stream
    vs = VideoStream(usePiCamera=args.picamera, resolution=(640, 480)).start()
    time.sleep(1)

    fps = FPS().start()

    while True:
        try:
            # Read frame from video
            screenshot = vs.read()
            image = Image.fromarray(screenshot)

            # Perfrom inference and keep time
            start_time = time.time()
            image_pred = image.resize((width ,height), Image.ANTIALIAS)
            results = classify_image(interpreter, image_pred)
            result = labels[results[0][0]]
            print(result)
            draw_image(image, result)

            if( cv2.waitKey( 5 ) & 0xFF == ord( 'q' ) ):
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