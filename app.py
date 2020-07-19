import argparse
import cv2
from inference import Network
from sys import platform

# Path of model
MODEL_XML = "models/face-detection-retail-0005.xml"
prev_face_counter = 0

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    t_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    optional.add_argument("-m", help=m_desc, default=MODEL_XML)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='RED')
    optional.add_argument("-t", help=t_desc, default=0.5)
    args = parser.parse_args()

    return args


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['RED']


def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''

    total_face = 0
    for box in result[0][0]: # Output shape is 1x1x200x7
        conf = box[2]
        if conf >= args.t:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 1)
            total_face += 1

    global prev_face_counter
    if prev_face_counter != total_face:
    	print("{} face(s) detected.".format(total_face))
    	prev_face_counter = total_face

    return frame


def infer_on_video(args):
    # Convert the args for color and threshold
    args.c = convert_color(args.c)
    args.t = float(args.t)

    plugin = Network()

    plugin.load_model(args.m, args.d)
    net_input_shape = plugin.get_input_shape()

    # Get and open cam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Face-Detection_v1")

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            print("Failed to catch frame")
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Perform inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()

            # Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            cv2.imshow("frame", frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
