import cv2


class CaptureWindow:
    """
    Wrapper class for the OpenCV :class:`cv2.VideoCapture` class.
    Performs input handling and shows captured data on the screen.
    """
    running = True
    frame = None
    font = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, device_id):
        """
        Initializes the capture window.

        :param device_id: The number identifying the device to use for capturing.
        :type device_id: int
        """
        # noinspection PyArgumentList
        self.capture = cv2.VideoCapture(device_id)

    def next_frame(self):
        """
        Tries to read the next frame from the capture source.

        :return: The function returns a tuple; the first item is a boolean value determining whether
                 the read succeeded.
                 If the read succeeds, the captured image is converted from the RGB color space
                 to the HSV color space and returned as the second item of the tuple.
        :rtype: tuple
        """
        read, frame = self.capture.read()
        if read:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            self.frame = frame
            return read, hsv
        return read, frame

    def show_frame(self):
        """
        Displays the most recent frame on the screen.

        :return: None
        """
        if self.frame is not None:
            cv2.imshow('Shapes', self.frame)

    def stop_capture(self):
        """
        Stops the capture and releases all resources associated with the capture window.

        :return: None
        """
        self.running = False
        self.capture.release()
        cv2.destroyAllWindows()

    def draw_recognized_region(self, region, result, color):
        """
        Draws a recognized shape region on the most recent frame.

        :param color: Color used to draw shape bounding box.
        :type color: tuple
        :param region: The shape region which has been recognized.
        :type region: src.data.types.Region
        :param result: The result classification returned by the classifier used.
        :type result: src.data.types.ShapeType
        :return: None
        """
        cv2.rectangle(self.frame, (region.x1, region.y1), (region.x2, region.y2), color=color, thickness=3)
        cv2.putText(self.frame, result.name, (region.x1 + 10, region.y2 - 10), self.font, 1.5, color, 2)

    def draw_timer(self, index, timer, color):
        """
        Draws the timer value on the screen.

        :param index: Index of the timer's classifier.
        :type index: int
        :param timer: Timer value string.
        :type timer: str
        :param color: Color of the timer.
        :type color: tuple
        :return: None
        """
        cv2.putText(self.frame, timer, (10, self.frame.shape[0] - 10 - index * 34), self.font, 1, color, 2)

    def draw_help(self, color1, color2, color3):
        """
        Draws help with keybindings on the screen.

        :param color1: Color of the first classifier.
        :type color1: tuple
        :param color2: Color of the second classifier.
        :type color2: tuple
        :param color3: Color of the third classifier.
        :type color3: tuple
        :return: None
        """
        x1, x2 = 10, 240
        scale = 0.75
        color = (255, 255, 255)
        cv2.putText(self.frame, "Zmiana klasyfikatora:", (x1, 28), self.font, scale, color, 2)
        cv2.putText(self.frame, "1 - Klasyfikator geometryczny", (x1, 28 * 2), self.font, scale, color1, 2)
        cv2.putText(self.frame, "2 - Siec neuronowa (wektorowa)", (x1, 28 * 3), self.font, scale, color2, 2)
        cv2.putText(self.frame, "3 - Siec neuronowa (konwolucyjna)", (x1, 28 * 4), self.font, scale, color3, 2)
        cv2.putText(self.frame, "Wybor ksztaltu:", (x1, 28 * 6), self.font, scale, color, 2)
        cv2.putText(self.frame, "Z - Kwadrat", (x1, 28 * 7), self.font, scale, color, 2)
        cv2.putText(self.frame, "X - Gwiazda", (x1, 28 * 8), self.font, scale, color, 2)
        cv2.putText(self.frame, "C - Kolo", (x1, 28 * 9), self.font, scale, color, 2)
        cv2.putText(self.frame, "V - Trojkat", (x1, 28 * 10), self.font, scale, color, 2)
        cv2.putText(self.frame, "Proba czasowa:", (x2, 28 * 6), self.font, scale, color, 2)
        cv2.putText(self.frame, "A - Start", (x2, 28 * 7), self.font, scale, color, 2)
        cv2.putText(self.frame, "S - Stop", (x2, 28 * 8), self.font, scale, color, 2)
        cv2.putText(self.frame, "R - Restart", (x2, 28 * 9), self.font, scale, color, 2)
        cv2.putText(self.frame, "D - Debug", (x2, 28 * 12), self.font, scale, color, 2)
        cv2.putText(self.frame, "Q - Wyjscie", (x2, 28 * 13), self.font, scale, color, 2)

    def draw_shape(self, shape):
        """
        Draws the shape name on the screen.

        :param shape: Expected shape.
        :type shape: src.data.types.ShapeType
        :return: None
        """
        cv2.putText(self.frame, shape.name, (10, self.frame.shape[0] - (34 * 3 + 10)), self.font, 1, (255, 255, 255), 2)
