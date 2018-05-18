import cv2


class CaptureWindow:
    """
    Wrapper class for the OpenCV :class:`cv2.VideoCapture` class.
    Performs input handling and shows captured data on the screen.
    """

    highlight_color = (0, 255, 0)
    """The highlight color used to draw boxes and text."""

    def __init__(self, device_number):
        """
        Initializes the capture window.

        :param device_number: The number of the device to use for capturing.
        :type device_number: int
        """
        self.capture = cv2.VideoCapture(device_number)
        self.running = True
        self.frame = None
        # insert your key bindings here
        # the function should be effectively a void callable (no arguments)
        # because if we store the delegates like this they all need to have the same signature
        # you can kinda cheat here by wrapping a function with arguments into a parameterless lambda
        self.key_bindings = {
            'q': self.stop_capture
        }

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

    def process_keys(self):
        """
        Checks all keys associated with input action.
        If one of the keys was pressed, the associated action is performed.

        :return: None
        """
        for key, func in self.key_bindings.items():
            if self._is_key_pressed(key):
                func()

    def stop_capture(self):
        """
        Stops the capture and releases all resources associated with the capture window.

        :return: None
        """
        self.running = False
        self.capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _is_key_pressed(key):
        """
        Determines whether a character key has been pressed.

        :param key: The character key to be checked.
        :type key: char
        :return: True, if the character key is pressed; false otherwise.
        :rtype: bool
        """
        return cv2.waitKey(1) & 0xff == ord(key)

    def draw_recognized_region(self, region, result):
        """
        Draws a recognized shape region on the most recent frame.

        :param region: The shape region which has been recognized.
        :type region: src.data.types.Region
        :param result: The result classification returned by the classifier used.
        :type result: src.data.types.ShapeType
        :return: None
        """
        cv2.rectangle(
            self.frame,
            (region.x1, region.y1),
            (region.x2, region.y2),
            color=self.highlight_color,
            thickness=3
        )
        cv2.putText(
            self.frame,
            result.name,
            (region.x1 + 10, region.y2 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            self.highlight_color,
            2
        )
