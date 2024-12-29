import numpy as np
import cv2


class MouseManager:
    """
    A class to manage mouse events in an OpenCV window and report the current mouse position.

    Attributes:
        window_name (str): The name of the OpenCV window to which the mouse callback is attached.
        position (tuple): The current position of the mouse within the window.

    Methods:
        mouse_event_handler(event, x, y, flags, param): Handles mouse events and updates the mouse position.
        get_pos(): Returns the current mouse position.
    """

    def __init__(self, window_name):
        """
        Initializes the MouseManager with a specified window name and sets up mouse event handling.

        Args:
            window_name (str): The name of the OpenCV window.
        """
        self.window_name = window_name
        self.position = (0, 0)  # Default mouse position
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_event_handler)

    def _mouse_event_handler(self, event, x, y, flags, param):
        """
        Callback method for handling mouse events. Updates the mouse position on mouse move events.

        Args:
            event: The type of mouse event.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags: Any relevant flags passed by OpenCV. This parameter is not used in this method.
            param: Extra parameters supplied by OpenCV. This parameter is not used in this method.
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.position = (x, y)

    def get_pos(self):
        """
        Retrieves the current mouse position.

        Returns:
            tuple: The current (x, y) coordinates of the mouse.
        """
        return self.position