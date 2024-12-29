import time

class KeyManager:
    """
    A class to manage keyboard inputs, particularly useful for handling key press states and timings.

    Attributes:
        key_states (dict): A dictionary to store the state of each key; True if pressed, otherwise not listed.
        key_timers (dict): A dictionary to store the last time a key was pressed.
        initial_press_interval (float): The minimum interval between consecutive valid presses of the same key.
        UP (int), LEFT (int), DOWN (int), RIGHT (int), ESC (int): Key codes for respective directional and escape keys.
    """

    def __init__(self, initial_press_interval=0.5):
        """
        Initializes the KeyManager with a specified initial press interval.

        Args:
            initial_press_interval (float): The time interval in seconds that must elapse
                                            between consecutive presses to consider them separate.
        """
        self.key_states = {}
        self.key_timers = {}
        self.initial_press_interval = initial_press_interval

        # Key code constants
        self.UP = 82
        self.LEFT = 81
        self.DOWN = 84
        self.RIGHT = 83
        self.ESC = 27

    def press_key(self, key):
        """
        Register a key press. If the key press interval has elapsed since the last press, updates the state.

        Args:
            key (int or str): The key code or character of the key being pressed.
        """
        if isinstance(key, str):
            key = ord(key.lower())
        current_time = time.time()
        if key not in self.key_timers or current_time - self.key_timers[key] >= self.initial_press_interval:
            self.key_states[key] = True
            self.key_timers[key] = current_time

    def is_key_pressed(self, key):
        """
        Checks if a key is currently considered pressed within the allowed press interval.

        Args:
            key (int or str): The key code or character to check.

        Returns:
            bool: True if the key is considered pressed, False otherwise.
        """
        if isinstance(key, str):
            key = ord(key.lower())
        current_time = time.time()
        if key in self.key_states:
            if current_time - self.key_timers[key] < self.initial_press_interval:
                return True
        return False

    def reset_key(self, key):
        """
        Resets the press state and timer for a specified key.

        Args:
            key (int or str): The key code or character whose state is to be reset.
        """
        if isinstance(key, str):
            key = ord(key.lower())
        if key in self.key_states:
            del self.key_states[key]
            del self.key_timers[key]
