class FSBase:
    def __init__(self):
        # Default sampling frequency (fs) set to 500.0 Hz
        self._fs = 500.0

    @property
    def fs(self):
        # Getter for the sampling frequency
        return self._fs

    @fs.setter
    def fs(self, value):
        # Setter to allow updating the sampling frequency
        self._fs = value
