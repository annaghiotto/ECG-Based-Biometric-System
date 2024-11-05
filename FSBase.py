class FSBase:
    def __init__(self):
        self._fs = 500.0

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        self._fs = value
