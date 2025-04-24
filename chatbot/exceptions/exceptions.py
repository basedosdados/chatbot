class EnvironmentVariableUnset(Exception):
    def __init__(self, message: str):
        super().__init__(message)

class NotInitializedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
