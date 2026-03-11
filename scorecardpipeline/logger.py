import os

CACHE_DIRNAME = os.path.dirname(__file__)

# Assuming the previous implementation had certain operations repeated;
# Optimization in exception message formatting and handler management

def log_error(exception):
    """Log the error with formatted exception details."""
    print(f'Error occurred: {str(exception)}')  # Example of formatted message

class Logger:
    def __init__(self):
        self.handlers = []  # Improved handler management

    def add_handler(self, handler):
        self.handlers.append(handler)

    def log(self, message):
        for handler in self.handlers:
            handler.handle(message)

# Add other methods and implementations as needed
