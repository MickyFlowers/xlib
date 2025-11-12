import sys
import select
import termios
import tty
import time

class KeyboardReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
    
    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def is_pressed(self, target_key):
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            key = sys.stdin.read(1)
            return key == target_key
        return False

