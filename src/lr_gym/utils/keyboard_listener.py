import sshkeyboard
import threading
import lr_gym.utils.dbg.ggLog as ggLog
import atexit
import copy

class KeyboardListener():
    def __init__(self):
        self._listener_thread = threading.Thread(target=self._listener)
        self._listener_thread.start()
        self._started = True
        atexit.register(self.close)

        self._dict_lock = threading.RLock()
        self._pressed_keys = set()
        
    def _on_press(self, key):
        with self._dict_lock:
            self._pressed_keys.add(key)
            # ggLog.info(f"added {key}")

    def _on_release(self, key):
        with self._dict_lock:
            try:
                self._pressed_keys.remove(key)
                # ggLog.info(f"removed {key}")
            except KeyError as e:
                ggLog.warn(f"Release unpressed key '{key}'")
    
    def get_pressed_keys(self):
        return copy.deepcopy(self._pressed_keys)

    def _listener(self):
        # ggLog.info(f"starting listener")
        sshkeyboard.listen_keyboard(on_press=self._on_press,
                                    on_release=self._on_release,
                                    until=None,
                                    delay_second_char=0.05)
        # ggLog.info("stoppping listener")
        
    def close(self):
        if self._started:
            sshkeyboard.stop_listening()
            self._listener_thread.join(timeout=30)
            if self._listener_thread.is_alive():
                ggLog.error(f"Failed to stop keyboard listener.")
            else:
                self._started = False