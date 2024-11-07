import sshkeyboard
import threading
import adarl.utils.dbg.ggLog as ggLog
import atexit
import copy

class KeyboardListener():
    def __init__(self):
        self._listener_thread = threading.Thread(target=self._listener, name="KeyboardListener")
        self._listener_thread.start()
        self._started = True
        atexit.register(self.close)

        self._dict_lock = threading.RLock()
        self._currently_pressed_keys = set()
        self._key_press_counter : dict[str,int] = {}
        
    def _on_press(self, key):
        with self._dict_lock:
            self._currently_pressed_keys.add(key)
            if key not in self._key_press_counter:
                self._key_press_counter[key] = 0
            self._key_press_counter[key] += 1
            # ggLog.info(f"added {key}")

    def _on_release(self, key):
        with self._dict_lock:
            try:
                self._currently_pressed_keys.remove(key)
                # ggLog.info(f"removed {key}")
            except KeyError as e:
                ggLog.warn(f"Release unpressed key '{key}'")
    
    def get_pressed_keys(self):
        return copy.deepcopy(self._currently_pressed_keys)

    def get_key_press_count(self, key):
        count = self._key_press_counter.get(key, 0)
        if count ==0 and key in self._currently_pressed_keys: # happens if the key is held pressed, and reset_key_press_counters has been called
            self._on_press(key)
            count = 1
        return count
    
    def reset_key_press_counters(self):
        self._key_press_counter = {}

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