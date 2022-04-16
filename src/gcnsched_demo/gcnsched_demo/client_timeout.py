import time
import threading
from typing import Any, Callable

from rclpy.node import Client, get_logger

class ClientTimeouter:
    def __init__(self, 
                 cli: Client, 
                 timeout: float,
                 success_callback: Callable[[float, Any], None],
                 error_callback: Callable) -> None:
        self.cli = cli
        self.timeout = timeout
        self.success_callback = success_callback
        self.error_callback = error_callback

    def _call_thread(self, req):
        try:
            start = time.time()
            get_logger("ClientTimeouter").info(f"****Calling service {self.cli.srv_name}****")
            fut = self.cli.call_async(req)
            while not fut.done():
                if time.time() - start > self.timeout:
                    self.error_callback(TimeoutError(f"Timeout calling {self.cli.srv_name}!"))
                    fut.cancel()
                    return
                time.sleep(0.01)
            self.success_callback(time.time() - start, fut.result())
        except Exception as e:
            self.error_callback(e)

    def call(self, req) -> None:
        thread = threading.Thread(target=self._call_thread, args=(req,))
        thread.start()
