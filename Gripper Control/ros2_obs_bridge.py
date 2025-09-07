#!/usr/bin/env python3
#source ros2: source /opt/ros/humble/setup.bash
import threading, socket, json
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

HOST = "127.0.0.1"
PORT = 5555           # must match IsaacLab sender

class ObsBridge(Node):
    def __init__(self):
        super().__init__("obs_bridge")
        self.pub = self.create_publisher(Float32MultiArray, "/lift/obs", 10)

        # start TCP server in a background thread
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self.get_logger().info(f"TCP server listening on {HOST}:{PORT}")

    def destroy_node(self):
        self._stop.set()
        return super().destroy_node()

    def _serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen(1)
            s.settimeout(1.0)
            buf = b""
            conn = None
            while not self._stop.is_set():
                if conn is None:
                    try:
                        conn, addr = s.accept()
                        conn.settimeout(1.0)
                        self.get_logger().info(f"Connected: {addr}")
                        buf = b""
                    except socket.timeout:
                        continue
                try:
                    data = conn.recv(65536)
                    if not data:
                        conn.close()
                        conn = None
                        continue
                    buf += data
                    # process newline-delimited JSON
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            pkt = json.loads(line.decode("utf-8"))
                            arr = pkt.get("obs", [])
                            msg = Float32MultiArray()
                            msg.data = [float(x) for x in arr]
                            self.pub.publish(msg)
                        except Exception as e:
                            self.get_logger().warn(f"Bad packet: {e}")
                except socket.timeout:
                    continue
                except Exception as e:
                    self.get_logger().error(f"Socket error: {e}")
                    if conn:
                        conn.close()
                    conn = None

def main():
    rclpy.init()
    node = ObsBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
