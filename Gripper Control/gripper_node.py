#!/usr/bin/env python3
# gripper_inference_node.py
# ROS2 (rclpy) node that:
#  - subscribes to an environment observation vector (Float64MultiArray)
#  - evaluates an RL policy (TorchScript or RSL-RL checkpoint reconstruction)
#  - maps the last 6 action entries to 6 gripper servos (degrees) and sends to Arduino
#
# Notes:
#  * For best results, export your policy to TorchScript from Isaac Lab ("exported/policy.pt"),
#    since that bundle usually includes the observation normalizer. If you only have the
#    training checkpoint (model_*.pt), we'll reconstruct the actor and (optionally) apply
#    a simple normalizer using provided ~obs_mean/~obs_std params.
#
# Topics:
#   Subscribed:  ~obs          (std_msgs/msg/Float64MultiArray)  # observation vector (len=~obs_dim)
#   Published :  ~actions      (std_msgs/msg/Float64MultiArray)  # full action vector from the policy
#
# Serial protocol:
#   "S,<id>,<angle>[,<pressure>]\n"  with angle in degrees [0,180].
#
# Parameters (declare via ROS2 params or --ros-args):
#   ~policy_path (string): path to TorchScript .pt OR RSL-RL checkpoint .pt  [required]
#   ~obs_dim (int): expected observation size (auto-inferred for TorchScript; default 55)
#   ~use_torchscript (bool): force treating policy_path as TorchScript (default: auto)
#   ~apply_tanh_output (bool): apply tanh to raw policy outputs [-1,1] (default: true)
#
#   ~grip_action_indices (int array, len=6): indices in the action vector used for the gripper
#       default: [7,8,9,10,11,12]  (assuming 7 arm + 6 gripper)
#   ~grip_deg_init (double array, len=6): initial degrees for each gripper joint
#       default: [78,96,90,125,85,80]  (from your GUI init)
#   ~grip_deg_limits (double[6][2]): per-joint [min,max] degree limits; default [[0,180]]*6
#   ~invert_deg (bool array, len=6): if true, send 180-angle for that joint; default [false,true,false,false,false,false]
#   ~grip_action_scale_deg (double): degrees per unit action (delta integration); default 20.0
#
#   ~port (string): serial port path; if empty we auto-detect Arduino (/dev/ttyACM*/ttyUSB*)
#   ~baudrate (int): default 9600
#
#   ~normalize_obs (bool): apply (x-mean)/std using ~obs_mean/~obs_std (default: false)
#   ~obs_mean (double array, len=obs_dim)
#   ~obs_std  (double array, len=obs_dim)
#
#   ~debug (bool): extra logs
#

# Source your ROS2 workspace first
#python3 gripper_node.py  --ros-args -p policy_path:=/path/to/model_1450.pt  -p obs_dim:=55 -p port:=/dev/ttyACM0 -p baudrate:=9600
  
            
             
             


import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import torch
import numpy as np
import serial, serial.tools.list_ports
import time
from threading import Lock

def find_arduino_port():
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        dev = (p.device or "").lower()
        vid = getattr(p, "vid", None)
        if ("arduino" in desc) or ("ttyacm" in dev) or ("ttyusb" in dev) or (vid == 0x2341) or (vid == 0x2a03):
            return p.device
    return None

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def reconstruct_actor_from_rslrl_checkpoint(path: str):
    """
    Returns (nn.Module actor, obs_dim, act_dim) by scanning 'actor.*' weights.
    Assumes architecture: Linear - Tanh - Linear - Tanh - ... - Linear.
    """
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt["model_state_dict"]
    layer_ids = sorted({int(k.split('.')[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
    sizes = [(sd[f"actor.{i}.weight"].shape[1], sd[f"actor.{i}.weight"].shape[0]) for i in layer_ids]
    from collections import OrderedDict
    layers = []
    for li, (inp, out) in enumerate(sizes):
        layers += [(f"lin{li}", torch.nn.Linear(inp, out))]
        if li < len(sizes)-1:
            layers += [(f"tanh{li}", torch.nn.Tanh())]
    actor = torch.nn.Sequential(OrderedDict(layers))
    # load weights
    mapping = {f"lin{li}.weight": sd[f"actor.{i}.weight"] for li, i in enumerate(layer_ids)}
    mapping.update({f"lin{li}.bias":   sd[f"actor.{i}.bias"]   for li, i in enumerate(layer_ids)})
    actor.load_state_dict(mapping, strict=True)
    obs_dim = sizes[0][0]
    act_dim = sizes[-1][1]
    return actor.eval(), obs_dim, act_dim

class GripperInferenceNode(Node):
    def __init__(self):
        super().__init__("gripper_inference_node")

        # ---- Declare parameters ----
        self.declare_parameter("policy_path", "")
        self.declare_parameter("obs_dim", 55)
        self.declare_parameter("use_torchscript", False)
        self.declare_parameter("apply_tanh_output", True)

        self.declare_parameter("grip_action_indices", [7,8,9,10,11,12])
        self.declare_parameter("grip_deg_init", [78.0, 96.0, 90.0, 125.0, 85.0, 80.0])
        self.declare_parameter("grip_deg_limits", [[0.0, 180.0]]*6)
        self.declare_parameter("invert_deg", [False, True, False, False, False, False])
        self.declare_parameter("grip_action_scale_deg", 20.0)

        self.declare_parameter("port", "")
        self.declare_parameter("baudrate", 9600)

        self.declare_parameter("normalize_obs", False)
        self.declare_parameter("obs_mean", [])
        self.declare_parameter("obs_std", [])
        self.declare_parameter("debug", False)

        # ---- Get params ----
        self.policy_path = self.get_parameter("policy_path").get_parameter_value().string_value
        self.obs_dim_param = self.get_parameter("obs_dim").get_parameter_value().integer_value
        self.force_ts = self.get_parameter("use_torchscript").get_parameter_value().bool_value
        self.apply_tanh = self.get_parameter("apply_tanh_output").get_parameter_value().bool_value

        self.grip_idxs = list(self.get_parameter("grip_action_indices").get_parameter_value().integer_array_value)
        self.grip_deg = list(self.get_parameter("grip_deg_init").get_parameter_value().double_array_value)
        self.grip_limits = [list(x.double_array_value) for x in self.get_parameter("grip_deg_limits").get_parameter_value().double_array_value]
        self.invert_deg = list(self.get_parameter("invert_deg").get_parameter_value().bool_array_value)
        self.grip_scale_deg = float(self.get_parameter("grip_action_scale_deg").get_parameter_value().double_value)

        self.port = self.get_parameter("port").get_parameter_value().string_value
        self.baudrate = int(self.get_parameter("baudrate").get_parameter_value().integer_value)

        self.normalize_obs = self.get_parameter("normalize_obs").get_parameter_value().bool_value
        self.obs_mean = list(self.get_parameter("obs_mean").get_parameter_value().double_array_value)
        self.obs_std  = list(self.get_parameter("obs_std").get_parameter_value().double_array_value)
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        if len(self.grip_deg) != 6 or len(self.grip_idxs) != 6 or len(self.invert_deg) != 6:
            self.get_logger().fatal("grip_deg_init, grip_action_indices, and invert_deg must be length 6")
            raise SystemExit(1)
        if len(self.grip_limits) != 6:
            self.get_logger().fatal("grip_deg_limits must be a 6x2 array")
            raise SystemExit(1)

        # ---- Load policy ----
        self.policy_kind = None
        self.policy = None
        self.obs_dim_model = None
        self.act_dim = None
        self._load_policy(self.policy_path)

        # ---- Serial ----
        self.ser = None
        self.ser_lock = Lock()
        self._connect_serial()
        # periodic reconnect if needed
        self.timer_reconnect = self.create_timer(2.0, self._periodic_reconnect)

        # ---- I/O topics ----
        qos = rclpy.qos.QoSProfile(depth=10)
        self.sub_obs = self.create_subscription(Float64MultiArray, "~obs", self.cb_obs, qos)
        self.pub_actions = self.create_publisher(Float64MultiArray, "~actions", qos)

        self.get_logger().info(f"Ready. Policy={self.policy_kind}, act_dim={self.act_dim}, obs_dim_model={self.obs_dim_model}")

    # ---------------- Policy ----------------
    def _load_policy(self, path):
        if not path:
            self.get_logger().fatal("~policy_path is required")
            raise SystemExit(1)
        # try torchscript unless user forced checkpoint only
        tried_ts = False
        if self.force_ts or True:
            try:
                ts = torch.jit.load(path, map_location="cpu")
                ts.eval()
                self.policy = ts
                self.policy_kind = "torchscript"
                # Try to infer dims from submodules
                self.obs_dim_model, self.act_dim = self._infer_dims_from_script(ts)
                tried_ts = True
                self.get_logger().info("Loaded TorchScript policy.")
            except Exception as e:
                if self.force_ts:
                    raise
        if self.policy is None:
            # Fallback: reconstruct actor from checkpoint (RSL-RL)
            try:
                actor, obs_dim, act_dim = reconstruct_actor_from_rslrl_checkpoint(path)
                self.policy = actor
                self.policy_kind = "checkpoint_actor"
                self.obs_dim_model = obs_dim
                self.act_dim = act_dim
                self.get_logger().info(f"Reconstructed actor from checkpoint: obs_dim={obs_dim}, act_dim={act_dim}")
            except Exception as e:
                self.get_logger().fatal(f"Failed to load policy from '{path}': {e}")
                raise SystemExit(1)

        # Finalize obs_dim to use at runtime
        self.obs_dim = self.obs_dim_param or self.obs_dim_model or 55

    def _infer_dims_from_script(self, module):
        obs_dim = None
        act_dim = None
        try:
            actor = getattr(module, "actor", None)
            if isinstance(actor, torch.nn.Module):
                # Inspect linear layers
                first_linear, last_linear = None, None
                for m in actor.modules():
                    if isinstance(m, torch.nn.Linear):
                        if first_linear is None:
                            first_linear = m
                        last_linear = m
                obs_dim = getattr(first_linear, "in_features", None) if first_linear else None
                act_dim = getattr(last_linear, "out_features", None) if last_linear else None
        except Exception:
            pass
        return obs_dim, act_dim

    def _normalize_obs(self, x_np):
        if not self.normalize_obs:
            return x_np
        if len(self.obs_mean) != len(x_np) or len(self.obs_std) != len(x_np):
            self.get_logger().warn("normalize_obs is True but obs_mean/std lengths do not match; skipping normalization.")
            return x_np
        std = np.array(self.obs_std, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        mean = np.array(self.obs_mean, dtype=np.float32)
        return (x_np - mean) / std

    # ---------------- Serial ----------------
    def _connect_serial(self):
        port = self.port or find_arduino_port()
        if not port:
            self.get_logger().warn("No serial port specified/detected yet; will retry...")
            return
        try:
            self.ser = serial.Serial(port=port, baudrate=self.baudrate, timeout=0.05)
            time.sleep(0.1)
            self.get_logger().info(f"Connected serial {port} @ {self.baudrate}")
            # Initialize current positions on hardware
            for i in range(6):
                self._send_servo(i, self.grip_deg[i])
                time.sleep(0.01)
        except Exception as e:
            self.get_logger().warn(f"Serial open failed on {port}: {e}")
            self.ser = None

    def _periodic_reconnect(self):
        if self.ser is None or (not self.ser.is_open):
            self._connect_serial()

    def _send_servo(self, idx, deg):
        # Inversion and clamping already applied to self.grip_deg state; here we only send
        # the integer degrees.
        sid = [5,7,8,10,12,14][idx]  # default IDs; can be overridden if desired
        # Allow overriding IDs via ROS param (we didn't declare separately here to keep API tight).
        # If you need custom IDs, change the line above or mirror the earlier separate param.
        deg_i = int(round(clamp(deg, self.grip_limits[idx][0], self.grip_limits[idx][1])))
        cmd = f"S,{sid},{deg_i}\n"
        with self.ser_lock:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode("utf-8"))
                    if self.debug:
                        self.get_logger().info(f">> {cmd.strip()}")
            except Exception as e:
                self.get_logger().warn(f"Serial write failed: {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    # ---------------- Inference & Mapping ----------------
    def cb_obs(self, msg: Float64MultiArray):
        x = np.array(msg.data, dtype=np.float32)
        if x.size != self.obs_dim:
            self.get_logger().warn_throttle(2.0, f"obs len {x.size} != expected {self.obs_dim}")
            return
        x = self._normalize_obs(x)
        # Policy forward
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            out = self.policy(xt)
            if isinstance(out, (tuple, list)):
                out = out[0]
            a = out.cpu().numpy().reshape(-1)
        if self.apply_tanh:
            a = np.tanh(a)  # squash to [-1,1]

        # Publish full action vector for debugging
        act_msg = Float64MultiArray()
        act_msg.data = a.tolist()
        self.pub_actions.publish(act_msg)

        # Extract gripper actions, integrate to degrees
        for local_idx, act_idx in enumerate(self.grip_idxs):
            delta_deg = a[act_idx] * self.grip_scale_deg
            new_deg = self.grip_deg[local_idx] + delta_deg
            # apply inversion before sending: store non-inverted state and apply on send
            # Simpler: store already-inverted in state; to keep intuition, we keep non-inverted here
            # and apply inversion just-in-time:
            # but our _send_servo expects already clamped and final degrees; let's convert here.
            if self.invert_deg[local_idx]:
                send_deg = 180.0 - new_deg
            else:
                send_deg = new_deg
            # clamp to limits
            lo, hi = self.grip_limits[local_idx]
            send_deg = clamp(send_deg, lo, hi)
            # Update internal state as the inverse of what we sent (keep non-inverted for math)
            self.grip_deg[local_idx] = 180.0 - send_deg if self.invert_deg[local_idx] else send_deg
            # Send
            self._send_servo(local_idx, send_deg)

def main():
    rclpy.init()
    node = GripperInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser is not None:
            try:
                node.ser.close()
            except Exception:
                pass
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
