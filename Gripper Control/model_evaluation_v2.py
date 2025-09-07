import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Tuple, Any
import numpy as np

# ---------- your observation ----------
obs = torch.tensor([[-2.6234e-01,  7.2478e-01,  1.2624e-01,  1.6485e-01, -1.7691e-01,
         -1.3001e-03, -1.2309e-01,  2.0589e-01, -3.4135e-01, -2.6182e-01,
         -8.5111e-01,  1.1012e+00, -8.5472e-01, -3.4503e-01, -7.1183e-02,
         -1.7307e-01,  2.7317e-01,  1.8754e+00,  3.0184e-02,  4.6729e-01,
          9.0789e-01,  1.7958e-01, -5.5122e-01,  5.3272e-01, -1.5985e+00,
         -6.5901e-04, -4.2095e+00,  4.8099e+00, -2.2401e+00, -4.2063e+00,
         -4.8099e+00, -2.2401e+00,  5.0491e-01, -7.4236e-02,  2.1000e-02,
          4.5180e-01,  1.2139e-01,  4.8012e-01,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00, -3.0629e-02, -4.8451e-01,  6.5701e-01,
          8.8100e-01, -2.9641e-01,  1.7707e-01, -3.0602e-01,  6.3581e-01,
          1.4042e-01, -2.1712e+00, -1.1143e+00,  2.3105e+00, -1.0717e+00]],
       dtype=torch.float32)
OBS_DIM = obs.shape[-1]  # = 55

CKPT = "model_1450.pt"

# ---------- tiny MLP policy ----------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, hidden: List[int], act_dim: int, tanh_out: bool = True):
        super().__init__()
        layers = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(d, act_dim)
        self.tanh_out = tanh_out

    def forward(self, x):
        x = self.backbone(x)
        a = self.head(x)
        return torch.tanh(a) if self.tanh_out else a

def is_tensor_dict(d: Dict[str, Any]) -> bool:
    # “state_dict-like” if it has at least a couple tensor entries
    t = 0
    for v in d.values():
        if isinstance(v, torch.Tensor):
            t += 1
            if t >= 2:
                return True
    return False

def all_nested_state_dicts(obj: Any) -> List[Dict[str, torch.Tensor]]:
    """Breadth-first: return all nested dicts that look like torch state_dicts."""
    out = []
    q = deque([("", obj)])
    while q:
        path, cur = q.popleft()
        if isinstance(cur, dict):
            if is_tensor_dict(cur):
                out.append(cur)
            for k, v in cur.items():
                q.append((f"{path}.{k}" if path else k, v))
    return out

def try_build_chain(sd: Dict[str, torch.Tensor], obs_dim: int) -> Tuple[List[Tuple[str,str,int,int]], Dict[str, torch.Tensor]]:
    """Find a chain of linear layers starting at obs_dim by following weight shapes [out,in]."""
    # filter actor-ish keys (avoid critic/value entries)
    actor_items = {k: v for k, v in sd.items()
                   if isinstance(v, torch.Tensor)
                   and ("critic" not in k and "value" not in k and "vf" not in k)}
    W_keys = [k for k, v in actor_items.items() if v.ndim == 2]  # [out, in]
    if not W_keys:
        return [], actor_items
    # stable sort, but we’ll also pick by shape continuity
    W_keys.sort()

    chain = []
    used = set()
    cur_in = obs_dim
    # Try to find the first layer whose in_features == obs_dim
    for k in W_keys:
        W = actor_items[k]
        out_f, in_f = W.shape
        if in_f == cur_in:
            b_key = k.replace("weight", "bias")
            chain.append((k, b_key if b_key in actor_items else "", in_f, out_f))
            used.add(k)
            cur_in = out_f
            break

    # If we didn’t find a starting layer, give up on this sd
    if not chain:
        return [], actor_items

    # Follow-on layers
    for _ in range(32):
        found = False
        for k in W_keys:
            if k in used:
                continue
            W = actor_items[k]
            out_f, in_f = W.shape
            if in_f == cur_in:
                b_key = k.replace("weight", "bias")
                chain.append((k, b_key if b_key in actor_items else "", in_f, out_f))
                used.add(k)
                cur_in = out_f
                found = True
                break
        if not found:
            break
    return chain, actor_items

# 1) load
ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
if not isinstance(ckpt, dict):
    raise TypeError("Unexpected checkpoint type; need a (nested) dict of tensors (state_dict).")

# 2) search for candidate state_dicts
candidates = [ckpt] + all_nested_state_dicts(ckpt)

best = None
best_chain = []
best_actor_items = None

for sd in candidates:
    chain, actor_items = try_build_chain(sd, OBS_DIM)
    if len(chain) > len(best_chain):
        best_chain = chain
        best_actor_items = actor_items
        best = sd

if not best_chain:
    # Print diagnostics to help map it
    print("\n[DIAG] Could not infer a chain. Showing candidate tensor keys (name -> shape):")
    shown = 0
    for sd in candidates:
        ks = [(k, tuple(v.shape)) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim in (1,2)]
        if ks:
            print("\n-- Candidate dict --")
            for k, shp in ks[:40]:
                print(f"{k:60s} {shp}")
            shown += 1
            if shown >= 3:
                break
    raise RuntimeError("Could not infer a linear-layer chain from any nested state_dict.")

# 3) rebuild policy with inferred sizes
hidden = [o for (_, _, i, o) in best_chain[:-1]]
act_dim = best_chain[-1][3]
policy = MLPPolicy(OBS_DIM, hidden, act_dim, tanh_out=True)

# 4) transplant weights (by order)
lin_layers = [m for m in policy.modules() if isinstance(m, nn.Linear)]
with torch.no_grad():
    for (layer, (w_key, b_key, _in_f, _out_f)) in zip(lin_layers, best_chain):
        layer.weight.copy_(best_actor_items[w_key])
        if b_key and b_key in best_actor_items:
            layer.bias.copy_(best_actor_items[b_key])
        else:
            nn.init.zeros_(layer.bias)

policy.eval()
with torch.no_grad():
    action = policy(obs)

print("\n[OK] Inferred hidden sizes:", hidden, "| act_dim:", act_dim)

action_numpy = action.numpy()
print("action numpy:", np.rad2deg(action_numpy))
print("Gripper Action: ", np.rad2deg(action_numpy[0][7:]))


'''
JointPositionActionCfg --> Absolut Joint Positions

    ".*Core_Bottom_Box_Right",                  :Phi Right
    ".*Core_Bottom_Umdrehung_104",              :Phi Left
    ".*Core_compact_Box_Thumb",                 :Phi Thumb
    ".*Motorbox_D5021_right_Connector_Right",   :Right
    ".*Motorbox_D5021_left_Connector_Left",     :Left
    ".*Motorbox_D5021_thumb_Connector_Thumb",   :Thumb

'''