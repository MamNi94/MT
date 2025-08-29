# run_one_step_from_rslrl_ckpt.py
import torch, numpy as np
from collections import OrderedDict

def rad2angle(rad):
    return rad / np.pi * 180


ckpt = torch.load("Gripper Control/model_1450.pt", map_location="cpu")
sd = ckpt["model_state_dict"]

# discover actor linear layers and sizes
layer_ids = sorted({int(k.split('.')[1]) for k in sd if k.startswith("actor.") and k.endswith(".weight")})
sizes = [ (sd[f"actor.{i}.weight"].shape[1], sd[f"actor.{i}.weight"].shape[0]) for i in layer_ids ]

layers = []
for li, (inp, out) in enumerate(sizes):
    layers += [(f"lin{li}", torch.nn.Linear(inp, out))]
    if li < len(sizes)-1:
        layers += [(f"tanh{li}", torch.nn.Tanh())]
actor = torch.nn.Sequential(OrderedDict(layers))
# load actor weights
mapping = {f"lin{li}.weight": sd[f"actor.{i}.weight"] for li, i in enumerate(layer_ids)}
mapping.update({f"lin{li}.bias":   sd[f"actor.{i}.bias"]   for li, i in enumerate(layer_ids)})
actor.load_state_dict(mapping, strict=True)
actor.eval()

obs_dim = sizes[0][0]
obs = torch.zeros(1, obs_dim)   # dummy obs
with torch.no_grad():
    act = actor(obs).squeeze(0).cpu().numpy()
print("obs_dim:", obs_dim, " action_dim:", sizes[-1][1])
print("action:", np.round(act, 4))
print("gripper action ", np.round(rad2angle(act[6:]), 4))
