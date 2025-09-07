# run_one_step_from_rslrl_ckpt.py
import torch, numpy as np
from collections import OrderedDict

'''
Observation vector
tensor([[ 0.0812,  0.6941,  0.1901,  0.0989,  0.0073, -0.0078, -0.0130, -0.1295,
         -0.4832, -0.2623, -1.0949,  1.4090, -0.7991, -0.5438, -0.3789, -0.0409,
         -0.7785, -1.1049,  0.8326,  0.2193,  0.5026, -1.2046,  1.9413, -2.8417,
         -0.9336, -0.0590, -5.3172,  3.4349,  3.3108, -5.2411, -3.4316,  3.3113,
          0.4388,  0.1062,  0.0279,  0.4443,  0.1182,  0.2838,  1.0000,  0.0000,
          0.0000,  0.0000,  0.3112, -1.6367,  0.6610,  0.7602,  0.0577, -0.0822,
          0.0890, -0.1480,  0.4089, -1.3171, -1.6444,  1.9522, -0.6715]],
       device='cuda:0')



Action Space
--- Action breakdown (policy group) ---
arm_action                     [0:7] dim=7
gripper_action                 [7:13] dim=6
TOTAL action dims = 13

[arm_action] raw: tensor([0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
[arm_action] processed: tensor([0., 0., 0., 0., 0., 0., 0.], device='cuda:0')
[gripper_action] raw: tensor([0., 0., 0., 0., 0., 0.], device='cuda:0')
[gripper_action] processed: tensor([0., 0., 0., 0., 0., 0.], device='cuda:0')
observation vector:  tensor([[-7.4954e-03,  1.6959e-02, -1.4156e-02,  5.7764e-03, -1.6915e-02,
         -2.5207e-02,  1.4417e-03,  5.9106e-03, -2.4225e-02, -4.6768e-02,
         -1.0973e-01,  5.5266e-02, -1.6324e-01,  3.2195e-01,  9.6257e-01,
          5.0286e-01, -5.7230e-01,  1.5263e+00, -1.2686e+00,  6.0775e-01,
         -4.3161e-01, -1.6757e+00, -6.5505e-01,  5.1974e-01, -1.7199e+00,
         -1.8744e+00, -1.0051e+00,  1.6468e+00,  2.7535e-02, -1.1612e+00,
         -1.7199e+00,  2.7533e-02,  5.2259e-01, -2.4497e-01,  5.2057e-02,
          4.6528e-01, -9.2576e-02,  3.0637e-01,  1.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00, -2.0669e+00,  2.7255e+00, -2.0903e+00,
         -2.9587e-01,  1.5107e-01, -4.5418e-01, -7.3689e-03,  3.2617e-01,
          5.8190e-01, -5.5371e-01, -5.3974e-01,  9.5231e-01, -1.5252e+00]],
       device='cuda:0')

JointPositionActionCfg --> Absolut Joint Positions

    ".*Core_Bottom_Box_Right",                  :Right
    ".*Core_Bottom_Umdrehung_104",              :Left
    ".*Core_compact_Box_Thumb",                 :Thumb
    ".*Motorbox_D5021_right_Connector_Right",   :Phi Right
    ".*Motorbox_D5021_left_Connector_Left",     :Phi Left
    ".*Motorbox_D5021_thumb_Connector_Thumb", : Phi Thumb

'''

def rad2angle(rad):
    return rad / np.pi * 180

def get_model():
    ckpt = torch.load("model_1450.pt", map_location="cpu")
    #print("ckpt", ckpt)
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
    return actor, sizes


model, sizes = get_model()
obs_dim = sizes[0][0]
obs = torch.zeros(1, obs_dim)   # dummy obs
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
       device='cpu')
with torch.no_grad():
    act = model(obs).squeeze(0).cpu().numpy()
print("obs_dim:", obs_dim, " action_dim:", sizes[-1][1])
print("action:", np.round(act, 4))
#print("gripper action ", np.round(rad2angle(act[7:]), 4))
print("gripper action ", np.round(rad2angle(act[7:]), 4))


test_actino = np.array([-0.0717, -0.3693,  0.5426,  0.8619, -0.1434,  0.1532, -0.2940,  0.4324,
          0.1436, -2.1475, -1.2667,  2.4290, -0.9997])

print("test action ", np.rad2deg(test_actino[7:]))
