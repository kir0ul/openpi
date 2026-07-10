#!/usr/bin/env python3

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import droid_policy

model_name = "pi0_fast_droid"
# model_name = "pi05_droid"
config = _config.get_config(model_name)
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/" + model_name)

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# # Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": ...,
#     "observation/wrist_image_left": ...,
#     ...
#     "prompt": "pick up the fork"
# }
# action_chunk = policy.infer(example)["actions"]

# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
print("Actions:", result["actions"])
