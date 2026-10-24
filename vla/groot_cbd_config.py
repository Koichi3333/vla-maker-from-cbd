"""Modality config that registers CBD's 117 dims as GR00T N1.7's NEW_EMBODIMENT.

Pass it to --modality-config-path of launch_finetune.py / gr00t/data/stats.py.
Importing it runs register_modality_config() (GR00T's convention).
Key names match the range names in meta/modality.json written by cbd_to_lerobot.py.

Action representation (provisional as of B-0, to be revisited in B-2):
- root_translation is RELATIVE (displacement from the current position). Absolute position depends on the
  starting point and does not help generalization.
- Bone rotations (6D) are ABSOLUTE. Element-wise differences of 6D are not rotation differences,
  so RELATIVE would mean regressing a meaningless quantity.
"""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

GROUPS = ["root_translation", "hips", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
ACTION_HORIZON = 16  # 0.8 s at 20 fps. N1.7's maximum is 40. If you change it, rerun stats.py


def _action(rep):
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


cbd_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=GROUPS),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=GROUPS,
        action_configs=[_action(ActionRepresentation.RELATIVE)]
        + [_action(ActionRepresentation.ABSOLUTE)] * (len(GROUPS) - 1),
    ),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.task_description"]
    ),
}

register_modality_config(cbd_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
