# Training data: sources, terms and how to obtain them

[日本語](data.ja.md)

The weights in this project were trained on three kinds of data. Each has its own terms. This page sums
them up (checked against the original texts on 2026-10-19) and explains how to obtain the data **through
the official routes**. It is a reading of the license texts, not legal advice.

| Data | Used for | Terms (summary) | Obtained from |
| --- | --- | --- | --- |
| **AMASS** (via HumanML3D) | 10,064 of the 10,427 training clips (KIT, CMU, BMLmovi, Eyes_Japan, HDM05, …) | Non-commercial scientific research, education or art, licensed **personally** to the user. Training is prohibited only "for commercial use of any kind". The data, modified or not, may not be made available to third parties. Cite the AMASS paper | [amass.is.tue.mpg.de](https://amass.is.tue.mpg.de) (registration and acceptance) |
| **HumanAct12** (via HumanML3D) | 363 clips | Distributed openly by its authors ([action-to-motion](https://github.com/EricGuo5513/action-to-motion), MIT repository) | Included in the HumanML3D repository (`pose_data/`) |
| **HumanML3D** text annotations and scripts | Captions, clip list, the conversion scripts | MIT repository. Its authors do not distribute the motion data itself, because AMASS's policy forbids it | [EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D) |
| **SMPL** body model | Fitting the torso of the video-derived clips (ROMP) | Non-commercial; derivative works allowed; the model itself may not be redistributed; training for commercial use prohibited. Cite the SMPL paper | [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) (registration and acceptance) |
| Author's own videos | 13 real clips in `data/video_reach/` | Recorded by the author, who is the only person in them. Only skeletons and object states are stored | — |

**What this means for you**

- Training and sharing models **for non-commercial research** is consistent with these terms. Training for a
  commercial product is not.
- **Never redistribute AMASS-derived motion data**, including HumanML3D's `new_joints`, and never download
  it from third-party mirrors. Every user obtains it from AMASS under their own license.
- Nothing in this repository or the model repository contains AMASS motions or the SMPL model. `vla/standing.json`
  and `vla/full_split.json` hold clip IDs, frame counts and HumanML3D captions (the MIT-licensed text annotations), no motion; `data/stats/statistics.json` holds aggregate statistics.
- For commercial use, contact the licensors (AMASS/SMPL: ps-license@tue.mpg.de; NVIDIA for GR00T), or
  better, train on data you own. Contributed CBD clips are the way there ([CONTRIBUTING.md](../CONTRIBUTING.md)).

## Building HumanML3D through the official route

You need this only to rebuild the HumanML3D training render (`b3_dataset.tar`) yourself.

1. **Register and accept the terms** at [AMASS](https://amass.is.tue.mpg.de) and download the **SMPL+H**
   body data for the subsets listed in the [HumanML3D README](https://github.com/EricGuo5513/HumanML3D).
2. **Download the body models** HumanML3D needs: Extended SMPL+H ([MANO site](https://mano.is.tue.mpg.de/download.php))
   and DMPL ([SMPL site](https://smpl.is.tue.mpg.de/download.php)).
3. **Run HumanML3D's scripts** in its repository: `raw_pose_processing.ipynb`, then `motion_representation.ipynb`.
   The result you need is `HumanML3D/new_joints/` (3D joint positions, 22 SMPL joints, 20 fps).
4. **Zip that folder and keep it in your own storage.** For the Colab scripts, put it in your Google Drive
   as `MyDrive/VLA/VLA_1_0/data/humanml3d_new_joints.zip` (or pass `JOINTS=<path on the VM>`).
5. Build the training render:

   ```bash
   bash colab/start.sh data mydata
   bash colab/go.sh data mydata      # HumanML3D → CBD → 117-d LeRobot → MuJoCo render → Drive data/b3_dataset.tar
   ```

   Locally, `python3 vla/fetch_and_curate.py --joints <new_joints dir or zip> --all` does step 1 of that pipeline.
   It fetches the captions and lists from the HumanML3D repository and copies your `new_joints` into place.

## Citations

If you use these models or this data pipeline in a publication, cite the data and models it is built on:

```bibtex
@inproceedings{AMASS:2019,
  title     = {AMASS: Archive of Motion Capture as Surface Shapes},
  author    = {Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019},
  month     = {Oct},
  url       = {https://amass.is.tue.mpg.de}
}

@article{SMPL:2015,
  author    = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
  title     = {{SMPL}: A Skinned Multi-Person Linear Model},
  journal   = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
  month     = oct,
  number    = {6},
  pages     = {248:1--248:16},
  publisher = {ACM},
  volume    = {34},
  year      = {2015}
}

@InProceedings{Guo_2022_CVPR,
  author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
  title     = {Generating Diverse and Natural 3D Human Motions From Text},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2022},
  pages     = {5152-5161}
}

@inproceedings{gr00tn1_2025,
  archivePrefix = {arxiv},
  eprint        = {2503.14734},
  title         = {{GR00T} {N1}: An Open Foundation Model for Generalist Humanoid Robots},
  author        = {NVIDIA and others},
  month         = {March},
  year          = {2025},
  booktitle     = {ArXiv Preprint}
}
```

HumanAct12: C. Guo et al., *Action2Motion: Conditioned Generation of 3D Human Motions*, ACM Multimedia 2020.
The full GR00T N1 author list is in the [Isaac-GR00T README](https://github.com/NVIDIA/Isaac-GR00T).
