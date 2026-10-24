#!/usr/bin/env python3
"""Convert CBD bone_rotations to MuJoCo qpos, and optionally render an mp4.

The coordinate transform was measured from the CBD repository's reference sample (which has both
frames.jsonl and motion.npz for the same clip). From CBD canonical (Y-up, facing +Z) to MuJoCo (Z-up,
facing -Y) is **+90 degrees about the X axis**: local rotations are conjugated (w,x,y,z) -> (w,x,-z,y)
and the root position maps (x,y,z) -> (x,-z,y).

Playback is kinematic, not physics (qpos is assigned every frame, then mj_forward).
"""
import argparse
import json
import pathlib
import shutil

import numpy as np

from rotation import qmul

# The qpos layout follows the body order in humanoid.xml (measured with this script and fixed)
SLOT = {"spine": 7, "chest": 11, "neck": 15, "head": 19,
        "left_upper_arm": 23, "left_lower_arm": 27, "left_hand": 31,
        "right_upper_arm": 40, "right_lower_arm": 44, "right_hand": 48,
        "left_upper_leg": 57, "left_lower_leg": 61, "left_foot": 65,
        "right_upper_leg": 69, "right_lower_leg": 73, "right_foot": 77}
NQ = 81


def to_mj(q):
    """Rotations from CBD coordinates to MuJoCo coordinates (conjugation by +90 deg about X). (...,4) wxyz."""
    q = np.asarray(q)
    return np.stack([q[..., 0], q[..., 1], -q[..., 3], q[..., 2]], -1)


def qpos_from_rotations(B, hips, fold_clavicle=True):
    """B: bone name -> (T,4) wxyz local rotations, hips: (T,3) root position (CBD coordinates)."""
    hips = np.asarray(hips, np.float64)
    q = np.zeros((len(hips), NQ), np.float64)
    q[:, 0:3] = np.stack([hips[:, 0], -hips[:, 2], hips[:, 1]], -1)
    q[:, 3:7] = to_mj(B["hips"])
    for name, s in SLOT.items():
        loc = B[name]
        if fold_clavicle and name in ("left_upper_arm", "right_upper_arm"):
            # humanoid.xml has no clavicle joint; the left_shoulder body sits directly
            # under chest. CBD goes neck (= SMPL spine3) -> clavicle -> upper arm, so
            # those two levels must be folded into the upper-arm joint or the arm points the wrong way.
            side = name.split("_")[0]
            loc = qmul(qmul(B["neck"], B[f"{side}_shoulder"]), loc)
        q[:, s:s + 4] = to_mj(loc)
    return q


def build_qpos(frames, fold_clavicle=True):
    rot = [f["human"]["bone_rotations_xyzw"] for f in frames]
    B = {k: np.array([[r[k][3], r[k][0], r[k][1], r[k][2]] for r in rot])  # xyzw -> wxyz
         for k in rot[0]}
    hips = [f["human"]["hips_position"] for f in frames]
    return qpos_from_rotations(B, hips, fold_clavicle)


def ground(model, data, q):
    """Shift the whole clip up or down by a constant so the lowest point of the feet sits on the floor (z=0).

    The HumanML3D skeleton and the humanoid.xml skeleton have different proportions, so as is the feet
    sink in or float. Playback is kinematic with no physics, so a visual correction is enough.
    The shift is constant so vertical motion such as jumps is preserved.
    """
    import mujoco
    # Skip geoms attached directly to the world, so the floor plane is not counted.
    # We want the bottom of each geom, not its center, so evaluate it lowered by its size.
    g = [i for i in range(model.ngeom) if model.geom_bodyid[i] != 0]
    rad = np.array([model.geom_size[i].max() for i in g])
    lo = np.inf
    for t in range(len(q)):
        data.qpos[:] = q[t]
        mujoco.mj_forward(model, data)
        lo = min(lo, float((data.geom_xpos[g, 2] - rad).min()))
    q[:, 2] -= lo
    return lo


def frames(model, data, q, r):
    """Render frame by frame with a third-person tracking camera and return the frames.

    r is a reused Renderer. Creating one per clip exhausts OSMesa's GL contexts, and from the
    second clip on it keeps returning the same image (the video becomes a still)."""
    import mujoco
    for t in range(len(q)):
        data.qpos[:] = q[t]
        mujoco.mj_forward(model, data)
        # Create the camera anew every frame. update_scene rewrites the MjvCamera, so
        # reusing it mixes in shots looking up from under the floor.
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.trackbodyid = -1
        cam.fixedcamid = -1
        cam.distance, cam.elevation, cam.azimuth = 3.2, -12, 135
        cam.lookat[:] = [q[t, 0], q[t, 1], 0.9]
        r.update_scene(data, cam)
        yield r.render()


def render(model, data, q, fps, path, r):
    import imageio
    imageio.mimwrite(path, [f.copy() for f in frames(model, data, q, r)],
                     fps=fps, quality=8, macro_block_size=1)


# humanoid.xml ships with the CBD repository's sample. In case it is placed on its own,
# also look next to this file.
XML_CANDIDATES = [
    "humanoid.xml",
    "examples/human-capture/sample_output/02_mujoco/humanoid.xml",
    "examples/language-to-motion/sample_output/"
    "01_the_person_picks_up_the_cup_drinks_and_p/humanoid.xml",
]


def find_xml():
    here = pathlib.Path(__file__).parent
    for c in XML_CANDIDATES:
        for base in (pathlib.Path.cwd(), here, here.parent):
            p = base / c
            if p.exists():
                return p
    raise SystemExit("humanoid.xml が見つからない。--xml で指定すること。\n"
                     "  探した場所: " + ", ".join(XML_CANDIDATES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", help="CBD クリップのディレクトリ")
    ap.add_argument("--xml", default=None,
                    help="humanoid.xml。省略時は既知の場所を順に探す")
    ap.add_argument("--out", default="work/mujoco_preview")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--no-fold-clavicle", action="store_true")
    a = ap.parse_args()

    import mujoco
    xml = pathlib.Path(a.xml) if a.xml else find_xml()
    model = mujoco.MjModel.from_xml_string(xml.read_text(encoding="utf-8"))
    assert model.nq == NQ, f"qpos 次元が {model.nq}（想定 {NQ}）"
    data = mujoco.MjData(model)
    w, h = (int(v) for v in a.size.split("x"))
    renderer = mujoco.Renderer(model, h, w) if a.render else None

    for c in a.clips:
        c = pathlib.Path(c)
        frames = [json.loads(l) for l in
                  open(c / "timeline" / "frames.jsonl", encoding="utf-8")]
        man = json.load(open(c / "manifest.json", encoding="utf-8"))
        fps = float(man["fps"])
        q = build_qpos(frames, not a.no_fold_clavicle)
        off = ground(model, data, q)

        out = pathlib.Path(a.out) / f"{c.parent.name}_{c.name}"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "motion.npz", qpos=q.astype(np.float32),
                 fps=np.array([fps], np.float32),
                 timestamp_sec=np.arange(len(q), dtype=np.float32) / fps)
        shutil.copy(xml, out / "humanoid.xml")
        ref = xml.parent / "replay_mujoco.py"
        if ref.exists():
            shutil.copy(ref, out / "replay_mujoco.py")
        cap = frames[0]["caption"]["en"]
        (out / "README.md").write_text(
            f"# {c.parent.name} / {c.name}\n\n{cap}\n\n"
            f"frames={len(q)} fps={fps} 接地補正={-off:+.3f} m\n"
            f"`python replay_mujoco.py`（要ディスプレイ）\n", encoding="utf-8")

        note = ""
        if a.render:
            render(model, data, q, fps, out / "preview.mp4", renderer)
            note = f"  -> {out/'preview.mp4'}"
        print(f"{c.parent.name:9s} {c.name}  {len(q):4d}f  接地補正 {-off:+.3f}m"
              f"  | {cap[:52]}{note}")


if __name__ == "__main__":
    main()
