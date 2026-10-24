"""Shared rotation utilities. Quaternions are in wxyz order.

Rotation matrix -> quaternion does not use the naive `w = sqrt(1+trace)/2`. Near 180-degree rotations
`1+trace` drops to 0 and the division loses precision. In the real data, `turn` clips reached |w|=0.0001,
and 1.75% of all frames were in the danger zone. Use Shepperd's method, which picks the largest
of the diagonal terms.
"""
import numpy as np


def normalize(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], -1)


def qinv(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    t = 2.0 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def qbetween(u, v):
    """Minimal rotation taking u onto v (swing only, zero twist)."""
    u, v = normalize(u), normalize(v)
    q = np.concatenate([1.0 + (u * v).sum(-1, keepdims=True), np.cross(u, v)], -1)
    deg = np.linalg.norm(q, axis=-1) < 1e-6          # opposite (180 degrees)
    if deg.any():
        alt = np.tile(np.array([0.0, 1.0, 0.0]), u.shape[:-1] + (1,))
        alt[np.abs(u[..., 1]) > 0.9] = np.array([1.0, 0.0, 0.0])
        ax = normalize(np.cross(u, alt))
        q = np.where(deg[..., None],
                     np.concatenate([np.zeros_like(ax[..., :1]), ax], -1), q)
    return normalize(q)


def qaxis(axis, angle):
    h = angle * 0.5
    return np.concatenate([np.cos(h)[..., None],
                           normalize(axis) * np.sin(h)[..., None]], -1)


def mat_to_quat(m):
    """(...,3,3) -> (...,4) wxyz. Columns are the basis vectors. Shepperd's method."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    tr = m00 + m11 + m22
    # of the 4 candidates, pick the one with the largest value under the square root
    cand = np.stack([1.0 + tr, 1.0 + m00 - m11 - m22,
                     1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], -1)
    k = np.argmax(cand, -1)
    s = 2.0 * np.sqrt(np.maximum(np.take_along_axis(cand, k[..., None], -1)[..., 0],
                                 1e-20))
    a, b, c = m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], \
        m[..., 1, 0] - m[..., 0, 1]
    d, e, f = m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0], \
        m[..., 1, 2] + m[..., 2, 1]
    out = np.stack([
        np.stack([0.25 * s, a / s, b / s, c / s], -1),
        np.stack([a / s, 0.25 * s, d / s, e / s], -1),
        np.stack([b / s, d / s, 0.25 * s, f / s], -1),
        np.stack([c / s, e / s, f / s, 0.25 * s], -1),
    ], -2)
    q = np.take_along_axis(out, k[..., None, None], -2)[..., 0, :]
    return normalize(q) * np.sign(q[..., :1] + 1e-30)     # make w >= 0


def quat_to_mat(q):
    """(...,4) wxyz -> (...,3,3). Columns are the basis vectors."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_to_6d(q):
    """(...,4) -> (...,6). The first 2 columns of the rotation matrix side by side (Zhou et al. 2019).

    Regressing quaternions directly makes the loss discontinuous because of the double cover. In the real data the sign
    flipped between adjacent frames in 0.38% of cases.
    """
    m = quat_to_mat(q)
    return np.concatenate([m[..., :, 0], m[..., :, 1]], -1)


def sixd_to_quat(a):
    """(...,6) -> (...,4). Orthonormalize the 2 columns with Gram-Schmidt and recover the rotation."""
    b0 = normalize(a[..., :3])
    b1 = normalize(a[..., 3:] - (a[..., 3:] * b0).sum(-1, keepdims=True) * b0)
    return mat_to_quat(np.stack([b0, b1, np.cross(b0, b1)], -1))
