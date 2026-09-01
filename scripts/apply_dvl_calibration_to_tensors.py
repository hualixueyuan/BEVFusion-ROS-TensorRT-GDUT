#!/usr/bin/python3

"""Apply a direct_visual_lidar_calibration result to BEVFusion tensor files.

DVL stores ``T_lidar_camera`` as camera optical frame -> LiDAR frame:

    p_lidar = T_lidar_camera @ p_camera

BEVFusion calls the same transform ``camera2lidar``.  This script replaces one
camera slot and recomputes:

    lidar2image = camera_intrinsics @ inverse(camera2lidar)

Without ``--apply`` the script only prints the values it would write.
"""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


MAGIC = 0x33FF1101
DTYPE_TO_ID = {
    np.dtype("float32"): 3,
    np.dtype("float16"): 2,
    np.dtype("int32"): 1,
    np.dtype("int64"): 4,
    np.dtype("uint64"): 5,
    np.dtype("uint32"): 6,
    np.dtype("int8"): 7,
    np.dtype("uint8"): 8,
}
ID_TO_DTYPE = {value: key for key, value in DTYPE_TO_ID.items()}


def load_tensor(path):
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"tensor header is truncated: {path}")

    magic, ndim, dtype_id = np.frombuffer(raw[:12], dtype="<i4")
    if int(magic) != MAGIC:
        raise ValueError(f"invalid tensor magic in {path}: 0x{int(magic):08x}")
    if not 0 < int(ndim) <= 16:
        raise ValueError(f"invalid tensor dimension count in {path}: {ndim}")
    if int(dtype_id) not in ID_TO_DTYPE:
        raise ValueError(f"unsupported tensor dtype id in {path}: {dtype_id}")

    header_size = 12 + int(ndim) * 4
    shape = tuple(int(value) for value in np.frombuffer(raw[12:header_size], dtype="<i4"))
    dtype = ID_TO_DTYPE[int(dtype_id)]
    expected_size = header_size + math.prod(shape) * dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError(f"tensor size mismatch in {path}: got {len(raw)}, expected {expected_size}")

    return np.frombuffer(raw[header_size:], dtype=dtype).reshape(shape).copy()


def encode_tensor(array):
    array = np.ascontiguousarray(array)
    dtype = np.dtype(array.dtype)
    if dtype not in DTYPE_TO_ID:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    header = np.asarray([MAGIC, array.ndim, DTYPE_TO_ID[dtype]], dtype="<i4").tobytes()
    dimensions = np.asarray(array.shape, dtype="<i4").tobytes()
    return header + dimensions + array.tobytes(order="C")


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    old_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, old_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def quaternion_xyzw_to_rotation(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"expected quaternion [qx,qy,qz,qw], got shape {quaternion.shape}")
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("quaternion is zero or non-finite")
    x, y, z, w = quaternion / norm
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return quaternion / norm, rotation


def matrix_to_lists(matrix):
    return [[float(value) for value in row] for row in np.asarray(matrix)]


def transform_to_list(matrix):
    return matrix_to_lists(matrix)


def rotation_difference_degrees(first, second):
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def parse_args():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calib_json", type=Path, help="DVL calib.json containing results.T_lidar_camera")
    parser.add_argument("--config-dir", type=Path, default=repository / "configs/gdut")
    parser.add_argument("--slot", type=int, default=1)
    parser.add_argument("--camera-name", default="FRONT_RIGHT")
    parser.add_argument("--result-key", default="T_lidar_camera")
    parser.add_argument("--apply", action="store_true", help="atomically replace tensors and metadata")
    return parser.parse_args()


def main():
    args = parse_args()
    calib_path = args.calib_json.resolve()
    config_dir = args.config_dir.resolve()
    if not calib_path.is_file():
        raise FileNotFoundError(calib_path)

    paths = {
        name: config_dir / name
        for name in ("camera2lidar.tensor", "camera_intrinsics.tensor", "lidar2image.tensor", "img_aug_matrix.tensor")
    }
    yaml_path = config_dir / "current_calibration.yaml"
    info_path = config_dir / "calibration_info.json"
    for path in [*paths.values(), yaml_path, info_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    calib = json.loads(calib_path.read_text())
    try:
        values = np.asarray(calib["results"][args.result_key], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"missing results.{args.result_key} in {calib_path}") from exc
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"results.{args.result_key} must contain seven finite numbers")

    translation = values[:3]
    quaternion, rotation = quaternion_xyzw_to_rotation(values[3:])
    camera2lidar_new = np.eye(4, dtype=np.float64)
    camera2lidar_new[:3, :3] = rotation
    camera2lidar_new[:3, 3] = translation

    camera2lidar = load_tensor(paths["camera2lidar.tensor"])
    intrinsics = load_tensor(paths["camera_intrinsics.tensor"])
    lidar2image = load_tensor(paths["lidar2image.tensor"])
    img_aug = load_tensor(paths["img_aug_matrix.tensor"])
    tensors = {
        "camera2lidar.tensor": camera2lidar,
        "camera_intrinsics.tensor": intrinsics,
        "lidar2image.tensor": lidar2image,
        "img_aug_matrix.tensor": img_aug,
    }
    for name, tensor in tensors.items():
        if tensor.shape != (1, 6, 4, 4) or tensor.dtype != np.float32:
            raise ValueError(f"{name} must be float32 [1,6,4,4], got {tensor.dtype} {tensor.shape}")
    if not 0 <= args.slot < camera2lidar.shape[1]:
        raise ValueError(f"slot {args.slot} is outside [0, {camera2lidar.shape[1] - 1}]")

    current_yaml = yaml.safe_load(yaml_path.read_text())
    camera_records = [camera for camera in current_yaml["cameras"] if int(camera["slot"]) == args.slot]
    if len(camera_records) != 1 or camera_records[0]["name"] != args.camera_name:
        raise ValueError(f"slot {args.slot} is not uniquely mapped to {args.camera_name} in {yaml_path}")
    camera_record = camera_records[0]

    old_camera2lidar = camera2lidar[0, args.slot].astype(np.float64)
    old_other_slots = np.delete(camera2lidar.copy(), args.slot, axis=1)
    old_lidar2image_other_slots = np.delete(lidar2image.copy(), args.slot, axis=1)

    old_quaternion = np.asarray(camera_record["camera2lidar_quaternion_xyzw"], dtype=np.float64)
    if np.dot(quaternion, old_quaternion) < 0.0:
        quaternion = -quaternion

    camera2lidar[0, args.slot] = camera2lidar_new.astype(np.float32)
    # Compute from the exact float32 matrix that BEVFusion will load.
    lidar2camera_new = np.linalg.inv(camera2lidar[0, args.slot].astype(np.float64))
    lidar2image_new = intrinsics[0, args.slot].astype(np.float64) @ lidar2camera_new
    lidar2image[0, args.slot] = lidar2image_new.astype(np.float32)

    if not np.array_equal(old_other_slots, np.delete(camera2lidar, args.slot, axis=1)):
        raise AssertionError("a non-target camera2lidar slot changed")
    if not np.array_equal(old_lidar2image_other_slots, np.delete(lidar2image, args.slot, axis=1)):
        raise AssertionError("a non-target lidar2image slot changed")

    formula_errors = []
    for slot in range(camera2lidar.shape[1]):
        expected = intrinsics[0, slot].astype(np.float64) @ np.linalg.inv(camera2lidar[0, slot].astype(np.float64))
        formula_errors.append(float(np.max(np.abs(expected - lidar2image[0, slot].astype(np.float64)))))
    max_formula_error = max(formula_errors)
    if max_formula_error > 1.0e-3:
        raise AssertionError(f"lidar2image formula error is too large: {max_formula_error}")

    rotation_orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    rotation_determinant = float(np.linalg.det(rotation))
    if rotation_orthogonality_error > 1.0e-9 or abs(rotation_determinant - 1.0) > 1.0e-9:
        raise AssertionError("computed quaternion rotation is invalid")

    translation_change = float(np.linalg.norm(camera2lidar_new[:3, 3] - old_camera2lidar[:3, 3]))
    rotation_change = rotation_difference_degrees(old_camera2lidar[:3, :3], rotation)
    optical_axis = rotation[:, 2]
    optical_axis_elevation = float(
        np.degrees(np.arctan2(optical_axis[2], np.hypot(optical_axis[0], optical_axis[1])))
    )

    tensor_bytes = {name: encode_tensor(tensor) for name, tensor in tensors.items()}
    hashes = {name: sha256_bytes(content) for name, content in tensor_bytes.items()}

    camera_record["camera2lidar_translation_xyz_m"] = [float(value) for value in translation]
    camera_record["camera2lidar_quaternion_xyzw"] = [float(value) for value in quaternion]
    camera_record["camera2lidar_4x4"] = transform_to_list(camera2lidar[0, args.slot])
    camera_record["lidar2camera_4x4"] = transform_to_list(lidar2camera_new.astype(np.float32))
    camera_record["lidar2image_4x4"] = transform_to_list(lidar2image[0, args.slot])
    current_yaml["source_tensors_sha256"] = hashes.copy()
    current_yaml["front_right_recalibration"] = {
        "source": "direct_visual_lidar_calibration",
        "result_file": str(calib_path),
        "result_key": args.result_key,
        "initial_was_camera2lidar": True,
        "applied_to_tensor_slot": args.slot,
        "translation_change_from_previous_norm_m": translation_change,
        "rotation_change_from_previous_deg": rotation_change,
        "optical_axis_elevation_deg": optical_axis_elevation,
    }

    info = json.loads(info_path.read_text())
    info["provenance_note"] = (
        "The base tensors were imported from the mounted source. FRONT_RIGHT camera2lidar and "
        "lidar2image were replaced by scripts/apply_dvl_calibration_to_tensors.py using the local "
        "direct_visual_lidar_calibration result; the other five tensor slots are unchanged."
    )
    info["front_right_recalibration"] = {
        "source": str(calib_path),
        "result_key": args.result_key,
        "direction": "p_lidar = camera2lidar * p_camera_optical",
        "tensor_slot": args.slot,
        "translation_xyz_m": [float(value) for value in translation],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "translation_change_from_previous_norm_m": translation_change,
        "rotation_change_from_previous_deg": rotation_change,
        "optical_axis_elevation_deg": optical_axis_elevation,
    }
    info.setdefault("validation", {})["shape"] = [1, 6, 4, 4]
    info["validation"]["dtype"] = "float32"
    info["validation"]["lidar2image_formula"] = "camera_intrinsics @ inverse(camera2lidar)"
    info["validation"]["lidar2image_error_per_slot"] = formula_errors
    info["validation"]["lidar2image_max_abs_error"] = max_formula_error
    info["validation"]["rotation_orthogonality_max_abs_error"] = rotation_orthogonality_error
    info["validation"]["rotation_determinant"] = rotation_determinant
    info["validation"]["warning"] = (
        "Runtime images are not undistorted by the current ROS1 wrapper; DVL used the calibrated "
        "plumb_bob distortion coefficients during optimization."
    )
    info["sha256"] = hashes.copy()

    report = {
        "mode": "apply" if args.apply else "preview",
        "generated_at": datetime.now().astimezone().isoformat(),
        "calibration_result": str(calib_path),
        "result_key": args.result_key,
        "camera_name": args.camera_name,
        "tensor_slot": args.slot,
        "transform_direction": "camera_optical_to_lidar",
        "old_camera2lidar_4x4": transform_to_list(old_camera2lidar),
        "new_camera2lidar_4x4": transform_to_list(camera2lidar[0, args.slot]),
        "new_lidar2camera_4x4": transform_to_list(lidar2camera_new.astype(np.float32)),
        "new_lidar2image_4x4": transform_to_list(lidar2image[0, args.slot]),
        "translation_xyz_m": [float(value) for value in translation],
        "quaternion_xyzw": [float(value) for value in quaternion],
        "translation_change_from_previous_norm_m": translation_change,
        "rotation_change_from_previous_deg": rotation_change,
        "optical_axis_elevation_deg": optical_axis_elevation,
        "lidar2image_error_per_slot": formula_errors,
        "lidar2image_max_abs_error": max_formula_error,
        "rotation_orthogonality_max_abs_error": rotation_orthogonality_error,
        "rotation_determinant": rotation_determinant,
        "sha256": hashes,
    }

    print(json.dumps(report, indent=2))
    if not args.apply:
        print("Preview only: pass --apply to replace tensors and metadata.")
        return

    yaml_bytes = yaml.safe_dump(current_yaml, sort_keys=False, allow_unicode=True, width=120).encode()
    info_bytes = (json.dumps(info, indent=2) + "\n").encode()
    report_path = config_dir / "last_tensor_update.json"
    report_bytes = (json.dumps(report, indent=2) + "\n").encode()

    for name in ("camera2lidar.tensor", "lidar2image.tensor"):
        atomic_write(paths[name], tensor_bytes[name])
    atomic_write(yaml_path, yaml_bytes)
    atomic_write(info_path, info_bytes)
    atomic_write(report_path, report_bytes)

    reloaded_camera2lidar = load_tensor(paths["camera2lidar.tensor"])
    reloaded_lidar2image = load_tensor(paths["lidar2image.tensor"])
    if not np.array_equal(reloaded_camera2lidar, camera2lidar):
        raise AssertionError("camera2lidar read-back verification failed")
    if not np.array_equal(reloaded_lidar2image, lidar2image):
        raise AssertionError("lidar2image read-back verification failed")
    for name in ("camera2lidar.tensor", "lidar2image.tensor"):
        actual_hash = hashlib.sha256(paths[name].read_bytes()).hexdigest()
        if actual_hash != hashes[name]:
            raise AssertionError(f"SHA256 read-back verification failed for {name}")
    print(f"Applied and verified slot {args.slot} ({args.camera_name}); report: {report_path}")


if __name__ == "__main__":
    main()
