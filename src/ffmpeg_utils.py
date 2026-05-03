import glob
import os
import shutil


def _normalize_executable_name(command_name):
    if os.name == "nt" and not command_name.lower().endswith(".exe"):
        return f"{command_name}.exe"
    return command_name


def _iter_env_override_candidates(command_name):
    command_key = command_name.upper().replace(".EXE", "")
    env_var_names = [
        f"{command_key}_PATH",
        f"{command_key}_BINARY",
    ]
    if command_key == "FFMPEG":
        env_var_names.append("IMAGEIO_FFMPEG_EXE")

    for env_var_name in env_var_names:
        candidate = os.environ.get(env_var_name)
        if candidate:
            yield os.path.abspath(candidate)


def _iter_windows_install_candidates(executable_name):
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    user_profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    chocolatey_root = os.environ.get("ChocolateyInstall") or r"C:\ProgramData\chocolatey"

    static_candidates = [
        os.path.join(local_app_data, "Microsoft", "WinGet", "Links", executable_name),
        os.path.join(local_app_data, "Programs", "ffmpeg", "bin", executable_name),
        os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin", executable_name),
        os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "ffmpeg", "bin", executable_name),
        os.path.join(chocolatey_root, "bin", executable_name),
        os.path.join(user_profile, "scoop", "shims", executable_name),
    ]

    for candidate in static_candidates:
        yield candidate

    winget_patterns = [
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "ffmpeg-*", "bin", executable_name),
        os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "FFmpeg*", "**", "bin", executable_name),
    ]
    for pattern in winget_patterns:
        for candidate in sorted(glob.glob(pattern, recursive=True), reverse=True):
            yield candidate


def find_ffmpeg_binary(command_name):
    executable_name = _normalize_executable_name(command_name)

    for candidate in _iter_env_override_candidates(executable_name):
        if os.path.isfile(candidate):
            return candidate

    path_hit = shutil.which(command_name) or shutil.which(executable_name)
    if path_hit:
        return os.path.abspath(path_hit)

    if os.name == "nt":
        for candidate in _iter_windows_install_candidates(executable_name):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    return None


def ensure_ffmpeg_tools_available(command_names=("ffmpeg", "ffprobe")):
    resolved = {}
    path_parts = os.environ.get("PATH", "").split(os.pathsep) if os.environ.get("PATH") else []

    for command_name in command_names:
        binary_path = find_ffmpeg_binary(command_name)
        if not binary_path:
            continue

        resolved[command_name] = binary_path
        binary_dir = os.path.dirname(binary_path)
        if binary_dir and binary_dir not in path_parts:
            path_parts.insert(0, binary_dir)

        command_key = command_name.upper()
        os.environ[f"{command_key}_PATH"] = binary_path
        os.environ[f"{command_key}_BINARY"] = binary_path
        if command_name == "ffmpeg":
            os.environ["IMAGEIO_FFMPEG_EXE"] = binary_path

    if path_parts:
        os.environ["PATH"] = os.pathsep.join(path_parts)

    return resolved
