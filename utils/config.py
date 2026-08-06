

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path):


    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError("Configuration file does not exist: {}".format(path))
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return simple_yaml_load(text)


def save_config(config, output_path):


    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        if yaml is not None:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        else:
            json.dump(config, file, ensure_ascii=False, indent=2)


def deep_update(base, updates):


    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def set_by_dotted_key(config, dotted_key, value):


    cursor = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[keys[-1]] = value


def apply_overrides(config, overrides):


    result = deepcopy(config)
    if not overrides:
        return result
    for item in overrides:
        if "=" not in item:
            raise ValueError("Invalid parameter format, should be key=value: {}".format(item))
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(result, key, parse_scalar(raw_value))
    return result


def resolve_paths(config, project_root):


    result = deepcopy(config)
    root = Path(project_root)
    for key, value in result.get("paths", {}).items():
        if isinstance(value, str) and value and not Path(value).is_absolute():
            result["paths"][key] = str(root / value)
    return result


def parse_scalar(value):


    if not isinstance(value, str):
        return value
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def simple_yaml_load(text):


    raw_lines = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        raw_lines.append((len(line) - len(line.lstrip(" ")), line.lstrip(" ")))
    value, index = _parse_yaml_block(raw_lines, 0, 0)
    if index != len(raw_lines):
        raise ValueError("simple_yaml_load function failed to fully parse the configuration file.")
    return value


def _parse_yaml_block(lines, index, indent):
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_dict(lines, index, indent)


def _parse_yaml_dict(lines, index, indent):
    result = {}
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError("YAML indentation error: {}".format(content))
        if ": " in content:
            key, value = content.split(": ", 1)
            result[key] = parse_scalar(value)
            index += 1
        elif content.endswith(":"):
            key = content[:-1]
            if index + 1 >= len(lines) or lines[index + 1][0] <= current_indent:
                result[key] = {}
                index += 1
            else:
                result[key], index = _parse_yaml_block(lines, index + 1, lines[index + 1][0])
        else:
            raise ValueError("Unable to parse the YAML line: {}".format(content))
    return result, index


def _parse_yaml_list(lines, index, indent):
    result = []
    while index < len(lines):
        current_indent, content = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ValueError("YAML list indentation error: {}".format(content))
        if not content.startswith("- "):
            break
        item_text = content[2:].strip()
        index += 1
        if not item_text:
            item, index = _parse_yaml_block(lines, index, lines[index][0])
            result.append(item)
        elif ": " in item_text:
            key, value = item_text.split(": ", 1)
            item = {key: parse_scalar(value)}
            if index < len(lines) and lines[index][0] > current_indent:
                child, index = _parse_yaml_dict(lines, index, lines[index][0])
                item.update(child)
            result.append(item)
        elif item_text.endswith(":"):
            key = item_text[:-1]
            child, index = _parse_yaml_block(lines, index, lines[index][0])
            result.append({key: child})
        else:
            result.append(parse_scalar(item_text))
    return result, index
