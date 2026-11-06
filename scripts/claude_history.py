"""Export project-scoped Claude history or restore an extracted export.

Only project transcripts/memory, matching file snapshots/tasks, and filtered
history entries are copied. Global settings, credentials and environment state
are deliberately excluded. Treat the output as private nevertheless.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shutil


def project_key(path):
    return re.sub(r"[^a-zA-Z0-9]", "-", str(path))


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_history(home, projects, output):
    projects = [str(Path(p).expanduser().resolve()) for p in projects]
    sources = [(p, home / "projects" / project_key(p)) for p in projects]
    missing = [str(src) for _, src in sources if not src.is_dir()]
    if missing:
        raise ValueError("Missing Claude project directories: " + ", ".join(missing))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    sessions = set()
    for _, src in sources:
        # Subagent directories and memory live alongside the main transcripts.
        shutil.copytree(src, output / "projects" / src.name)
        sessions.update(p.stem for p in src.glob("*.jsonl"))
    for category in ("file-history", "tasks"):
        for session in sorted(sessions):
            src = home / category / session
            if src.is_dir():
                shutil.copytree(src, output / category / session)
    rows = [row for row in read_jsonl(home / "history.jsonl")
            if row.get("project") in projects or row.get("sessionId") in sessions]
    (output / "history.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "format": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "projects": [{"path": p, "key": src.name} for p, src in sources],
        "sessions": sorted(sessions),
        "excluded": ["credentials", "global settings", "shell snapshots", "session environment"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, output / "claude_history.py")
    return manifest


def remap_metadata(value, mappings):
    """Rebase path metadata, never rewrite historical prompts or tool output."""
    if isinstance(value, list):
        return [remap_metadata(item, mappings) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in ("cwd", "project", "projectPath", "rootPath", "worktreePath") and isinstance(item, str):
            for old, new in mappings:
                if item == old or item.startswith(old + "/"):
                    item = new + item[len(old):]
                    break
            result[key] = item
        else:
            result[key] = remap_metadata(item, mappings)
    return result


def restore_history(home, source, project):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["format"] != 1:
        raise ValueError("Unsupported export format")
    project = str(project.expanduser().resolve())
    if not Path(project).is_dir():
        raise ValueError("Clone/create the destination project first: " + project)
    mappings = [(entry["path"], project) for entry in manifest["projects"]]
    destination = home / "projects" / project_key(project)
    pending = {}
    for category in ("projects", "file-history", "tasks"):
        root = source / category
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("Unexpected symlink in export: " + str(path))
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            target = (destination.joinpath(*relative.parts[1:]) if category == "projects"
                      else home / category / relative)
            data = path.read_bytes()
            if path.suffix == ".jsonl":
                data = "".join(json.dumps(remap_metadata(row, mappings), ensure_ascii=False) + "\n"
                               for row in read_jsonl(path)).encode()
            elif path.suffix == ".json":
                data = (json.dumps(remap_metadata(json.loads(data), mappings),
                                   ensure_ascii=False, indent=2) + "\n").encode()
            if target in pending and pending[target] != data:
                raise ValueError("Conflicting exported files: " + str(target))
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Refusing to overwrite existing history: " + str(target))
            if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
                raise ValueError("Refusing symlink destination: " + str(target))
            pending[target] = data
    history = home / "history.jsonl"
    existing = read_jsonl(history)
    encoded = {json.dumps(row, sort_keys=True) for row in existing}
    additions = []
    for row in read_jsonl(source / "history.jsonl"):
        row = remap_metadata(row, mappings)
        key = json.dumps(row, sort_keys=True)
        if key not in encoded:
            additions.append(row)
            encoded.add(key)
    # Validate everything before writing; stop Claude before restoring.
    for target, data in pending.items():
        if not target.exists():
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            target.chmod(0o600)
    if additions:
        history.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if history.is_symlink():
            raise ValueError("Refusing symlink history file")
        with history.open("a") as stream:
            # Existing histories normally end in a newline; tolerate those that do not.
            if history.stat().st_size and not history.read_bytes().endswith(b"\n"):
                stream.write("\n")
            for row in additions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        history.chmod(0o600)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-home", type=Path, default=Path.home() / ".claude")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--project", action="append", required=True)
    export.add_argument("--output", type=Path, required=True)
    restore = commands.add_parser("restore")
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "export":
        manifest = export_history(args.claude_home, args.project, args.output)
    else:
        manifest = restore_history(args.claude_home, args.source, args.project)
    print("{}: {} project(s), {} session(s)".format(
        args.command, len(manifest["projects"]), len(manifest["sessions"])))


if __name__ == "__main__":
    main()
