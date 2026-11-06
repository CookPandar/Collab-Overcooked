import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "claude_history", Path(__file__).resolve().parents[1] / "scripts/claude_history.py")
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)


class HistoryMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.home = self.root / "claude"
        self.project = self.root / "old-project"
        self.project.mkdir()
        self.key = history.project_key(self.project)
        directory = self.home / "projects" / self.key
        directory.mkdir(parents=True)
        self.row = {"cwd": str(self.project), "message": {"content": str(self.project)}}
        (directory / "session.jsonl").write_text(json.dumps(self.row) + "\n")
        (directory / "memory").mkdir()
        (directory / "memory/MEMORY.md").write_text("project memory")
        (self.home / ".credentials.json").write_text("do not export")
        (self.home / "history.jsonl").write_text(
            json.dumps({"project": str(self.project), "sessionId": "session"}) + "\n" +
            json.dumps({"project": "/unrelated", "sessionId": "other"}) + "\n")
        self.output = self.root / "export"

    def test_export_is_project_scoped(self):
        manifest = history.export_history(self.home, [self.project], self.output)
        self.assertEqual(manifest["sessions"], ["session"])
        self.assertFalse((self.output / ".credentials.json").exists())
        self.assertEqual(len(history.read_jsonl(self.output / "history.jsonl")), 1)
        self.assertTrue((self.output / "projects" / self.key / "memory/MEMORY.md").exists())

    def test_restore_remaps_metadata_not_dialogue_and_is_idempotent(self):
        history.export_history(self.home, [self.project], self.output)
        target = self.root / "new-project"
        target.mkdir()
        home = self.root / "new-claude"
        history.restore_history(home, self.output, target)
        history.restore_history(home, self.output, target)
        row = history.read_jsonl(home / "projects" / history.project_key(target) / "session.jsonl")[0]
        self.assertEqual(row["cwd"], str(target))
        self.assertEqual(row["message"]["content"], str(self.project))
        self.assertEqual(len(history.read_jsonl(home / "history.jsonl")), 1)

    def test_restore_refuses_conflicts(self):
        history.export_history(self.home, [self.project], self.output)
        target = self.root / "new-project"
        target.mkdir()
        directory = self.root / "new-claude/projects" / history.project_key(target)
        directory.mkdir(parents=True)
        transcript = directory / "session.jsonl"
        transcript.write_text('{"existing": true}\n')
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            history.restore_history(self.root / "new-claude", self.output, target)
        self.assertEqual(transcript.read_text(), '{"existing": true}\n')

    def test_missing_project_does_not_create_export(self):
        with self.assertRaises(ValueError):
            history.export_history(self.home, [self.root / "absent"], self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
