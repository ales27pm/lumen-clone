#!/usr/bin/env python3
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tripleboot_aio.sh"


class TripleBootAIOTests(unittest.TestCase):
    def run_script(self, *args, check=True):
        env = os.environ.copy()
        result = subprocess.run(
            [str(SCRIPT), *args],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            self.fail(f"command failed: {result.args}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        return result

    def test_workspace_manifest_and_readme_are_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "tripleboot"
            self.run_script("init-workspace", "--workdir", str(workdir))

            manifest = json.loads((workdir / "tripleboot-manifest.json").read_text())
            self.assertEqual(manifest["schema"], "lumen.tripleboot.manifest.v1")
            self.assertEqual(manifest["payload_roots"]["ubuntu"], "isos/ubuntu")
            self.assertTrue((workdir / "staging" / "TripleBoot" / "README.txt").exists())

    def test_register_iso_copies_payload_and_writes_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "ubuntu.iso"
            source.write_bytes(b"fake-iso-content")
            workdir = root / "work"

            self.run_script(
                "register-iso",
                "--workdir",
                str(workdir),
                "--kind",
                "ubuntu",
                "--source",
                str(source),
            )

            copied = workdir / "isos" / "ubuntu" / "ubuntu.iso"
            self.assertEqual(copied.read_bytes(), b"fake-iso-content")
            self.assertRegex((copied.with_suffix(".iso.sha256")).read_text(), r"^[0-9a-f]{64}$")

    def test_stage_payloads_builds_ventoy_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            usb = root / "usb"
            usb.mkdir()
            (workdir / "isos" / "ubuntu").mkdir(parents=True)
            (workdir / "isos" / "windows").mkdir(parents=True)
            (workdir / "staging" / "TripleBoot").mkdir(parents=True)
            (workdir / "isos" / "ubuntu" / "ubuntu.iso").write_text("ubuntu")
            (workdir / "isos" / "windows" / "windows.iso").write_text("windows")
            (workdir / "staging" / "TripleBoot" / "README.txt").write_text("readme")

            self.run_script("stage-payloads", "--workdir", str(workdir), "--usb-mount", str(usb))

            self.assertEqual((usb / "ISO" / "Ubuntu" / "ubuntu.iso").read_text(), "ubuntu")
            self.assertEqual((usb / "ISO" / "Windows" / "windows.iso").read_text(), "windows")
            self.assertEqual((usb / "TripleBoot" / "README.txt").read_text(), "readme")

    def test_download_ventoy_requires_explicit_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script("download-ventoy", "--workdir", tmp, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires --version or --url", result.stderr)


if __name__ == "__main__":
    unittest.main()
