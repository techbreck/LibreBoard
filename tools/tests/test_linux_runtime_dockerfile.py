import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "runtime" / "onnxruntime" / "Dockerfile.linux"


class LinuxRuntimeDockerfileTest(unittest.TestCase):
    def test_path_host_tools_are_debian_cmake_and_ninja(self):
        text = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("bookworm-backports", text)
        self.assertIn("ninja-build", text)
        self.assertIn("cmake=3.31.6-2~bpo12+1", text)
        self.assertIn("cmake-data=3.31.6-2~bpo12+1", text)
        self.assertIn("'cmake;3.31.6'", text)
        self.assertIn('test "$(command -v cmake)" = /usr/bin/cmake', text)
        self.assertIn('test "$(command -v ninja)" = /usr/bin/ninja', text)
        self.assertNotIn(
            'ENV PATH="/opt/android-sdk/cmake/3.31.6/bin:${PATH}"',
            text,
        )
