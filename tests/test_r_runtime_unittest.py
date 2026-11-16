import subprocess
import unittest
from unittest.mock import patch

from extensions.r_runtime import run_rscript


class RRuntimeTests(unittest.TestCase):
    def test_clean_r_session_is_used(self):
        completed = subprocess.CompletedProcess(["Rscript"], 0, "ok", "")
        with patch("extensions.r_runtime.subprocess.run", return_value=completed) as invoke:
            self.assertIs(completed, run_rscript(["Rscript", "script.R"], purpose="test"))
        self.assertEqual(["Rscript", "--vanilla", "script.R"], invoke.call_args.args[0])
        self.assertIs(subprocess.DEVNULL, invoke.call_args.kwargs["stdin"])
        self.assertTrue(invoke.call_args.kwargs["start_new_session"])

    def test_timeout_restarts_once(self):
        completed = subprocess.CompletedProcess(["Rscript"], 0, "ok", "")
        timeout = subprocess.TimeoutExpired(["Rscript"], 1)
        with patch("extensions.r_runtime.subprocess.run", side_effect=[timeout, completed]) as invoke:
            result = run_rscript(["Rscript", "script.R"], purpose="test")
        self.assertIs(completed, result)
        self.assertEqual(2, invoke.call_count)

    def test_repeated_timeout_has_a_clear_failure(self):
        timeout = subprocess.TimeoutExpired(["Rscript"], 1)
        with patch("extensions.r_runtime.subprocess.run", side_effect=[timeout, timeout]):
            with self.assertRaisesRegex(RuntimeError, "timed out 2 times"):
                run_rscript(["Rscript", "script.R"], purpose="test")


if __name__ == "__main__":
    unittest.main()
