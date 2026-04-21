import sys
import unittest
from unittest.mock import patch

# Standard test skipping if dependencies fail in restricted env
try:
    from services.features.function_forensic.manager import ForensicManager
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

class TestForensicManager(unittest.TestCase):
    @unittest.skipUnless(HAS_DEPS, "Dependencies not met")
    def setUp(self):
        self.manager = ForensicManager()

    @unittest.skipUnless(HAS_DEPS, "Dependencies not met")
    @patch("services.features.function_forensic.manager.datetime")
    def test_create_case_error_path(self, mock_datetime):
        """Test the error path of ForensicManager.create_case"""
        # Mock datetime.now() to raise an exception
        mock_datetime.now.side_effect = Exception("Simulated error in datetime.now()")

        # Verify that the exception is logged and re-raised correctly
        with self.assertRaises(Exception) as context:
            self.manager.create_case(
                case_name="Test Case",
                investigator="Test Investigator"
            )

        self.assertEqual(str(context.exception), "Simulated error in datetime.now()")

if __name__ == '__main__':
    unittest.main()
