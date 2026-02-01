
import unittest
import sys
import os
import re

# Adjust path to allow imports
sys.path.append(os.getcwd())

try:
    from services.scan.python_engine.passive_analyzer import PassiveAnalyzer, PassiveFinding
except ImportError:
    # Fallback if run from a different directory structure
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
    from services.scan.python_engine.passive_analyzer import PassiveAnalyzer, PassiveFinding

class TestPassiveAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PassiveAnalyzer()
        self.url = "http://example.com"

    def test_stack_trace_detection(self):
        content = "Error in /var/www/html/index.php on line 123"
        response = {
            'status': 500,
            'content': {'text': content},
            'headers': []
        }
        findings = self.analyzer._check_error_disclosure(response, self.url)
        # Check if any finding has category 'Stack Trace Disclosure'
        found = False
        for f in findings:
            if f.category == 'Stack Trace Disclosure':
                found = True
                break
        self.assertTrue(found, f"Expected Stack Trace Disclosure, but got: {[f.category for f in findings]}")

    def test_db_error_detection(self):
        # Pattern: r'SQL syntax.*MySQL'
        content = "You have an error in your SQL syntax; check the manual that corresponds to your MySQL server version"
        response = {
            'status': 500,
            'content': {'text': content},
            'headers': []
        }
        findings = self.analyzer._check_error_disclosure(response, self.url)
        found = False
        for f in findings:
            if f.category == 'Database Error Disclosure':
                found = True
                break
        self.assertTrue(found, f"Expected Database Error Disclosure, but got: {[f.category for f in findings]}")

    def test_sensitive_data_in_url(self):
        # Pattern: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' (Email)
        # Wait, the test in previous attempt was for API Key, let's use Email as it is in sensitive_patterns
        url = "http://example.com?email=test@example.com"
        findings = self.analyzer._check_sensitive_data_in_url(url)
        self.assertTrue(any(f.category == 'Sensitive Data in URL' for f in findings))

    def test_sensitive_data_in_body(self):
        content = "Here is an email: test@example.com"
        findings = self.analyzer._check_sensitive_data_in_body(content, self.url)
        self.assertTrue(any(f.category == 'Sensitive Data in Body' for f in findings))

    def test_no_false_positives(self):
        content = "This is a normal response."
        response = {
            'status': 200,
            'content': {'text': content},
            'headers': []
        }
        findings = self.analyzer._check_error_disclosure(response, self.url)
        self.assertEqual(len(findings), 0)

if __name__ == '__main__':
    unittest.main()
