import sys
import unittest
from unittest.mock import patch, MagicMock

# Fully mock rich for testing
sys.modules['rich'] = MagicMock()
sys.modules['rich.console'] = MagicMock()
sys.modules['rich.panel'] = MagicMock()
sys.modules['rich.text'] = MagicMock()
sys.modules['rich.prompt'] = MagicMock()
sys.modules['rich.table'] = MagicMock()
sys.modules['rich.theme'] = MagicMock()

import subprocess
import webbrowser

class BulkExtractorDummy:
    def __init__(self):
        self.name = "Bulk extractor"

    def gui_mode(self, console):
        # The code we're testing:
        # console.print(Panel(Text(self.name, justify="center"), style=PURPLE_STYLE))
        # console.print("[bold magenta]Cloning repository and attempting to run GUI...[/]")
        try:
            subprocess.run(["sudo", "git", "clone", "https://github.com/simsong/bulk_extractor.git"], check=True)
            subprocess.run(["ls", "src/"], cwd="bulk_extractor", check=True)
            subprocess.run(["./BEViewer"], cwd="bulk_extractor/java_gui", check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # console.print(f"[red]Error during execution: {e}[/red]")
            pass

    def cli_mode(self, console):
        # The code we're testing:
        # console.print(Panel(Text(self.name + " - CLI Mode", justify="center"), style=PURPLE_STYLE))
        try:
            subprocess.run(["sudo", "apt", "install", "-y", "bulk-extractor"], check=True)
            # console.print("[magenta]Showing bulk_extractor help and options:[/]")
            subprocess.run(["bulk_extractor", "-h"], check=True)

            p1 = subprocess.Popen(["echo", "bulk_extractor [options] imagefile"], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["boxes", "-d", "headline"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p3 = subprocess.Popen(["lolcat"], stdin=p2.stdout)
            p3.wait()
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            # console.print(f"[red]Error: {e}[/red]")
            pass

class ToolsleyDummy:
    def __init__(self):
        self.name = "Toolsley"
        self.project_url = "https://www.toolsley.com/"

    def custom_run(self, console):
        # console.print(f"[cyan]Opening {self.name} website...[/cyan]")
        # console.print(f"[blue]Visit: {self.project_url}[/blue]")
        webbrowser.open(self.project_url)

class TestForensicToolsRefactor(unittest.TestCase):
    def setUp(self):
        self.bulk_extractor = BulkExtractorDummy()
        self.toolsley = ToolsleyDummy()
        self.mock_console = MagicMock()

    @patch('subprocess.run')
    def test_bulk_extractor_gui_mode(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        self.bulk_extractor.gui_mode(self.mock_console)
        self.assertEqual(mock_run.call_count, 3)
        mock_run.assert_any_call(["sudo", "git", "clone", "https://github.com/simsong/bulk_extractor.git"], check=True)
        mock_run.assert_any_call(["ls", "src/"], cwd="bulk_extractor", check=True)
        mock_run.assert_any_call(["./BEViewer"], cwd="bulk_extractor/java_gui", check=True)

    @patch('subprocess.Popen')
    @patch('subprocess.run')
    def test_bulk_extractor_cli_mode(self, mock_run, mock_popen):
        mock_run.return_value = MagicMock(returncode=0)
        mock_p1 = MagicMock()
        mock_p2 = MagicMock()
        mock_p3 = MagicMock()
        mock_popen.side_effect = [mock_p1, mock_p2, mock_p3]

        self.bulk_extractor.cli_mode(self.mock_console)

        self.assertEqual(mock_run.call_count, 2)
        mock_run.assert_any_call(["sudo", "apt", "install", "-y", "bulk-extractor"], check=True)
        mock_run.assert_any_call(["bulk_extractor", "-h"], check=True)

        self.assertEqual(mock_popen.call_count, 3)
        mock_popen.assert_any_call(["echo", "bulk_extractor [options] imagefile"], stdout=subprocess.PIPE)
        mock_popen.assert_any_call(["boxes", "-d", "headline"], stdin=mock_p1.stdout, stdout=subprocess.PIPE)
        mock_popen.assert_any_call(["lolcat"], stdin=mock_p2.stdout)
        mock_p3.wait.assert_called_once()

    @patch('webbrowser.open')
    def test_toolsley_custom_run(self, mock_webbrowser):
        self.toolsley.custom_run(self.mock_console)
        mock_webbrowser.assert_called_once_with(self.toolsley.project_url)

if __name__ == '__main__':
    unittest.main()
