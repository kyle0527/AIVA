"""
Cryptographic Security Command Handler

This module provides the command handler for the `function_crypto` module,
enabling the AI Commander to execute cryptographic security checks.
It strictly interfaces with the high-performance Rust core (`crypto_analyzer`).
"""

import time
import subprocess
import json
import os
from typing import Optional, Dict, Any
from datetime import datetime

from services.aiva_common.command_center import CommandHandler
from services.aiva_common.schemas.commands import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandType,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)

class CryptoCommandHandler(CommandHandler):
    """
    Handles cryptographic security check commands.

    This handler requires the Rust-based `crypto_analyzer` binary to be present.
    No fallback or degradation to Python-based heuristics is permitted.
    """

    def __init__(self):
        self.logger = logger
        self.logger.info("✅ CryptoCommandHandler initialized")
        # Path to the compiled Rust binary
        self.rust_binary_path = os.path.join(
            os.path.dirname(__file__), "rust_core", "target", "release", "crypto_analyzer"
        )
        # Strict validation on initialization
        if not os.path.exists(self.rust_binary_path):
             self.logger.warning(
                 f"⚠️ Critical: Rust binary not found at {self.rust_binary_path}. "
                 "Execution will fail if called."
             )

    async def handle_command(
        self,
        command: AICommand,
        context: Optional[CommandContext] = None
    ) -> AICommandResult:
        """
        Executes a crypto security check command.

        Payload expected:
        {
            "target": "string",
            "check_type": "string"
        }
        """
        start_time = time.time()

        try:
            # 1. Validate Binary Existence (Strict)
            if not os.path.exists(self.rust_binary_path):
                raise FileNotFoundError(
                    f"Rust binary 'crypto_analyzer' not found at {self.rust_binary_path}. "
                    "Please compile the Rust core to use this feature."
                )

            # 2. Validate Command Inputs
            payload = command.payload or {}
            target = payload.get("target")
            check_type = payload.get("check_type", "identify")

            if not target:
                raise ValueError("Missing 'target' in payload")

            self.logger.info(f"🔒 Starting Crypto Check: {check_type} on target")

            # 3. Execute Logic via Rust Binary
            result_data = self._run_rust_binary(target, check_type)

            # 4. Build Result
            execution_time = time.time() - start_time

            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.COMPLETED,
                success=True,
                result=result_data,
                execution_time=execution_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                metrics={
                    "check_type": check_type,
                    "target_length": len(str(target)),
                    "engine": "rust_crypto_analyzer"
                }
            )

        except Exception as e:
            self.logger.error(f"❌ Crypto check failed: {e}", exc_info=True)
            return AICommandResult(
                command_id=command.command_id,
                status=CommandStatus.FAILED,
                success=False,
                result={},
                execution_time=time.time() - start_time,
                started_at=datetime.fromtimestamp(start_time),
                completed_at=datetime.now(),
                error=str(e),
                error_code="CRYPTO_EXECUTION_ERROR"
            )

    def _run_rust_binary(self, target: str, check_type: str) -> Dict[str, Any]:
        """Runs the compiled Rust binary."""
        try:
            cmd = [self.rust_binary_path, "--target", target, "--check", check_type, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = f"Rust binary execution failed: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON output from Rust binary: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
