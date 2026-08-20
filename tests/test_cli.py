import subprocess
import sys


def test_cli_module_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_negotiation.cli",
            "--product",
            "Test Widget",
            "--seller_initial",
            "100",
            "--seller_min",
            "80",
            "--buyer_initial",
            "100",
            "--buyer_max",
            "120",
            "--rounds",
            "5",
            "--buyer_strategy",
            "cooperative",
            "--seller_strategy",
            "cooperative",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    assert "--- Negotiation History ---" in output
    assert "Deal reached: True" in output
    assert "rounds_used: 1" in output
