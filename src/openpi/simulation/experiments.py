import re
import sys
import tomllib

from main import run_scheduler_comparison


def _preprocess_toml(text: str) -> str:
    """Evaluate Python list expressions in TOML values before parsing.

    Allows expressions like:
        d_infer = [0] + [3]
        execution_horizon = [50] * 10
        d_infer = [0] + [2 + i for i in range(1, 11)]
    """
    lines = []
    for line in text.splitlines():
        m = re.match(r"^(\s*\w+\s*=\s*)(.+)$", line)
        if m and any(op in m.group(2) for op in [" + ", " * ", " for "]):
            try:
                result = eval(m.group(2))
                if isinstance(result, list):
                    line = m.group(1) + str(result)
            except Exception:
                pass
        lines.append(line)
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run experiments.py <config.toml>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        raw = f.read()
    config = tomllib.loads(_preprocess_toml(raw))

    defaults = config.get("defaults", {})
    experiments = config.get("experiments", [])

    if not experiments:
        # Single experiment at top level
        params = {k: v for k, v in config.items() if k != "defaults"}
        run_scheduler_comparison(**params)
        return

    for exp in experiments:
        params = {**defaults, **exp}
        name = params.pop("name", None)
        if name:
            base = params.get("output_dir", "results")
            params["output_dir"] = f"{base}/{name}"
            print(f"\n{'=' * 60}")
            print(f"Experiment: {name}")
        run_scheduler_comparison(**params)


if __name__ == "__main__":
    main()
