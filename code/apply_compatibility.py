"""Apply the small compatibility fixes required by the pinned Qlib environment."""
from pathlib import Path

import qlib


def main():
    package = Path(qlib.__file__).resolve().parent
    replacements = [
        ("utils/index_data.py", "pd.Series(self.data, index=self.index)",
         "pd.Series(self.data, index=self.index.tolist())"),
        ("rl/contrib/backtest.py", "pd.concat(record_list, 0)",
         "pd.concat(record_list, axis=0)"),
        ("rl/trainer/trainer.py", 'torch.load(ckpt_path, map_location="cpu")',
         'torch.load(ckpt_path, map_location="cpu", weights_only=False)'),
    ]
    for relative, old, new in replacements:
        path = package / relative
        source = path.read_text(encoding="utf-8")
        if old in source:
            path.write_text(source.replace(old, new), encoding="utf-8")
            print(f"Patched {path}")
        elif new in source:
            print(f"Already compatible: {path}")
        else:
            raise RuntimeError(f"Unexpected Qlib source in {path}")


if __name__ == "__main__":
    main()
