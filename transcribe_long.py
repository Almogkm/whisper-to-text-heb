from __future__ import annotations

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(PROJECT_SRC))

from whisper_md_transcriber.longform import main


raise SystemExit(main())
