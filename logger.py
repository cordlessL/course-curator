"""
Этап 5. Логирование вызовов ask_curator() в JSONL.

Формат - JSON Lines (по одному JSON-объекту на строку): дозапись новой
строки в конец файла не требует перечитывать и переписывать файл целиком
(в отличие от одного большого JSON-массива), а список источников (переменной
длины - 0, 1 или несколько чанков) ложится в JSON нативно, без обходных
путей, характерных для CSV с ячейками переменной структуры.

Использование:
    from logger import log_interaction
    log_interaction(question=..., student_level=..., question_type=...,
                    topic=..., sources=..., context_found=..., answer=...)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_PATH = BASE_DIR / "logs" / "interactions.jsonl"


def log_interaction(*,
                    question: str,
                    student_level: str,
                    question_type: str,
                    topic: str | None,
                    sources: list[str],
                    context_found: bool,
                    answer: str) -> None:
    """Дописывает одну строку-запись о вызове ask_curator() в лог."""
    LOG_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "student_level": student_level,
        "question_type": question_type,
        "topic": topic,
        "sources": sources,
        "context_found": context_found,
        "answer": answer,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
