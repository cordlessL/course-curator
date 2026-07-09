"""
Этап 5. Простая сводная аналитика по логу взаимодействий (logs/interactions.jsonl).

Без pandas - для 2-3 колонок статистики стандартной библиотеки достаточно,
тащить лишнюю зависимость ради этого не нужно.

Запуск:
    python analytics.py

Использование как модуля:
    from analytics import load_log, topic_counts, refusal_rate
"""

import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_PATH = BASE_DIR / "logs" / "interactions.jsonl"

# Куратору предписано (system_prompt.txt) при честном отказе использовать
# формулировку "не нашёл/нашла/нашли ответа" - это ориентир, а не гарантия:
# LLM перефразирует. На реальном прогоне (2026-07-10) поймали живой пример -
# вопрос про рекурсию честно отклонён ("не нашёл информации о рекурсии"), но
# исходный список маркеров ловил только "ответ", не "информаци-"/"данны-", и
# посчитал это не отказом. Список ниже - не полное решение (LLM может
# перефразировать и дальше), а расширение по факту увиденных и вероятных
# вариантов: "не нашёл/нашла/нашли" + "ответ" (как научили в промпте) или
# "информаци-"/"данны-" (синонимичные формулировки, которые модель уже
# использовала или может использовать вместо "ответа"). Основы без окончаний
# (например "информаци" вместо "информации"/"информацию"), чтобы `in` ловил
# падежные формы - тот же приём, что и с "ответ" ловит "ответа"/"ответ на".
REFUSAL_MARKERS = [
    "не нашёл ответ", "не нашла ответ", "не нашли ответ",
    "не нашёл информаци", "не нашла информаци", "не нашли информаци",
    "не нашёл данны", "не нашла данны", "не нашли данны",
]


def load_log() -> list[dict]:
    """Читает все записи из logs/interactions.jsonl. Пустой список, если лога ещё нет."""
    if not LOG_PATH.exists():
        return []
    entries = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def is_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def topic_counts(entries: list[dict]) -> Counter:
    """Топ тем по частоте вопросов. LMS-вопросы считаются под темой "LMS", а не по теме модуля."""
    return Counter(e["topic"] or "(тема не определена)" for e in entries)


def refusal_rate(entries: list[dict]) -> float:
    """Доля вопросов, на которые куратор честно ответил отказом.

    Считается по тексту ответа (is_refusal), а не по полю context_found -
    context_found показывает только факт непустого контекста, а не то,
    отказался ли куратор отвечать по существу (см. docs/stage5.md).
    """
    if not entries:
        return 0.0
    refusals = sum(1 for e in entries if is_refusal(e["answer"]))
    return refusals / len(entries)


def print_summary() -> None:
    entries = load_log()
    if not entries:
        print("Лог пуст (logs/interactions.jsonl не найден или не содержит записей).")
        return

    print(f"Всего вопросов в логе: {len(entries)}\n")

    print("Топ тем по частоте вопросов:")
    for topic, count in topic_counts(entries).most_common():
        print(f"  {count:>3}  {topic}")

    print(f"\nДоля честных отказов (по тексту ответа): {refusal_rate(entries):.0%}")

    kb_entries = [e for e in entries if e["question_type"] == "kb"]
    if kb_entries:
        kb_refusal = sum(1 for e in kb_entries if is_refusal(e["answer"])) / len(kb_entries)
        print(f"Доля честных отказов среди учебных (kb) вопросов: {kb_refusal:.0%} "
              f"(из {len(kb_entries)})")


if __name__ == "__main__":
    print_summary()
