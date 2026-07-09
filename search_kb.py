"""
Этап 2. Поиск по базе знаний: семантика + фильтры по метаданным.

Основная функция - search_kb(). Её же будет импортировать ask_curator()
на этапе 4, поэтому она оформлена как переиспользуемый модуль.

Пример из командной строки:
    python search_kb.py "как обработать деление на ноль"
    python search_kb.py "что такое срез" --module 4
    python search_kb.py "правила пересдачи" --type методичка
    python search_kb.py "методы списков" --topic "Списки и словари, индексация, срезы, методы"
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # подхватывает OPENAI_API_KEY из .env в корне проекта

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "course_kb"
# Модель настраивается через EMBEDDING_MODEL в .env; значение по умолчанию
# ("text-embedding-3-small") зашито здесь как fallback на случай, если переменная
# не задана - должно совпадать с тем, что использовалось при индексации в
# index_kb.py, иначе расстояния между чанками теряют смысл (эмбеддинги из
# разных моделей несравнимы).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Порог релевантности: у Chroma score - это расстояние (меньше = ближе).
# Откалибровано на тестовом прогоне (run_tests.py): 1.32 - минимум, при
# котором чанк про срезы списков (lec_m4_collections, 1.308) ещё проходит.
# Полностью отсечь ложные срабатывания вроде вопроса про Django одним
# порогом не получается - его дистанция (1.270) ниже дистанции валидного
# "среза", то есть эти два случая пересекаются в пространстве расстояний.
# Финальная честность ("в базе ответа нет") на таких пограничных случаях
# держится на системном промпте куратора (этап 3), а не только на пороге.
MAX_DISTANCE = 1.32

_db: Chroma | None = None  # ленивая инициализация, чтобы не грузить БД при импорте


def get_db() -> Chroma:
    global _db
    if _db is None:
        if not CHROMA_DIR.exists():
            raise SystemExit("Индекс не найден. Сначала запустите: python index_kb.py")
        _db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=str(CHROMA_DIR),
        )
    return _db


def build_filter(module: int | None = None,
                 resource_type: str | None = None,
                 topic: str | None = None) -> dict | None:
    """Собирает where-фильтр Chroma из заданных условий.

    Одно условие передаётся как есть, несколько - оборачиваются в $and
    (таков синтаксис фильтров Chroma). topic ищет точное совпадение со
    значением поля topic в метаданных чанка (как оно записано в YAML-шапке
    файла) - это фильтр "равно", а не текстовый поиск по теме.
    """
    conditions = []
    if module is not None:
        conditions.append({"module": module})
    if resource_type is not None:
        conditions.append({"resource_type": resource_type})
    if topic is not None:
        conditions.append({"topic": topic})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_kb(query: str,
              k: int = 4,
              module: int | None = None,
              resource_type: str | None = None,
              topic: str | None = None,
              max_distance: float = MAX_DISTANCE,
              ) -> list[tuple[Document, float]]:
    """Семантический поиск с опциональной фильтрацией по метаданным.

    Возвращает список пар (чанк, расстояние), отфильтрованный по порогу
    релевантности. Пустой список = "в базе ответа нет" - этот сигнал
    использует логика честного отказа на этапе 3.
    """
    db = get_db()
    where = build_filter(module=module, resource_type=resource_type, topic=topic)
    results = db.similarity_search_with_score(query, k=k, filter=where)
    return [(doc, score) for doc, score in results if score <= max_distance]


def format_results(results: list[tuple[Document, float]]) -> str:
    """Человекочитаемый вывод результатов - для отладки и тестов."""
    if not results:
        return "Ничего релевантного не найдено (все чанки за порогом расстояния)."
    lines = []
    for doc, score in results:
        m = doc.metadata
        header = (f"[{score:.3f}] {m['source']} / чанк {m['chunk_index']} "
                  f"(модуль {m['module']}, {m['resource_type']}: {m['topic']})")
        preview = doc.page_content[:180].replace("\n", " ")
        lines.append(f"{header}\n    {preview}...")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск по базе знаний курса")
    parser.add_argument("query", help="вопрос студента")
    parser.add_argument("--module", type=int, default=None,
                        help="искать только в указанном модуле")
    parser.add_argument("--type", dest="resource_type", default=None,
                        help="тип ресурса: лекция / методичка / faq")
    parser.add_argument("--topic", default=None,
                        help="точное значение поля topic из YAML-шапки файла")
    parser.add_argument("-k", type=int, default=4, help="сколько чанков вернуть")
    args = parser.parse_args()

    found = search_kb(args.query, k=args.k, module=args.module,
                      resource_type=args.resource_type, topic=args.topic)
    print(format_results(found))
