"""
Этап 2. Скрипт индексации базы знаний.

Что делает:
1. Читает все .txt из папки knowledge_base/.
2. Отрезает YAML-шапку (frontmatter) и превращает её в метаданные.
3. Режет текст на чанки ~800 символов с перекрытием.
4. Считает эмбеддинги через OpenAI и пишет всё в локальную Chroma.

Запуск (нужен OPENAI_API_KEY в файле .env в корне проекта):
    python index_kb.py

Переиндексация после добавления/изменения файлов — просто повторный запуск:
скрипт удаляет старую коллекцию и строит её заново. Для учебной базы
из 8 файлов это занимает секунды и избавляет от логики "что изменилось".
"""

import os
import re
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # подхватывает OPENAI_API_KEY из .env в корне проекта

# ---------- Настройки ----------

BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "knowledge_base"      # папка с txt-файлами базы знаний
CHROMA_DIR = BASE_DIR / "chroma_db"       # сюда Chroma сохранит индекс
COLLECTION_NAME = "course_kb"
# Модель настраивается через EMBEDDING_MODEL в .env; значение по умолчанию
# ("text-embedding-3-small") зашито здесь как fallback на случай, если переменная
# не задана - дешёвая и достаточная для учебного проекта. Важно: если модель
# сменить, нужна полная переиндексация (build_index() всё равно сносит и строит
# коллекцию заново) - иначе часть чанков в Chroma останется с эмбеддингами от
# старой модели, и расстояния между чанками перестанут быть сравнимыми.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

CHUNK_SIZE = 800       # середина диапазона 500-1000 из ТЗ
CHUNK_OVERLAP = 150    # перекрытие, чтобы мысль не рвалась на границе чанков

# Frontmatter: блок между двумя строками "---" в начале файла
FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------- Чтение файлов ----------

def load_kb_file(path: Path) -> tuple[str, dict]:
    """Читает txt-файл, возвращает (текст без шапки, метаданные из шапки)."""
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(raw)
    if not match:
        raise ValueError(
            f"В файле {path.name} не найдена YAML-шапка (--- ... ---). "
            "Каждый файл базы знаний обязан начинаться с метаданных."
        )
    metadata = yaml.safe_load(match.group(1))
    body = raw[match.end():].strip()

    # Chroma принимает в метаданных только str/int/float/bool - проверим
    for key, value in metadata.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{path.name}: метаданное {key!r} имеет тип {type(value)}")

    metadata["source"] = path.name
    return body, metadata


# ---------- Чанкование ----------

def split_into_chunks(body: str, metadata: dict) -> list[Document]:
    """Режет текст на чанки, копируя метаданные файла в каждый чанк."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # "\n== " - маркер заголовка секции в наших лекциях:
        # сплиттер сначала попробует резать по границам секций,
        # и только потом - по абзацам, строкам и словам.
        separators=["\n== ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(body)
    docs = []
    for i, chunk in enumerate(chunks):
        meta = dict(metadata)              # копия, чтобы не делить один dict
        meta["chunk_index"] = i
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


# ---------- Основной сценарий ----------

def build_index() -> None:
    txt_files = sorted(KB_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"В {KB_DIR} нет .txt-файлов - нечего индексировать.")

    all_docs: list[Document] = []
    for path in txt_files:
        body, metadata = load_kb_file(path)
        docs = split_into_chunks(body, metadata)
        all_docs.extend(docs)
        print(f"  {path.name}: {len(docs)} чанков "
              f"(модуль {metadata.get('module')}, {metadata.get('resource_type')})")

    # Полная переиндексация: сносим старый индекс, строим новый
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"\nГотово: {len(all_docs)} чанков из {len(txt_files)} файлов -> {CHROMA_DIR}/")


if __name__ == "__main__":
    print("Индексация базы знаний...")
    build_index()
