"""
Этап 4. Сборка ask_curator() - главной функции чат-бота куратора.

Пайплайн одного вопроса:
1. Классификация вопроса LLM'ом: организационный (расписание/дедлайны -> LMS)
   или учебный (всё остальное, включая правила ДЗ и оценки -> поиск по базе
   знаний, engine из этапа 2).
2. Сборка КОНТЕКСТа под найденный тип.
3. Подстановка student_level в системный промпт (этап 3).
4. Вызов LLM с системным промптом + few-shot примерами + КОНТЕКСТом + вопросом.
5. Логирование вызова (этап 5) - см. logger.py.

Использование:
    from curator import ask_curator
    answer = ask_curator("Когда дедлайн по hw3?", student_level="начинающий")
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from logger import log_interaction
from search_kb import search_kb

load_dotenv()  # подхватывает OPENAI_API_KEY (и CURATOR_CHAT_MODEL) из .env в корне проекта

BASE_DIR = Path(__file__).parent
PROMPTS_DIR = BASE_DIR / "prompts"
LMS_PATH = BASE_DIR / "knowledge_base" / "lms_stub.json"

# Модель настраивается через CURATOR_CHAT_MODEL в .env; значение по умолчанию
# ("gpt-4o-mini") зашито здесь как fallback на случай, если переменная не задана -
# дешёвой модели достаточно и для классификации, и для ответов.
CHAT_MODEL = os.getenv("CURATOR_CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT_TEMPLATE = (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")

# Классификация - отдельный, короткий вызов LLM, а не набор ключевых слов:
# вопросы про сроки перефразируются слишком по-разному ("когда дедлайн",
# "до какого числа сдавать", "во сколько вебинар по функциям"), чтобы
# надёжно ловить их регуляркой.
CLASSIFY_PROMPT = """Определи тип вопроса студента курса. Ответь ровно одним словом:
- lms - если вопрос про даты вебинаров, дедлайны заданий, сроки модулей или курса;
- kb - для всех остальных вопросов (темы лекций, код, правила сдачи ДЗ, оценки/апелляции, общие вопросы не по курсу).

Вопрос: {question}
Тип:"""


# Few-shot примеры для финального вызова (не для классификации) - тексты в
# prompts/few_shot_examples.txt, а не здесь: та же логика, что и у
# system_prompt.txt/few_shot.txt - промпты редактируются без правки
# Python-кода и хранятся в единообразном текстовом формате (та же разметка
# "=== Пример N ===", что и в few_shot.txt), а не как отдельный JSON.
# Отобраны 3 из 7 пар в prompts/few_shot.txt - не все семь, чтобы не
# раздувать каждый вызов лишними токенами; выбраны по разным типам
# поведения, а не дублируют друг друга:
#   1) обычный хороший ответ - эталон формата/ссылки на источник, без него
#      модель видела бы в контексте только отказы;
#   2) честный отказ (Django) - целится в найденную слабость (docs/stage4.md):
#      модель иногда сначала пишет "не нашёл ответа", но затем всё равно
#      поясняет из общих знаний;
#   3) отказ вне компетенции (смена оценки) - другой тип отказа, под
#      эмоциональным давлением студента.
# Контекст в каждом примере - реальный чанк из search_kb(), а не выдуманный,
# чтобы формат в примере совпадал с тем, что модель увидит в проде. Берём
# только пару вопрос/хороший-ответ - "плохой ответ" и разбор из
# few_shot.txt не передаются: это комментарий для человека, а не то, что
# нужно показывать модели в контексте.
FEW_SHOT_BLOCK_RE = re.compile(
    r"КОНТЕКСТ:\n(?P<context>.*?)\n\nВОПРОС:\n(?P<question>.*?)\n\nОТВЕТ:\n(?P<answer>.*)",
    re.DOTALL,
)


def load_few_shot_examples() -> list[dict]:
    """Парсит prompts/few_shot_examples.txt в список {context, question, answer}.

    Формат - блоки "=== Пример N ===" с разделами КОНТЕКСТ:/ВОПРОС:/ОТВЕТ:,
    та же разметка, что уже используется в few_shot.txt.
    """
    raw = (PROMPTS_DIR / "few_shot_examples.txt").read_text(encoding="utf-8")
    blocks = re.split(r"=== Пример \d+ ===\n", raw)[1:]  # [0] - пусто, до первого маркера
    examples = []
    for i, block in enumerate(blocks, start=1):
        match = FEW_SHOT_BLOCK_RE.match(block.strip())
        if not match:
            raise ValueError(f"Не удалось разобрать пример {i} в few_shot_examples.txt")
        examples.append({
            "context": match.group("context").strip(),
            "question": match.group("question").strip(),
            "answer": match.group("answer").strip(),
        })
    return examples


FEW_SHOT_EXAMPLES = load_few_shot_examples()


def build_few_shot_messages() -> list[dict]:
    """Разворачивает FEW_SHOT_EXAMPLES в чередующиеся user/assistant сообщения.

    Формат user-сообщения такой же, как у реального вопроса (КОНТЕКСТ +
    ВОПРОС СТУДЕНТА), чтобы модель видела ровно тот же шаблон входа, что и
    в проде, а не упрощённую заглушку.
    """
    messages = []
    for ex in FEW_SHOT_EXAMPLES:
        user_content = f"КОНТЕКСТ:\n{ex['context']}\n\nВОПРОС СТУДЕНТА:\n{ex['question']}"
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ex["answer"]})
    return messages


def classify_question(question: str, llm: ChatOpenAI) -> str:
    """Возвращает "lms" или "kb". При любом неожиданном ответе LLM - откатываемся на "kb"."""
    response = llm.invoke(CLASSIFY_PROMPT.format(question=question))
    label = response.content.strip().lower()
    return "lms" if "lms" in label else "kb"


def build_kb_context(question: str) -> tuple[str, list[str], str | None]:
    """Контекст из векторного поиска (этап 2) - полный текст найденных чанков с указанием источника.

    Возвращает (текст контекста, список файлов-источников, тема топ-1 чанка) -
    вторые два значения нужны только для лога (этап 5), не для самого промпта.
    """
    results = search_kb(question)
    if not results:
        return "", [], None
    blocks = []
    sources = []
    for doc, _score in results:
        m = doc.metadata
        header = f"[{m['resource_type']}, модуль {m['module']}, тема: {m['topic']}] (источник: {m['source']})"
        blocks.append(f"{header}\n{doc.page_content}")
        sources.append(m["source"])
    top_topic = results[0][0].metadata.get("topic")
    return "\n\n---\n\n".join(blocks), sources, top_topic


def build_lms_context() -> str:
    """Контекст для организационных вопросов - вся заглушка LMS целиком.

    Файл маленький (расписание + 6 заданий), поэтому проще отдать его
    целиком в контекст, чем городить отдельный парсер/поиск по JSON.
    """
    return LMS_PATH.read_text(encoding="utf-8")


def ask_curator(question: str, student_level: str = "начинающий") -> str:
    """Главная функция куратора: вопрос -> тип -> контекст -> ответ LLM -> лог."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    question_type = classify_question(question, llm)
    if question_type == "lms":
        context = build_lms_context()
        sources = ["lms_stub.json"]
        topic = "LMS"
        context_found = True
    else:
        context, sources, topic = build_kb_context(question)
        context_found = bool(context)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(student_level=student_level)
    user_message = (
        f"КОНТЕКСТ:\n{context or '(ничего релевантного не найдено)'}\n\n"
        f"ВОПРОС СТУДЕНТА:\n{question}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(build_few_shot_messages())
    messages.append({"role": "user", "content": user_message})

    response = llm.invoke(messages)
    answer = response.content

    log_interaction(
        question=question,
        student_level=student_level,
        question_type=question_type,
        topic=topic,
        sources=sources,
        context_found=context_found,
        answer=answer,
    )
    return answer


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "Когда дедлайн по hw3?"
    level = sys.argv[2] if len(sys.argv) > 2 else "начинающий"
    print(ask_curator(q, student_level=level))
