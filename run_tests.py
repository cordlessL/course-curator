"""
Этап 2. Контрольный прогон поиска: 15 тестовых вопросов.

Для каждого вопроса задан ожидаемый источник (doc_id файла, где лежит
ответ). Метрика - hit@k: попал ли нужный документ в топ-k найденных
чанков. Для учебного проекта целимся в hit@4 >= 0.85.

Три особых случая в конце списка:
- вопросы с фильтром по метаданным (проверяем, что фильтр работает);
- вопрос, ответа на который в базе НЕТ (expected=None) - проверяем,
  что порог расстояния отсекает нерелевантные чанки. Это фундамент
  для честного "в базе ответа нет" на этапе 3.

Запуск:
    python run_tests.py
"""

from search_kb import search_kb

# (вопрос, ожидаемый doc_id или None, фильтры)
TEST_CASES = [
    # --- обычный семантический поиск ---
    ("Как узнать тип переменной?",                        "lec_m1_variables", {}),
    ("Почему нельзя сложить строку и число?",             "lec_m1_variables", {}),
    ("Чем elif отличается от else?",                      "lec_m2_conditions_loops", {}),
    ("Как остановить бесконечный цикл while?",            "lec_m2_conditions_loops", {}),
    ("В чём разница между print и return?",               "lec_m3_functions", {}),
    ("Что вернёт функция без return?",                    "lec_m3_functions", {}),
    ("Как перевернуть список?",                           "lec_m4_collections", {}),
    ("Как безопасно достать значение из словаря, если ключа может не быть?",
                                                          "lec_m4_collections", {}),
    ("Как правильно открыть файл с русским текстом?",     "lec_m5_files_errors", {}),
    ("Что делать, если программа падает с ValueError при вводе?",
                                                          "lec_m5_files_errors", {}),
    ("Python пишет, что не является внутренней или внешней командой",
                                                          "method_env_setup", {}),
    ("Сколько баллов снимают за сдачу после дедлайна?",   "method_homework_rules", {}),
    ("Можно ли пользоваться ChatGPT в домашках?",         "faq_general", {}),

    # --- поиск с фильтрами по метаданным ---
    ("что такое срез",            "lec_m4_collections",   {"module": 4}),
    ("правила пересдачи заданий", "method_homework_rules", {"resource_type": "методичка"}),

    # --- вопроса нет в базе: ожидаем пустой результат ---
    ("Как развернуть Django-приложение на сервере?",      None, {}),
]

K = 4


def run() -> None:
    hits = 0
    scored = 0  # вопросы, по которым есть ожидаемый ответ
    failures = []

    for question, expected, filters in TEST_CASES:
        results = search_kb(question, k=K, **filters)
        found_ids = [doc.metadata["doc_id"] for doc, _ in results]

        if expected is None:
            # Успех = поиск честно вернул пустоту
            ok = len(results) == 0
            status = "OK (пусто, как и должно)" if ok else f"FAIL: нашлось {found_ids}"
            if not ok:
                failures.append((question, expected, found_ids))
            print(f"[{'+' if ok else '!'}] {question}\n      {status}")
            continue

        scored += 1
        ok = expected in found_ids
        if ok:
            hits += 1
        else:
            failures.append((question, expected, found_ids))
        top = f"{found_ids[0]} ({results[0][1]:.3f})" if results else "ничего"
        print(f"[{'+' if ok else '!'}] {question}\n      топ-1: {top}, ожидали: {expected}")

    print("\n" + "=" * 60)
    print(f"hit@{K}: {hits}/{scored} = {hits / scored:.0%}")
    if failures:
        print("\nПромахи (кандидаты на доработку чанкования/порога/формулировок):")
        for q, exp, got in failures:
            print(f"  - {q!r}: ожидали {exp}, получили {got}")
    else:
        print("Все тесты пройдены.")


if __name__ == "__main__":
    run()
