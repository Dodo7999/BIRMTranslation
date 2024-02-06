import random
import re


def mask_spans(text, r=0.25, u=3):
    min_len = int(u * 0.7)
    max_len = int(u * 1.3)

    span_len = random.randint(min_len, max_len) - 1

    # создаем регулярное выражение для нахождения спанов
    pattern = re.compile(r'\b\w+(?!\s*:)(?:\W+\w+(?!\s*:)){0,' + str(span_len) + r'}\b')

    # находим все спаны в тексте
    spans = pattern.findall(text)
    spans = [i for i in spans if len(i.split()) >= min_len]

    # определяем количество спанов, которые нужно замаскировать
    num_to_mask = int(len(spans) * r)

    # выбираем случайные спаны для замаскировки
    spans_to_mask = random.sample(spans, num_to_mask)

    # создаем словарь, в котором ключи - замаскированные спаны, а значения - токены
    masked_spans = {}
    for i, span in enumerate(spans_to_mask):
        masked_spans[span] = f"<extra_id_{i}>"

        # заменяем спаны на соответствующие токены в тексте
    masked_text = text
    for span, token in masked_spans.items():
        masked_text = masked_text.replace(span, token, 1)

        # создаем текст с пропущенными спанами
    missing_spans = []
    for i, span in enumerate(spans_to_mask):
        missing_spans.append(f"<extra_id_{i}>{span}")
    missing_text = ' '.join(missing_spans)

    return [masked_text, missing_text]
