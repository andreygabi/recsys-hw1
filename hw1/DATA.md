# Данные

Студент: **Габидулин Андрей**.

Датасет: **Amazon Review Data (2018), Musical Instruments, 5-core**.

Первоисточник: [Jianmo Ni / Julian McAuley, UC San Diego](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/).

- [Отзывы, официальный gzip JSONL](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz).
- [Метаданные товаров, официальный gzip JSONL](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz).

Загрузка: `python data.py` из `hw1/` или запуск ноутбука. Файлы сохраняются в `.cache/raw/`, проверяются по SHA256 из `data.py`. Также можно скачать их вручную в эту директорию с исходными именами. Данные в Git не включены.

Исходно 231 392 отзыва; после дедупликации пар — 219 156. Используем 149 744 взаимодействия, 20 159 пользователей, 6000 товаров. Поля: `reviewerID`, `asin`, `overall` (1–5), `unixReviewTime`; метаданные `brand`, последняя `category`, `title`.

Из первых 70% времени выбираются 6000 самых частых товаров; итеративно оставляются пользователи/товары с ≥ 3 взаимодействиями в train. Отбор детерминирован; подробности и все исключения — в `experiments.ipynb`. Исходная 5-core использует весь временной период: это ограничение отбора, а не гарантия отсутствия смещения.
