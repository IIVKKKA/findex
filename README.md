# findex
[![CI](https://github.com/IIVKKKA/findex/actions/workflows/ci.yml/badge.svg)](https://github.com/IIVKKKA/findex/actions/workflows/ci.yml)

Власна пошукова система, яку будуємо поетапно (курс Python, лаби 1–8).
Цей тег: **lab-01** — лінивий потік документів, потоковий токенізатор, статистика корпусу.

## Корпус

Тексти з [Project Gutenberg](https://www.gutenberg.org/) (публічне надбання), ~58 МБ, 91 книга.
`data/` у `.gitignore`, тож корпус у репозиторій не входить. Завантажити:

```bash
uv sync
uv run python scripts/download_corpus.py          # -> data/gutenberg/pgN.txt
```

Скрипт качає книги по порядку id, відкидає ліцензійний заголовок/підвал Gutenberg і зупиняється на ~55 МБ.

## Запуск

```bash
uv run python -m findex.stats data/gutenberg              # lazy, повний корпус
uv run python -m findex.stats data/gutenberg --limit 20   # перші 20 документів (itertools.islice)
uv run python -m findex.stats data/gutenberg --eager      # «жадібна» версія для порівняння
uv run pytest && uv run ruff check .
```

Виводить: кількість документів, токенів, розмір словника, топ-50 термів, час і пікову пам'ять (`tracemalloc`).

## Правила токенізатора ([tokenize.py](src/findex/tokenize.py))

1. `unicodedata.normalize("NFC")` — `e` + U+0301 і `é` стають одним рядком.
2. `str.casefold()`, а не `lower()` — `Straße` → `strasse`.
3. `re.finditer` (лінивий) по `[^\W_]+(?:['’ʼ-][^\W_]+)*`; `\w` Unicode-aware, тож кирилиця й акценти працюють.

| Що | Рішення |
|---|---|
| Апострофи (`'`, `’`, `ʼ`) всередині слова | лишаються й нормалізуються до `'`: `don't`, `п'ять`; на краях (лапки) відкидаються |
| Дефіси | всередині слова лишаються (`well-known`); на краях або з пробілами — роздільник |
| Цифри | частина слова (`abc123`); `3.14` → `3`, `14`; `1990-2000` — один токен |
| `_` | роздільник |

## Вимірювання: eager проти lazy

Увесь корпус (91 документ, 58 МБ, 9 752 266 токенів, словник 379 464), AMD Ryzen 5 5600X, Python 3.13.
Час і пам'ять заміряно з увімкненим `tracemalloc` (він уповільнює виконання, тож абсолютний час завищений).

| Версія | Документів | Пікова пам'ять | Час |
|---|---|---|---|
| eager (списки) | 91 | 604.0 MiB | 41.9 с |
| lazy (генератори) | 91 | 88.8 MiB | 46.2 с |

**Куди пішла пам'ять.** Eager-версія спершу тримає в пам'яті тексти всіх 91 документа, а потім для кожного ще й список його токенів. Кожен токен — окремий Python-рядок (десятки байтів накладних витрат) плюс 8-байтовий вказівник у списку, а токенів майже 10 мільйонів, тому пік зростає до ~600 МіБ і росте разом із корпусом. Lazy-версія в кожен момент тримає один документ (найбільший — кілька МБ) і активний токен, а накопичує лише `Counter` зі словником; тому пік ~89 МіБ визначається розміром словника, а не корпусу. Нуля вона не дає, бо `Counter` на 379 тис. унікальних термів сам займає десятки мегабайт, плюс у пам'яті лежить поточний документ. Час майже однаковий: токенізація — та сама робота, генератори лише не створюють проміжних списків (у lazy трохи повільніше через накладні витрати `yield` та лічильник токенів у циклі).

## Структура

```
src/findex/corpus.py     iter_documents(root) -> Iterator[Document]  (.txt-дерево або .jsonl)
src/findex/tokenize.py   tokenize(text) -> Iterator[str]
src/findex/stats.py      python -m findex.stats
tests/test_tokenize.py   10 перевірок (регістр, кирилиця, combining mark, пунктуація, порожній рядок…)
scripts/download_corpus.py
```


---

# Лаба 2 — інвертований індекс (тег `lab-02`)

Індекс будується за один прохід по лінивому конвеєру з лаби 1 (`iter_documents` → `tokenize`), документ за документом:
`Counter` поточного документа скидається у `defaultdict` постингів, коли документ закінчився. `doc_id` зростають,
тому списки постингів відсортовані «з коробки».

```bash
uv run python -m findex.index data/gutenberg --out index.bin                      # pickle, slots
uv run python -m findex.index data/gutenberg --out index.bin --layout array --format binary --positions
uv run python -m findex.search index.bin "whale NOT ship"                         # merge (за замовчуванням)
uv run python -m findex.search index.bin "whale NOT ship" --engine set
```

Обидві команди друкують час і пік пам'яті (`tracemalloc`). Прапорці побудови: `--layout {dataclass,slots,array}`,
`--format {pickle,binary}`, `--positions` (позиції токенів), `--limit N`.

**Структури.** `Posting(doc_id, tf, positions=())` і `DocMeta(path, title)` — `@dataclass(frozen=True, slots=True)`, хешовані
(лежать у `set`). Запити: `a b` = AND, явні `OR` / `NOT`, обчислення зліва направо; `a NOT b` = `a AND NOT b`.
Merge — два вказівники по відсортованих списках (`and_merge`, `or_merge`, `not_merge`); альтернатива — операції над `set`.

## Вимірювання (корпус: 91 книга, 58 МБ, Ryzen 5 5600X, Python 3.13)

Скрипти: `scripts/bench_lab2.py`, `scripts/mem_synthetic.py`.

### A. Пам'ять: три представлення постингів (пік під `tracemalloc` при побудові)

| Представлення постингів | Пік пам'яті (build) | Файл індексу (pickle) | Час load (pickle) |
|---|---|---|---|
| `list[Posting]`, звичайний `@dataclass` | 166.3 MiB | 23.58 MiB | 3.56 с |
| `list[Posting]`, `slots=True` | 131.8 MiB | 18.40 MiB | 8.37 с |
| пари `array('I')`, без об'єктів | 141.2 MiB | 30.04 MiB | 3.91 с |

На *цьому* корпусі масиви не виграли — це чесний результат, а не помилка заміру: у корпусі лише 91 документ і 379 464
терми, і більшість термів трапляється в одному-двох документах. Тому кожен терм має 1–2 постинги, і важить не вартість
одного постингу, а **накладні витрати на терм**. Перша версія моєї `array`-реалізації взагалі була гіршою (184.7 MiB):
кожен терм мав `ArrayPostings` з `__dict__` плюс порожній третій масив позицій. Після `slots=True` на `ArrayPostings` і
ліниво створюваного масиву позицій стало 141.2 MiB. Решта піку — спільна для всіх трьох рядків: рядки-терми
(≈ 380 тис. об'єктів `str`), словники `doc_lengths`/`doc_meta`, `Counter` документа.

Щоб побачити вартість *одного постинга*, окремо виміряно довгі списки (1 000 000 постингів у 1 000 термів, без рядків-термів):

| Представлення | Разом | Байтів на постинг |
|---|---|---|
| звичайний `@dataclass` | 122.7 MiB | 128.7 |
| `slots=True` | 84.6 MiB | 88.7 |
| пари `array('I')` | 8.0 MiB | 8.4 |

**Куди йдуть байти.** Звичайний dataclass-екземпляр — це заголовок об'єкта (≈ 16 Б) + вказівники на `__dict__` і `__weakref__` +
власний `dict` з трьома ключами (≈ 100+ Б), плюс 8 Б вказівника в списку; цілі `doc_id`/`tf` — окремі об'єкти `int` (≈ 28–32 Б,
малі числа кешуються, а великі `doc_id` — ні). `slots=True` прибирає `__dict__` і `__weakref__`: поля лежать прямо в об'єкті
(по 8 Б), тож ≈ 40 Б на постинг економії, але `int`-и лишаються об'єктами (≈ 88.7 Б). `array('I')` зберігає два беззнакові
32-бітні числа підряд: 4 + 4 = 8 Б на постинг (8.4 з урахуванням заголовків масивів) — у ≈ 10–15 разів менше. Ціна масивів:
об'єкт `Posting` доводиться створювати на льоту при читанні, а `frozen`/хешованість зникають; на «розріджених» корпусах
(багато унікальних термів) виграш з'їдають накладні витрати самих масивів (≈ 64 Б на порожній `array`).

### B. Формати збереження

| Формат | Розмір файлу | Save | Load |
|---|---|---|---|
| pickle (layout `slots`) | 18.40 MiB | 4.29 с | 8.37 с |
| pickle (layout `array`) | 30.04 MiB | 3.30 с | 3.91 с |
| binary (`array` + JSON-заголовок) | 13.40 MiB | 0.62 с | 2.65 с |

Бінарний формат (`store.py`): `FIDX1`, довжина й JSON-заголовок (метадані документів, список термів і кількість постингів),
далі «сирі» байти `array('I')`: довжини документів, потім для кожного терма `doc_ids`, `tf`, за потреби позиції.
Він найменший і найшвидший, і — головне — його завантаження читає тільки числа й JSON.
pickle у layout `array` більший за pickle у `slots`: pickle серіалізує кожен `array` окремим викликом реконструктора з
кодом типу й байтами, а для ≈ 380 тис. масивів із одним елементом ці накладні витрати переважають самі дані.

> ⚠️ **Безпека `pickle`.** `pickle.load` виконує код, закладений у файл (через `__reduce__` можна викликати будь-яку функцію,
> наприклад `os.system`). Завантаження чужого файлу індексу = віддалене виконання коду. Тому в `store.load` є коментар про це;
> файли, зроблені не вами, відкривайте лише бінарним форматом.

### C. Ціна позицій (`--positions`, layout `slots`, pickle)

| Варіант | Пік пам'яті (build) | Файл |
|---|---|---|
| без позицій | 131.8 MiB | 18.40 MiB |
| з позиціями | 500.6 MiB | 57.88 MiB |

Позиції збільшують пік у ≈ 3.8 раза, файл — у ≈ 3.1 раза: на кожне входження (≈ 9.75 млн токенів) додається ціле число.
У layout `array` позиції дешевші (4 Б на входження): бінарний індекс з позиціями — 50.6 MiB.

### D. merge проти set (AND/OR двох термів; час однієї операції)

| Випадок | Довжини списків | Op | merge | set | Переможець |
|---|---|---|---|---|---|
| два найчастіші: `the`, `of` | 91, 91 | AND | 8.17 мкс | 4.09 мкс | set |
| два найчастіші: `the`, `of` | 91, 91 | OR | 8.21 мкс | 4.26 мкс | set |
| два найрідші: `achsa`, `zebedee's` | 1, 1 | AND | 0.18 мкс | 0.29 мкс | merge |
| два найрідші: `achsa`, `zebedee's` | 1, 1 | OR | 0.28 мкс | 0.30 мкс | merge |
| частий + рідкий: `the`, `achsa` | 91, 1 | AND | 0.20 мкс | 1.26 мкс | merge |
| частий + рідкий: `the`, `achsa` | 91, 1 | OR | 0.60 мкс | 2.26 мкс | merge |
| синтетика 500k × 500k | 500 000, 500 000 | AND | 95.0 мс | 97.1 мс | merge (майже нічия) |
| синтетика 500k × 500k | 500 000, 500 000 | OR | 123.9 мс | 144.5 мс | merge |
| синтетика 500k × 50 | 500 000, 50 | AND | 42.3 мс | 22.4 мс | set |
| синтетика 500k × 50 | 500 000, 50 | OR | 58.5 мс | 62.7 мс | merge (майже нічия) |

Merge — цикл на чистому Python, `set(a) & set(b)` працює в C, але спершу будує хеш-таблиці. Тому на коротких списках
середнього розміру (91 × 91) виграє `set`, на списках довжини 1 — merge (немає вартості побудови множин), на двох величезних —
приблизно нічия (обидва лінійні; `set` ще й змушений зрештою `sorted(...)`). Якщо один список у 10 000 разів коротший,
merge усе одно обходить усі 500 тис. елементів, а `set` ітерує по меншому — тут потрібні стрибки (skip-pointers / бінарний пошук).
Реальна відповідь — «залежить від форми даних».



---

# Лаба 3 — рейтинг і об'єктна модель (тег `lab-03`)

Словники з лаби 2 тепер — об'єкт `Index`; булів пошук став рейтинговим (TF-IDF / BM25), з мовою запитів, сніпетами,
`@timed` і `lru_cache`.

```bash
# індекс з позиціями (потрібні для фраз "...") — layout array, бінарний формат, 50.6 MiB
uv run python -m findex.index data/gutenberg --out index.bin --layout array --positions --format binary
uv run python -m findex.search index.bin '(whale OR harpoon) "white whale"' --scorer bm25 -v
```

## `Index` — API

`Index` наслідує `collections.abc.Mapping[str, list[Posting]]` (файл `index.py`).

```python
from findex.store import open_index

with open_index("index.bin") as index:      # завантажує на вході, звільняє в finally (і при винятку)
    len(index)                  # 379464 термів
    "whale" in index            # True  (не матеріалізує постинги)
    index["whale"][0]           # Posting(doc_id=..., tf=..., positions=(...))
    index["no-such-term"]       # KeyError
    for term in index: ...      # ітерація по термах
    index.num_docs, index.avg_doc_length, index.df("whale"), index.doc_length(6)
    repr(index)                 # Index(terms=379464, docs=91, layout='array', positions=True)
```

`avg_doc_length` — `functools.cached_property`. `Mapping` задає `__eq__` через порівняння всіх елементів, що робить
об'єкт нехешованим; оскільки `lru_cache` потребує хеш, `Index` порівнюється й хешується за ідентичністю.

## Scorer і формули (`scoring.py`)

`Scorer` — `typing.Protocol` з одним методом `score(term, posting, index) -> float`; `TfIdf` і `BM25` не наслідують нічого й
взаємозамінні (duck typing).

* **TF-IDF:** `score = (1 + ln tf) · ln(N / df)` — сублінійний tf, натуральний логарифм. Терм, що є в усіх документах, дає 0.
* **BM25** (`k1=1.5`, `b=0.75`): `idf = ln(1 + (N − df + 0.5)/(df + 0.5))` (варіант Lucene, ≥ 0);
  `score = idf · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))`.

`search(index, query, scorer=BM25(), k=10)` (`ranking.py`) розбирає запит, обчислює множину збігів, додає бали по термах
лише для документів цієї множини і бере топ-k через `heapq.nlargest` (O(n log k), тримає k елементів, на відміну від
`sorted(...)[:k]` — O(n log n)). Результат — `SearchResult(score, doc_id, title)` з `order=True`, тож `sorted(results)` працює.

## Мова запитів (`query.py`)

Рекурсивний спуск: `or_expr → and_expr ('OR' and_expr)*`, `and_expr → not_expr ('AND'? not_expr)*`,
`not_expr → 'NOT' atom | atom`, `atom → слово | "фраза" | '(' or_expr ')'`. Вузли `Term / Phrase / And / Or / Not` мають
перевантаження `&`, `|`, `~` та `evaluate(index) -> set[doc_id]`. Слова проходять той самий `tokenize`, що й індекс.
`a OR b c` → `Or(a, And(b, c))`: `AND` зв'язує сильніше за `OR`. `Phrase` перевіряє послідовні позиції і вимагає індекс з
`--positions`. Тести парсера порівнюють **дерева**, а не рядки (`tests/test_ranking.py`).

## Таблиця 1 — три перевірки рейтингу (BM25, повний корпус)

| # | Перевірка | Що спостерігаємо |
|---|---|---|
| 1 | Рідкий терм вище частого | `mersenne` (df=1, документ 59, tf=2) → **8.492**; `the` (df=91, tf=64 113!) → **0.014**. У запиті `the OR mersenne` перший — док. 59. |
| 2 | 20-те повторення майже нічого не додає | Для `whale` у документі середньої довжини: tf=1 → 2.007; tf=2 → 2.862; tf=5 → 3.846; tf=10 → 4.344; tf=20 → 4.645 (+0.301); tf=40 → 4.811 (+0.166). Крива виходить на межу `idf·(k1+1)`. |
| 3 | Короткий документ з одним входженням вище довгого | Один `whale`: найкоротший док. 53 (2 598 токенів) → **3.559**; найдовший док. 1 (964 721 токен, Shakespeare) → **0.434**. |

(Перевірки 2–3 змінюють tf/довжину в самій формулі при реальних статистиках індексу: на корпусі немає пари документів, що
різняться лише одним фактором.) Скрипт: `scripts/eval_lab3.py`.

## Таблиця 2 — precision@5, 10 запитів, TF-IDF проти BM25

Мітки «що має бути в топ-5» поставлено **до** запуску скорерів, за змістом книг (`doc_id` → `data/gutenberg/pgN.txt`:
6 Moby-Dick, 74 Frankenstein, 87 The Monster Men, 11/17 Paradise Lost, 48 Paradise Regained, 30/39/65/70/46 тексти про
Інтернет, 20/28 перепис США 1990, 5/38 CIA Factbook, 55/59 прості числа, 0/21 Біблія, 8 Книга Мормона, 89 Tale of Two
Cities, 50 Scarlet Pimpernel, 90/14 Дуґлас, 57 Black Experience, 35/42 Anne, 9 Federalist, 80 Lincoln).

| # | Запит | Релевантні | TF-IDF топ-5 | P@5 | BM25 топ-5 | P@5 |
|---|---|---|---|---|---|---|
| 1 | `frankenstein OR monster` | 74, 87 | 74, 1, 6, 87, 13 | 0.4 | 74, 87, 6, 3, 27 | 0.4 |
| 2 | `paradise OR satan OR eden` | 11, 17, 48 | 11, 17, 21, 0, 8 | 0.4 | 11, 17, 48, 8, 24 | 0.6 |
| 3 | `internet OR email OR network` | 30, 39, 46, 65, 70 | 70, 25, 29, 39, 65 | 0.6 | 70, 25, 29, 39, 46 | 0.6 |
| 4 | `census OR population` | 5, 20, 28, 38 | 28, 20, 77, 38, 16 | 0.6 | 28, 20, 9, 57, 38 | 0.6 |
| 5 | `prime OR primes OR mersenne` | 55, 59 | 59, 72, 77, 38, 55 | 0.4 | 59, 72, 55, 38, 5 | 0.4 |
| 6 | `lord OR god OR commandments` | 0, 8, 21 | 21, 0, 8, 1, 18 | 0.6 | 8, 0, 21, 48, 88 | 0.6 |
| 7 | `guillotine OR paris OR revolution` | 50, 89 | 50, 89, 1, 24, 60 | 0.4 | 50, 89, 60, 24, 71 | 0.4 |
| 8 | `slavery OR freedom OR slaves` | 14, 57, 90 | 57, 14, 76, 1, 90 | 0.6 | 57, 14, 61, 90, 76 | 0.6 |
| 9 | `anne OR gilbert OR marilla` | 35, 42 | 35, 37, 42, 1, 77 | 0.4 | 35, 42, 37, 16, 38 | 0.4 |
| 10 | `constitution OR union OR states` | 9, 80 | 9, 77, 38, 16, 5 | 0.2 | 9, 38, 77, 16, 5 | 0.2 |
| | **Середнє** | | | **0.46** | | **0.48** |

Як читати: P@5 тут обмежене зверху числом розмічених документів (у запиті з 2 релевантними максимум 0.4 — його досягнуто у
запитах 1, 5, 7, 9). Єдина різниця між скорерами — запит 2: BM25 підняв «Paradise Regained» (48) в топ-5, а TF-IDF віддав
місця великим книгам (Біблія, Книга Мормона) — ймовірно, через відсутність нормалізації за довжиною. Запит 10 найгірший:
`union` і `states` — звичайні слова в десятках документів, і Lincoln (80) не потрапив у топ-5. **Оцінка маленька й шумна**:
10 запитів, 91 документ, мітки поставив один автор (і я лише здогадуюсь про релевантність книг, не читав їх повністю),
різниця 0.46 проти 0.48 — це один документ в одному запиті, а не доказ переваги BM25.

## Таблиця 3 — `@timed` і `lru_cache` у логах

Запит `python OR "event loop" OR whale`, двічі підряд (`scripts/eval_lab3.py`; у CLI — прапорець `-v`):

```
INFO findex.ranking: query 'python OR "event loop" OR whale': 16 matches (cache miss)
INFO findex.timing: search took 0.66 ms
INFO findex.ranking: query 'python OR "event loop" OR whale': 16 matches (cache hit)
INFO findex.timing: search took 0.23 ms
CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
```

`@lru_cache(maxsize=256)` стоїть на `match_ids(index, query) -> frozenset[doc_id]` (парсинг + обчислення збігів), тож повторний
запит пропускає саме цю частину; підрахунок балів усе одно виконується. `@timed` (з `functools.wraps`, логує в `finally`) стоїть
на `search`, `build_index`, `load`.

## Приклад сніпета

Запит `(whale OR harpoon) "white whale"` → документ 6 «MOBY-DICK;», бал 13.223:

> ...e whole crew were half suffocated as they were tossed helter-skelter into the **white** curdling cream of the squall. Squall, **whale**, and **harpoon** had all blended toget...

Вікно ±80 символів навколо входження, що покриває найбільше різних термів запиту; терми підсвічено `**...**`.



---

# Лаба 4 — типи, тести, пакування, CLI (тег `lab-04`, реліз `v0.4.0`)

Тепер `findex` ставиться однією командою й працює з будь-якого каталогу:

```bash
uv tool install .                     # або: pip install findex-0.4.0-py3-none-any.whl (з GitHub Release)
findex index data/gutenberg --out index.bin --positions     # смуга прогресу rich (у stderr)
findex search index.bin 'whale AND (harpoon OR ship) NOT "moby dick"' -k 5   # таблиця rich
findex search index.bin whale --json > hits.json            # чистий JSON у stdout
findex stats index.bin
findex -v search ...    # логи INFO у stderr;  -vv — DEBUG
```

* **Помилки.** Очікувана помилка (немає файлу індексу, зламаний запит, пошкоджений файл, немає корпусу) — один рядок
  `error: ...` у stderr і код виходу 1, без стека викликів. Діагностика йде в stderr через `logging`,
  логер налаштовується один раз у колбеку CLI (`cli.py`); результати — у stdout, тому `--json` можна перенаправляти.
* **Типи.** `typeCheckingMode = "strict"` для `src/` і `tests/`, `pyright`: **0 помилок**. `Scorer` — `typing.Protocol`
  (обидві реалізації перевіряються структурно, без наслідування); генератори позначені `Iterator`/`Generator`; `engine`,
  `scorer`, `layout`, `format` — `Literal`. Скрипти вимірювань у `scripts/` — одноразові й до перевірки не входять.
  Кожен `# type: ignore` має коментар, навіщо він (напр. присвоєння полю frozen-датакласу в тесті, що очікує `AttributeError`).
* **Пакування.** `pyproject.toml`: метадані, залежності (`typer`, `rich`), група `dev`
  (`pytest`, `pytest-cov`, `hypothesis`, `pyright`, `ruff`), `[project.scripts] findex = "findex.cli:app"`; `uv.lock` закомічено.
  Версія береться з метаданих пакета (`findex --version` → `findex 0.4.0`).
* **CI.** GitHub Actions на кожен push і PR: `uv sync`, `ruff`, `pyright`, `pytest --cov` (бейдж вгорі README).
* **Реліз.** Тег `v0.4.0`, wheel `findex-0.4.0-py3-none-any.whl` прикріплено до GitHub Release; перевірено в чистому
  віртуальному оточенні: `pip`-встановлення wheel → `findex --help` працює.

## Тести і покриття

150 тестів у `tests/` (разом із параметризацією): токенізатор (parametrize, Unicode), збірка індексу на малому корпусі
(`conftest.py`, усі три layout), merge, парсер запитів (порівняння дерев), обидва скорери (три перевірки з лаби 3),
save/load (pickle і бінарний), CLI через `typer.testing.CliRunner`. Три властивості Hypothesis (`tests/test_properties.py`):
`load(save(index))` дає той самий індекс; постінги завжди відсортовані, а сума `tf` документа = його довжині;
merge збігається з операціями над `set`. Тести довші за секунду позначено `@pytest.mark.slow`
(`pytest -m "not slow"` їх пропускає).

```bash
uv run pytest --cov            # покриття: 85 % (гілки включено)
uv run ruff check . && uv run pyright
```

**Що навмисно лишилось поза покриттям:** застарілі точки входу `python -m findex.index` / `python -m findex.search`
(argparse; з лаб 1–2, їх замінює `findex`; код merge/пошуку звідти покрито окремо), `python -m findex.stats`
(`main()` лаби 1; перевіряється лише порівняння lazy/eager), скрипт завантаження корпусу (мережа), рідкісні гілки читання
`.jsonl` і недекодованих файлів та гілки `__main__`.
