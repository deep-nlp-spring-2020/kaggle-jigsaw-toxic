# kaggle-jigsaw-toxic

### Итоги участия в соревновании

Наше финальное место место в jigsaw:  
![Current place](/jigsaw_lb.png)

Команда For the Horde, 132 место, бронза (https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/leaderboard)

### Проделанная работа

#### QtRoS

Если кратко, то мой вклад заключается в применении модели USE (https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3).  
Модель `USE` позволяет трансформировать небольшой текст в вектор размерностью `512`. Доступна так же мультиязыковая модель, которая училась на `16` языках. Я предположил, что можно воспользоваться моделью (в базовой версии, без дообучения) для трансформирования всех доступных текстов (`use_baseline.py`), а поверх полученных эмбеддингов обучить классификатор на основе NN `use_pytorch.py`. Даже однослойная сеть (по сути - линейная регрессия) уже давала порядка 0.88 на валидации, что является отличным результатом для одной модели. Такой подход позволил эффективно обучать решение на домашней машине, без использования Kaggle или Collab. В следующей версии классификатор стал многослойным, а к данным добавились дополнительные примеры из расширенного датасета (`use_extra.py`). В итоге удалось добиться порядка 0.901 на валидации. Данная модель была смешана с популярным кернелом, и это послужило отправной точкой для выхода в топ 10%.

#### evilden
Мой вклад заключается в написании бейзлайна на `Bert multilingual` на pytorch(`bert_baseline.ipynb`). Был проведен ряд экспериментов для поиска лучших гипермараметров, а также конфигураций модели. 
Совместно с `@mtalimanchuk` был проведен поиск наиболее подходящих для данного соревнования дополнительных датасетов, а также анализ датасетов на корректность оценок токсичности. В результате этого был создан скрипт(`datasets_merge.ipynb`) для создания объединенного датасета из двух стандартных, с фильтрацией по оценке токсичности.
Также, совместно с `@mtalimanchuk` был произведен выбор наиболее успешных моделей, а также подбор коэффициентов для блендинга, в результате которого и были получены финальные сабмишны(`submissions/submission_max.csv` один из них).

