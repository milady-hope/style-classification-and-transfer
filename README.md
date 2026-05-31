# Автоматическая классификация научного и научно-популярного стилей русскоязычных текстов по компьютерным наукам с применением в задаче преобразования стиля

## Запуск

```
make all

```
make corpus      # анализ корпуса
make classify       # все 4 классификатора: RuBERT, CNN, SVM, Baseline
make transfer    # ruT5-base, все 4 конфигурации (требует RuBERT)
make summary     # сводная таблица
```

## Структура

```
src/
  data.py, features.py, metrics.py, interpret.py, transfer_metrics.py
  models/{rubert,cnn_bilstm,svm,baseline,rut5}.py
scripts/
  analyze_corpus.py
  classify.py
  train_rut5.py
  summarize.py
data/                 data500.csv
Makefile
requirements.txt
```
