# 🚦 SmartTraffic: CV Traffic Analyzer

Мониторинг трафика перекрёстков в реальном времени. YOLOv8 + DeepSORT, аналитика и API.

## 📦 Стек
- YOLOv8
- DeepSORT / Norfair
- FastAPI
- Docker + CI/CD

## 🚀 Быстрый старт
```bash
docker-compose up --build
```

## 📂 Структура

- src/ - основная логика CV и трекинга
api/ - FastAPI сервис
tests/ - тесты
docker/ - контейнеры