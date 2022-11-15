Для тестирования нужно положить в корень модель https://cloud.mail.ru/public/tDFV/nTQk76xrY/FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth
предварительно переименовав ее в `FDST-HR.pth` (это назвоние дефолтное в настройках)

Собрать образ 
```
docker image build -t iim .
```
Запустить контейнер
```
docker run -d -p 5000:5000 iim
```

Сервис должен стать доступен на http://localhost:5000/

В случае проблем можно стянуть обрать с докерхаба:
```
docker pull salos/iim
```
