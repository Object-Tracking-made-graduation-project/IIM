Для тестирования нужно положить в корень модель https://cloud.mail.ru/public/tDFV/nTQk76xrY/FDST-HR-ep_177_F1_0.969_Pre_0.984_Rec_0.955_mae_1.0_mse_1.5.pth

Подготовить виртуальное окружение
```
call prepare_venv.bat
```
Выполнить 
```
call venv\Scripts\activate.bat
(venv) python one_pic_inference.py
```

Скрипт должен отработать без ошибок.
`requirements.txt` содержит общие модули для IIM и Bytetrack.