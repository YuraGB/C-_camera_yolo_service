# Camera CV Service

Сервіс для захоплення та обробки відеопотоків з камер (USB, вбудовані) або відеофайлів з використанням OpenCV та ONNX Runtime.

    IMPORTANT!!!
    This service was created by chatGPT (vide coding as you want).
    Needed  Code review by real C++ developers

    ------

    VERY IMPORTANT!!!!
    CMakeLists.txt  uses dependensies for WINDOWS WITH MY CURRENT configuration (paths fpr deps)
    change configuration for your system
---

## Основні можливості

- Підключення декількох джерел відео одночасно:
  - Вбудовані камери
  - USB камери
  - Відеофайли (`.mp4`, `.avi` тощо)
- Захоплення та обробка кадрів у реальному часі
- Інференс моделей нейромереж через ONNX Runtime (`yolov8x.onnx` за замовчуванням)
- Підготовка результатів для gRPC-сервера (розширювано)
- Потокова обробка з чергами кадрів для уникнення затримок

---

## Стек технологій

- **C++17 / C++20**
- **OpenCV 4.12** — обробка відео та камер
- **ONNX Runtime** — inference нейромереж
- **gRPC** — відправка результатів детекцій (розширювано)
- **Protobuf** — опис структур даних для gRPC
- **CMake** — система збірки
- **vcpkg** — менеджер пакетів для Windows
- **Multithreading** (`std::thread`, `std::mutex`, `std::condition_variable`) для паралельної обробки кадрів

---

## Підготовка проекту

1. Клонувати репозиторій:

```bash
git clone <your-repo-url>
cd camera_cv_service
# Встановити залежності через vcpkg:
cd E:/tools/vcpkg
.\vcpkg install opencv4[core,videoio]:x64-windows
.\vcpkg install onnxruntime:x64-windows
.\vcpkg install protobuf:x64-windows
# Згенерувати build-файли через CMake:
cd E:/Progects/test/camera_cv_service
cmake .. -B build -DCMAKE_TOOLCHAIN_FILE=E:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake
# Зібрати проект:
cmake --build build --config Release
```
Бінарник з’явиться у:

    build/build/bin/Release/camera_cv_service.exe

Запуск сервісу
    
    camera_cv_service.exe

Автоматично підключає доступні камери (0..10).
Для додавання відеофайлів у коді:


    camera_manager.addCamera("video1", "example_video.mp4");

Отримання та обробка кадрів:


    auto frame = camera_manager.getLatestFrame("camera_0");
    
    if (frame) {
      inference_engine.processFrame(frame);
      auto result = inference_engine.getResult();
    if (result) {
        std::cout << "Frame " << result->frame_id
                  << " from camera_0"
                  << " processed, " << result->detections.size()
                  << " detections" << std::endl;
    }
}

Зупинка сервісу через Ctrl+C або виклик методів:
    
    camera_manager.stopAllCameras();
    inference_engine.stop();
    grpc_server.stop();

Структура проекту
```
camera_cv_service/
├─ include/
│  ├─ camera_manager.h
│  ├─ inference_engine.h
│  └─ models/
│     └─ frame.h
├─ src/
│  ├─ main.cpp
│  ├─ camera_manager.cpp
│  ├─ inference_engine.cpp
│  └─ grpc_server.cpp
├─ CMakeLists.txt
└─ build/
```

### Особливості сервісу
Кадри з камер обробляються у потоках з власними чергами (std::deque) для уникнення блокувань.
Підтримка обробки відеофайлів або камер одночасно.
Інференс виконується у InferenceEngine та повертає детекції для подальшої обробки або gRPC.
Можна масштабувати на кілька камер без блокування головного потоку.
Приклад додавання камер та відео


    // Додати USB або вбудовану камеру
    camera_manager.addCamera("camera_0", "0");
    camera_manager.addCamera("camera_1", "1");


    // Додати відеофайл
    camera_manager.addCamera("video_file", "example_video.mp4");

    // Запуск усіх камер та обробка кадрів
    camera_manager.startAllCameras();
    inference_engine.start();
  
### TODO / Плани
Тест на декількох камерах
Тести ```yolo``` моделей 

    Attention

-  I couldn't dawnload 
```.\vcpkg install onnxruntime:x64-windows```
  So, I dawnload it manualy and I added .dll file into 
```/buld/bin/Release/``` folder. Where main  <...>.exe file is.
    

Контакти
  
    Автор: Yurii H.
    GitHub / Repo URL: <your-repo-url>