# Live Preview в WebUI Forge

## Обзор механизма Live Preview

WebUI Forge имеет встроенный механизм для отображения промежуточных результатов генерации изображений в реальном времени. Этот механизм называется "Live Preview" и позволяет пользователю видеть, как изображение формируется на каждом шаге диффузии.

### Ключевые компоненты системы Live Preview

1. **Хранение состояния** (в `modules/shared_state.py`):
   - `current_latent` - текущий латентный вектор во время генерации
   - `current_image` - текущее преобразованное изображение для предпросмотра
   - `id_live_preview` - счетчик, увеличивающийся при каждом обновлении изображения

2. **Обновление изображения** (в `modules/shared_state.py`):
   - `set_current_image()` - проверяет, нужно ли обновить изображение
   - `do_set_current_image()` - преобразует латентный вектор в изображение
   - `assign_current_image(image)` - сохраняет изображение и увеличивает счетчик

3. **Управление задачами** (в `modules/progress.py`):
   - `create_task_id(task_type)` - создает уникальный ID для задачи
   - `start_task(id_task)` - запускает задачу
   - `finish_task(id_task)` - завершает задачу
   - `add_task_to_queue(id_job)` - добавляет задачу в очередь

4. **API для получения прогресса** (в `modules/progress.py`):
   - Эндпоинт `/internal/progress` принимает ID задачи и ID последнего полученного изображения
   - Возвращает текущий прогресс, ETA и live preview в формате base64

## Как работает система прогресса и live preview

1. Во время генерации на каждом шаге вызывается `set_current_image()`, который проверяет, нужно ли обновить изображение
2. Если прошло достаточно шагов (`sampling_step - current_image_sampling_step >= show_progress_every_n_steps`), вызывается `do_set_current_image()`
3. `do_set_current_image()` преобразует текущий латентный вектор в изображение и вызывает `assign_current_image()`
4. `assign_current_image()` устанавливает текущее изображение и увеличивает `id_live_preview`
5. API эндпоинт `/internal/progress` проверяет, изменился ли `id_live_preview`, и если да, возвращает новое изображение

## Рекомендации по интеграции в predict.py

Для интеграции live preview в `predict.py` необходимо выполнить следующие шаги:

### 1. Включить настройки Live Preview

В методе `setup` после инициализации WebUI Forge добавьте следующий код:

```python
# Включаем live preview
from modules import shared
shared.opts.set('live_previews_enable', True)  # Включить live preview
shared.opts.set('show_progress_every_n_steps', 1)  # Обновлять каждый шаг
shared.opts.set('live_previews_image_format', 'jpeg')  # Формат изображения
```

### 2. Добавить функцию для получения прогресса и live preview

Добавьте следующую функцию в класс `Predictor`:

```python
def get_progress_and_live_preview(self, task_id, last_preview_id=-1):
    """Получить текущий прогресс и live preview для задачи"""
    from modules.progress import progressapi
    from modules.api.models import ProgressRequest
    
    # Создаем запрос для получения прогресса
    req = ProgressRequest(
        id_task=task_id,
        id_live_preview=last_preview_id,
        live_preview=True
    )
    
    # Получаем прогресс и live preview
    progress_data = progressapi(req)
    
    # Если есть новое изображение, возвращаем его
    if progress_data.live_preview:
        # Извлекаем base64 из data URI
        base64_data = progress_data.live_preview.split(",")[1]
        return progress_data, base64_data
    
    return progress_data, None
```

### 3. Модифицировать метод predict для получения промежуточных результатов

Есть два подхода к модификации метода `predict`:

#### Подход 1: Изменить возвращаемый тип на dict

```python
def predict(
    self,
    # Существующие параметры...
) -> dict:  # Изменяем возвращаемый тип с list[Path] на dict
    # Существующий код...
    
    # Импортируем необходимые модули
    from modules.progress import create_task_id, start_task, finish_task
    import threading
    
    # Создаем ID задачи
    task_id = create_task_id("txt2img")
    
    # Запускаем генерацию в отдельном потоке
    def generate_images():
        nonlocal resp
        start_task(task_id)
        try:
            resp = self.api.text2imgapi(**req)
        finally:
            finish_task(task_id)
    
    resp = None
    generation_thread = threading.Thread(target=generate_images)
    generation_thread.start()
    
    # Получаем промежуточные результаты
    last_preview_id = -1
    intermediate_results = []
    
    while generation_thread.is_alive():
        progress_data, base64_image = self.get_progress_and_live_preview(task_id, last_preview_id)
        
        if base64_image:
            # Сохраняем промежуточный результат
            intermediate_results.append(base64_image)
            last_preview_id = progress_data.id_live_preview
        
        time.sleep(0.1)  # Небольшая задержка
    
    # Дожидаемся завершения генерации
    generation_thread.join()
    
    # Обрабатываем результаты как обычно
    info = json.loads(resp.info)
    outputs = []
    
    with catchtime(tag="Total Encode Time"):
        for i, image in enumerate(resp.images):
            # Существующий код...
    
    # Возвращаем как финальные, так и промежуточные результаты
    return {
        "final_images": outputs,
        "intermediate_images": intermediate_results
    }
```

#### Подход 2: Добавить параметр для включения/выключения live preview

```python
def predict(
    self,
    # Существующие параметры...
    enable_live_preview: bool = Input(
        description="Включить live preview во время генерации",
        default=False
    ),
) -> list[Path] | dict:  # Возвращаемый тип зависит от enable_live_preview
    # Существующий код...
    
    if not enable_live_preview:
        # Стандартная генерация без live preview
        with catchtime(tag="Total Prediction Time"):
            resp = self.api.text2imgapi(**req)
        
        # Обрабатываем результаты как обычно
        info = json.loads(resp.info)
        outputs = []
        
        with catchtime(tag="Total Encode Time"):
            for i, image in enumerate(resp.images):
                # Существующий код...
        
        return outputs
    else:
        # Генерация с live preview
        # Код из Подхода 1...
        
        return {
            "final_images": outputs,
            "intermediate_images": intermediate_results
        }
```

### 4. Полная реализация с параметром enable_live_preview

Вот полная реализация метода `predict` с параметром `enable_live_preview`:

```python
def predict(
    self,
    prompt: str = Input(description="Prompt"),
    # Другие существующие параметры...
    enable_live_preview: bool = Input(
        description="Включить live preview во время генерации",
        default=False
    ),
) -> list[Path] | dict:
    print("Cache version 105")
    """Run a single prediction on the model"""
    from modules.extra_networks import ExtraNetworkParams
    from modules import scripts
    from modules.api.models import (
        StableDiffusionTxt2ImgProcessingAPI,
    )
    from PIL import Image
    import uuid
    import base64
    from io import BytesIO
    
    if debug_flux_checkpoint_url:
        self.setup(force_download_url=debug_flux_checkpoint_url)
    
    lora_paths = self._download_loras(lora_urls)
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "batch_size": num_outputs,
        "steps": num_inference_steps,
        "cfg_scale": guidance_scale,
        "seed": seed,
        "do_not_save_samples": True,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "enable_hr": enable_hr,
        "hr_upscaler": hr_upscaler,
        "hr_second_pass_steps": hr_steps,
        "denoising_strength": denoising_strength if enable_hr else None,
        "hr_scale": hr_scale,
        "distilled_cfg_scale": distilled_guidance_scale,
        "hr_additional_modules": [],
    }
    
    alwayson_scripts = {}
    
    if alwayson_scripts:
        payload["alwayson_scripts"] = alwayson_scripts
    
    print(f"Финальный пейлоад: {payload=}")
    print("Available scripts:", [script.title().lower() for script in scripts.scripts_txt2img.scripts])
    
    req = dict(
        txt2imgreq=StableDiffusionTxt2ImgProcessingAPI(**payload),
        extra_network_data={
            "lora": [
                ExtraNetworkParams(
                    items=[
                        lora_path.split('/')[-1].split('.safetensors')[0],
                        str(lora_scale)
                    ]
                )
                for lora_path, lora_scale in zip(lora_paths, lora_scales)
            ]
        },
        additional_modules={
            "clip_l.safetensors": enable_clip_l,
            "t5xxl_fp16.safetensors": enable_t5xxl_fp16,
            "ae.safetensors": enable_ae,
        },
    )
    
    for lora in req['extra_network_data']['lora']:
        print(f"LoRA: {lora.items=}")
    
    if not enable_live_preview:
        # Стандартная генерация без live preview
        with catchtime(tag="Total Prediction Time"):
            resp = self.api.text2imgapi(**req)
        
        info = json.loads(resp.info)
        outputs = []
        
        with catchtime(tag="Total Encode Time"):
            for i, image in enumerate(resp.images):
                seed = info["all_seeds"][i]
                gen_bytes = BytesIO(base64.b64decode(image))
                gen_data = Image.open(gen_bytes)
                filename = "{}-{}.png".format(seed, uuid.uuid1())
                gen_data.save(fp=filename, format="PNG")
                output = Path(filename)
                outputs.append(output)
        
        return outputs
    else:
        # Генерация с live preview
        from modules.progress import create_task_id, start_task, finish_task
        import threading
        
        # Создаем ID задачи
        task_id = create_task_id("txt2img")
        
        # Запускаем генерацию в отдельном потоке
        def generate_images():
            nonlocal resp
            start_task(task_id)
            try:
                resp = self.api.text2imgapi(**req)
            finally:
                finish_task(task_id)
        
        resp = None
        generation_thread = threading.Thread(target=generate_images)
        generation_thread.start()
        
        # Получаем промежуточные результаты
        last_preview_id = -1
        intermediate_results = []
        
        while generation_thread.is_alive():
            progress_data, base64_image = self.get_progress_and_live_preview(task_id, last_preview_id)
            
            if base64_image:
                # Сохраняем промежуточный результат
                intermediate_results.append(base64_image)
                last_preview_id = progress_data.id_live_preview
            
            time.sleep(0.1)  # Небольшая задержка
        
        # Дожидаемся завершения генерации
        generation_thread.join()
        
        # Обрабатываем результаты как обычно
        info = json.loads(resp.info)
        outputs = []
        
        with catchtime(tag="Total Encode Time"):
            for i, image in enumerate(resp.images):
                seed = info["all_seeds"][i]
                gen_bytes = BytesIO(base64.b64decode(image))
                gen_data = Image.open(gen_bytes)
                filename = "{}-{}.png".format(seed, uuid.uuid1())
                gen_data.save(fp=filename, format="PNG")
                output = Path(filename)
                outputs.append(output)
        
        return {
            "final_images": outputs,
            "intermediate_images": intermediate_results
        }
```

## Дополнительные рекомендации

1. **Оптимизация частоты обновления**:
   - Если генерация происходит слишком быстро, можно увеличить значение `show_progress_every_n_steps` для уменьшения количества промежуточных изображений
   - Например: `shared.opts.set('show_progress_every_n_steps', 2)` - обновлять каждые 2 шага

2. **Формат изображения**:
   - Для более быстрой передачи используйте формат JPEG: `shared.opts.set('live_previews_image_format', 'jpeg')`
   - Для лучшего качества используйте PNG: `shared.opts.set('live_previews_image_format', 'png')`

3. **Обработка ошибок**:
   - Добавьте обработку исключений при получении прогресса и live preview
   - Если задача завершилась с ошибкой, убедитесь, что вы вызываете `finish_task(task_id)`

4. **Отображение прогресса**:
   - Вы можете использовать поле `progress_data.progress` для отображения процента выполнения
   - Вы можете использовать поле `progress_data.eta` для отображения оставшегося времени

## Заключение

Интеграция live preview в `predict.py` позволит получать промежуточные результаты генерации в реальном времени, что улучшит пользовательский опыт, особенно для длительных генераций с большим количеством шагов.

Рекомендуется использовать подход с параметром `enable_live_preview`, так как он позволяет сохранить обратную совместимость с существующим API и дает пользователю возможность выбора.