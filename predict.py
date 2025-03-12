# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
#
# Live Preview Implementation:
# - Когда enable_live_preview=True, генерация запускается в отдельном потоке
# - Метод predict возвращает генератор stream_output, который отправляет события SSE
# - События SSE содержат промежуточные изображения в формате base64 с префиксом data:image/png;base64,
# - Replicate API получает эти события и отображает их в реальном времени
# - Формат SSE соответствует документации Replicate: https://replicate.com/docs/streaming

import json
import os
import re
import subprocess  # Для запуска внешних процессов
import sys
import time
import threading
from cog import BasePredictor, Input, Path
from time import perf_counter
from contextlib import contextmanager
from typing import Callable, Dict, List, Union, Any, Iterator
from weights import WeightsDownloadCache


@contextmanager
def catchtime(tag: str) -> Callable[[], float]:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f'[Timer: {tag}]: {perf_counter() - start:.3f} seconds')


# Ссылка на чекпоинт перенесена в параметр debug_flux_checkpoint_url
sys.path.extend(["/src"])


def download_base_weights(url: str, dest: Path):
    """
    Загружает базовые веса модели.
    
    Args:
        url: URL для загрузки весов
        dest: Путь для сохранения весов
    """
    start = time.time()  # Засекаем время начала загрузки
    print("downloading url: ", url)
    print("downloading to: ", dest)
    # Используем pget для эффективной загрузки файлов
    # Убираем параметр -xf, так как файл не является архивом
    subprocess.check_call(["pget", "-f", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)  # Выводим время загрузки


class Predictor(BasePredictor):
    weights_cache = WeightsDownloadCache()

    def _download_loras(self, lora_urls: list[str]):
        lora_paths = []

        for url in lora_urls:
            if re.match(r"^https?://replicate.delivery/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+/trained_model.tar", url):
                print(f"Downloading LoRA weights from - Replicate URL: {url}")
                lora_path = self.weights_cache.ensure(
                    url=url,
                    mv_from="output/flux_train_replicate/lora.safetensors",
                )
                print(f"{lora_path=}")
                lora_paths.append(lora_path)
            elif re.match(r"^https?://civitai.com/api/download/models/[0-9]+\?type=Model&format=SafeTensor", url):
                # split url to get first part of the url, everythin before '?type'
                civitai_slug = url.split('?type')[0]
                print(f"Downloading LoRA weights from - Civitai URL: {civitai_slug}")
                lora_path = self.weights_cache.ensure(url, file=True)
                lora_paths.append(lora_path)
            elif url.endswith('.safetensors'):
                print(f"Downloading LoRA weights from - safetensor URL: {url}")
                try:
                    lora_path = self.weights_cache.ensure(url, file=True)
                except Exception as e:
                    print(f"Error downloading LoRA weights: {e}")
                    continue
                print(f"{lora_path=}")
                lora_paths.append(lora_path)

        files = [os.path.join(self.weights_cache.base_dir, f) for f in os.listdir(self.weights_cache.base_dir)]
        print(f'Available loras: {files}')

        return lora_paths

    def setup(self, force_download_url: str = None) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Загружаем модель Flux во время сборки, чтобы ускорить генерацию
        target_dir = "/src/models/Stable-diffusion"
        os.makedirs(target_dir, exist_ok=True)
        model_path = os.path.join(target_dir, "flux_checkpoint.safetensors")

        # Проверяем наличие файла и скачиваем только если его нет или указан force_download_url
        if force_download_url:
            print(f"Загружаем модель Flux с указанного URL... {force_download_url=}")
            download_base_weights(url=force_download_url, dest=model_path)
        elif not os.path.exists(model_path):
            # Используем URL из параметра debug_flux_checkpoint_url, который задан по умолчанию
            from inspect import signature
            default_url = signature(self.predict).parameters['debug_flux_checkpoint_url'].default
            print(f"Модель Flux не найдена, загружаем с URL по умолчанию...")
            download_base_weights(url=default_url, dest=model_path)
        else:
            print(f"Модель Flux уже загружена: {model_path}")

        # workaround for replicate since its entrypoint may contain invalid args
        os.environ["IGNORE_CMD_ARGS_ERRORS"] = "1"


        # Безопасный импорт memory_management
        try:
            from backend import memory_management
            self.has_memory_management = True
        except ImportError as e:
            print(f"Предупреждение: Не удалось импортировать memory_management: {e}")
            self.has_memory_management = False

        # moved env preparation to build time to reduce the warm-up time
        # from modules import launch_utils

        # with launch_utils.startup_timer.subcategory("prepare environment"):
        #     launch_utils.prepare_environment()

        from modules import initialize_util
        from modules import initialize
        from modules import timer

        startup_timer = timer.startup_timer
        startup_timer.record("launcher")

        initialize.imports()

        initialize.check_versions()

        initialize.initialize()

        # Импортируем shared после initialize.initialize()
        from modules import shared

        # Устанавливаем forge_preset на 'flux'
        shared.opts.set('forge_preset', 'flux')

        # Устанавливаем чекпоинт
        shared.opts.set('sd_model_checkpoint', 'flux_checkpoint.safetensors')

        # Устанавливаем unet тип на 'Automatic (fp16 LoRA)' для Flux, чтобы LoRA работали правильно
        shared.opts.set('forge_unet_storage_dtype', 'Automatic (fp16 LoRA)')
        
        # Включаем и настраиваем live preview
        shared.opts.set('live_previews_enable', True)
        shared.opts.set('show_progress_every_n_steps', 1)  # Обновлять каждый шаг
        shared.opts.set('live_preview_content', 'Prompt')  # Показывать содержимое промпта
        shared.opts.set('live_preview_refresh_period', 250)  # Период обновления в мс
        shared.opts.set('live_preview_fast_interrupt', True)  # Быстрое прерывание
        shared.opts.set('show_progress_grid', False)  # Отключаем сетку прогресса для экономии ресурсов

        # Оптимизация памяти для лучшего качества и скорости с Flux
        if self.has_memory_management:
            # Выделяем больше памяти для загрузки весов модели (90% для весов, 10% для вычислений)
            total_vram = memory_management.total_vram
            inference_memory = int(total_vram * 0.1)  # 10% для вычислений
            model_memory = total_vram - inference_memory

            memory_management.current_inference_memory = inference_memory * 1024 * 1024  # Конвертация в байты
            print(
                f"[GPU Setting] Выделено {model_memory} MB для весов модели и {inference_memory} MB для вычислений"
            )

            # Настройка Swap Method на ASYNC для лучшей производительности
            try:
                from backend import stream
                # Для Flux рекомендуется ASYNC метод, который может быть до 30% быстрее
                stream.stream_activated = True  # True = ASYNC, False = Queue
                print("[GPU Setting] Установлен ASYNC метод загрузки для лучшей производительности")

                # Настройка Swap Location на Shared для лучшей производительности
                memory_management.PIN_SHARED_MEMORY = True  # True = Shared, False = CPU
                print("[GPU Setting] Установлен Shared метод хранения для лучшей производительности")
            except ImportError as e:
                print(f"Предупреждение: Не удалось импортировать stream: {e}")
        else:
            print("[GPU Setting] memory_management не доступен, используются настройки по умолчанию")

        from fastapi import FastAPI

        app = FastAPI()
        initialize_util.setup_middleware(app)

        from modules.api.api import Api
        from modules.call_queue import queue_lock

        # Create a custom API class that patches the script handling functions
        class CustomApi(Api):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Patch the get_script function to handle LoRA scripts
                original_get_script = self.get_script

                def patched_get_script(script_name, script_runner):
                    try:
                        return original_get_script(script_name, script_runner)
                    except Exception as e:
                        # If the script is not found and it's the LoRA script, handle it specially
                        if script_name in ["lora", "sd_forge_lora"]:
                            print(
                                f"LoRA script '{script_name}' not found in standard scripts, using extra_network_data instead"
                            )
                            return None
                        raise e

                self.get_script = patched_get_script

                # Patch the init_script_args function to handle missing scripts
                original_init_script_args = self.init_script_args

                def patched_init_script_args(
                    request, default_script_args, selectable_scripts, selectable_idx, script_runner, *,
                    input_script_args=None
                ):
                    try:
                        return original_init_script_args(
                            request, default_script_args, selectable_scripts, selectable_idx, script_runner,
                            input_script_args=input_script_args
                        )
                    except Exception as e:
                        # If there's an error with alwayson_scripts, try to continue without them
                        if hasattr(request, 'alwayson_scripts') and request.alwayson_scripts:
                            print(f"Error initializing alwayson_scripts: {e}")
                            # Remove problematic scripts
                            for script_name in list(request.alwayson_scripts.keys()):
                                if script_name in ["lora", "sd_forge_lora"]:
                                    print(f"Removing problematic script: {script_name}")
                                    del request.alwayson_scripts[script_name]

                            # Try again without the problematic scripts
                            if not request.alwayson_scripts:
                                request.alwayson_scripts = None

                            return original_init_script_args(
                                request, default_script_args, selectable_scripts, selectable_idx,
                                script_runner, input_script_args=input_script_args
                            )
                        raise e

                self.init_script_args = patched_init_script_args

        self.api = CustomApi(app, queue_lock)
        
    def get_progress_and_live_preview(self, task_id, last_preview_id=-1):
        """Получить текущий прогресс и live preview для задачи"""
        from modules.progress import progressapi
        from modules.api.models import ProgressRequest
        
        try:
            # Создаем запрос для получения прогресса
            req = ProgressRequest(
                id_task=task_id,
                id_live_preview=last_preview_id,
                live_preview=True
            )
            
            # Получаем прогресс и live preview
            progress_data = progressapi(req)
            
            # Если есть новое изображение, возвращаем его
            if hasattr(progress_data, 'live_preview') and progress_data.live_preview:
                # Изображение уже в формате base64
                return progress_data, progress_data.live_preview
            
            return progress_data, None
        except Exception as e:
            print(f"Ошибка при получении прогресса: {e}")
            # Создаем заглушку для progress_data
            from types import SimpleNamespace
            dummy_progress = SimpleNamespace(
                completed=False,
                progress=0,
                eta_relative=0,
                id_live_preview=-1,
                textinfo="Ожидание..."
            )
            return dummy_progress, None
    
    def stream_output(self, task_id) -> Iterator[str]:
        """Генератор для потоковой передачи промежуточных результатов в формате SSE для Replicate"""
        from modules import shared
        
        last_preview_id = -1
        last_progress = 0
        max_wait_time = 60  # Максимальное время ожидания в секундах
        start_time = time.time()
        event_id = int(time.time())
        
        while True:
            try:
                # Проверяем, завершилась ли задача
                if hasattr(shared, 'task_cache') and task_id in shared.task_cache:
                    result = shared.task_cache[task_id]
                    
                    # Если результат содержит ошибку, отправляем ее
                    if isinstance(result, dict) and "error" in result:
                        error_json = json.dumps({"detail": result["error"]})
                        yield f"event: error\nid: {event_id}:0\ndata: {error_json}\n\n"
                        yield f"event: done\ndata: {{\"reason\": \"error\"}}\n\n"
                    else:
                        # Отправляем финальные результаты
                        for path in result:
                            # Отправляем финальное изображение как output
                            with open(path, "rb") as img_file:
                                import base64
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                # Добавляем префикс для правильного отображения в браузере
                                img_data = f"data:image/png;base64,{img_data}"
                                yield f"event: output\nid: {event_id}:0\ndata: {img_data}\n\n"
                                event_id += 1
                        
                        # Отправляем событие done
                        yield f"event: done\ndata: {{}}\n\n"
                    
                    # Удаляем задачу из кэша
                    del shared.task_cache[task_id]
                    break
                
                # Получаем прогресс и live preview
                progress_data, base64_image = self.get_progress_and_live_preview(task_id, last_preview_id)
                
                # Если задача завершена, но результат еще не в кэше, ждем немного
                if progress_data.completed:
                    time.sleep(0.5)
                    continue
                
                # Если есть новое изображение, отправляем его
                if base64_image:
                    last_preview_id = progress_data.id_live_preview
                    
                    # Проверяем, содержит ли base64_image префикс data:image
                    if not base64_image.startswith("data:image"):
                        # Добавляем префикс для правильного отображения в браузере
                        base64_image = f"data:image/png;base64,{base64_image}"
                    
                    # Отправляем событие output с изображением в формате base64
                    yield f"event: output\nid: {event_id}:0\ndata: {base64_image}\n\n"
                    event_id += 1
                
                # Если прогресс изменился, отправляем его как комментарий в SSE
                current_progress = progress_data.progress
                if current_progress != last_progress:
                    last_progress = current_progress
                    progress_percent = int(current_progress * 100)
                    yield f": Progress: {progress_percent}%\n\n"
                
                # Проверяем таймаут
                if time.time() - start_time > max_wait_time:
                    error_json = json.dumps({"detail": "Timeout waiting for generation to complete"})
                    yield f"event: error\nid: {event_id}:0\ndata: {error_json}\n\n"
                    yield f"event: done\ndata: {{\"reason\": \"error\"}}\n\n"
                    break
                
                time.sleep(0.1)  # Небольшая задержка
            except Exception as e:
                # В случае ошибки отправляем событие error
                error_json = json.dumps({"detail": str(e)})
                yield f"event: error\nid: {event_id}:0\ndata: {error_json}\n\n"
                yield f"event: done\ndata: {{\"reason\": \"error\"}}\n\n"
                break

    def predict(
        self,
        prompt: str = Input(description="Prompt"
        ),
        width: int = Input(
            description="Width of output image", ge=1, le=1280, default=768
        ),
        height: int = Input(
            description="Height of output image", ge=1, le=1280, default=1280
        ),
        num_outputs: int = Input(
            description="Number of images to output", ge=1, le=4, default=1
        ),
        sampler: str = Input(
            description="Sampling method для Flux моделей",
            choices=[
                "[Forge] Flux Realistic",
                "Euler",
                "DEIS",
                "Euler a",
                "DPM++ 2M",
                "DPM++ SDE",
                "DPM++ 2M SDE",
                "DPM++ 2M SDE Karras",
                "DPM++ 2M SDE Exponential",
                "DPM++ 3M SDE",
                "DPM++ 3M SDE Karras",
                "DPM++ 3M SDE Exponential"
            ],
            default="[Forge] Flux Realistic",
        ),
        scheduler: str = Input(
            description="Schedule type для Flux моделей",
            choices=[
                "Simple",
                "Karras",
                "Exponential",
                "SGM Uniform",
                "SGM Karras",
                "SGM Exponential",
                "Align Your Steps",
                "Align Your Steps 11",
                "Align Your Steps 32",
                "Align Your Steps GITS",
                "KL Optimal",
                "Normal",
                "DDIM",
                "Beta",
                "Turbo"
            ],
            default="Simple",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=50, default=28
        ),
        guidance_scale: float = Input(
            description="CFG Scale (для Flux рекомендуется значение 1.0)", ge=0, le=50, default=1.0
        ),
        distilled_guidance_scale: float = Input(
            description="Distilled CFG Scale (основной параметр для Flux, рекомендуется 3.5)", ge=0, le=30,
            default=3.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=-1
        ),
        # image: Path = Input(description="Grayscale input image"),
        enable_hr: bool = Input(
            description="Hires. fix",
            default=False,
        ),
        hr_upscaler: str = Input(
            description="Upscaler for Hires. fix",
            choices=[
                "Latent",
                "Latent (antialiased)",
                "Latent (bicubic)",
                "Latent (bicubic antialiased)",
                "Latent (nearest)",
                "Latent (nearest-exact)",
                "None",
                "Lanczos",
                "Nearest",
                "ESRGAN_4x",
                "LDSR",
                "R-ESRGAN 4x+",
                "R-ESRGAN 4x+ Anime6B",
                "ScuNET GAN",
                "ScuNET PSNR",
                "SwinIR 4x",
            ],
            default="Latent",
        ),
        hr_steps: int = Input(
            description="Inference steps for Hires. fix", ge=0, le=100, default=10
        ),
        hr_scale: float = Input(
            description="Factor to scale image by", ge=1, le=4, default=1.5
        ),
        denoising_strength: float = Input(
            description="Denoising strength. 1.0 corresponds to full destruction of information in init image",
            ge=0,
            le=1,
            default=0.1,
        ),
        lora_urls: list[str] = Input(
            description="Ссылки на LoRA файлы",
            default=[],
        ),
        lora_scales: list[float] = Input(
            description="Lora scales",
            default=[1],
        ),
        debug_flux_checkpoint_url: str = Input(
            description="Flux checkpoint URL",
            default=""
        ),
        enable_clip_l: bool = Input(
            description="Enable encoder",
            default=False
        ),
        enable_t5xxl_fp16: bool = Input(
            description="t5xxl_fp16",
            default=False
        ),
        enable_ae: bool = Input(
            description="Enable ae",
            default=False
        ),
        enable_live_preview: bool = Input(
            description="Включить потоковую передачу промежуточных результатов",
            default=True
        ),
    ) -> Union[list[Path], Iterator[str]]:
        print("Cache version 107")
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
            "sampler_name": sampler,  # Используем выбранный пользователем sampler
            "scheduler": scheduler,  # Устанавливаем scheduler для Flux
            "enable_hr": enable_hr,
            "hr_upscaler": hr_upscaler,
            "hr_second_pass_steps": hr_steps,
            "denoising_strength": denoising_strength if enable_hr else None,
            "hr_scale": hr_scale,
            "distilled_cfg_scale": distilled_guidance_scale,
            "hr_additional_modules": [],
        }

        alwayson_scripts = {}

        # Добавляем все скрипты в payload, если они есть
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

        # Если включен live preview, запускаем генерацию в отдельном потоке и возвращаем генератор
        if enable_live_preview:
            from modules import shared
            
            # Создаем уникальный ID для задачи
            task_id = str(uuid.uuid4())
            
            # Запускаем генерацию в отдельном потоке
            def run_generation():
                try:
                    # Запускаем генерацию
                    resp = self.api.text2imgapi(**req)
                    
                    # Сохраняем результат в shared.task_cache для последующего получения
                    info = json.loads(resp.info)
                    outputs = []
                    
                    for i, image in enumerate(resp.images):
                        seed = info["all_seeds"][i]
                        gen_bytes = BytesIO(base64.b64decode(image))
                        gen_data = Image.open(gen_bytes)
                        filename = "{}-{}.png".format(seed, uuid.uuid1())
                        gen_data.save(fp=filename, format="PNG")
                        output = Path(filename)
                        outputs.append(output)
                    
                    # Сохраняем результаты в кэше задач
                    if not hasattr(shared, 'task_cache'):
                        shared.task_cache = {}
                    shared.task_cache[task_id] = outputs
                    
                except Exception as e:
                    print(f"Ошибка при генерации: {e}")
                    if not hasattr(shared, 'task_cache'):
                        shared.task_cache = {}
                    shared.task_cache[task_id] = {"error": str(e)}
            
            # Запускаем поток
            thread = threading.Thread(target=run_generation)
            thread.daemon = True
            thread.start()
            
            # Возвращаем генератор для потоковой передачи
            return self.stream_output(task_id)
        
        # Если live preview отключен, используем обычный режим
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
