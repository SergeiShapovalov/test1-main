"""
Патч для включения live preview в процессе генерации изображений.
Заменяет функцию process_images в modules/processing.py для вызова колбэков на каждом шаге генерации.
"""

import sys
import traceback
import base64
from io import BytesIO
from typing import List, Dict, Any, Callable, Optional, Union
from PIL import Image


def apply_processing_patch():
    """
    Применяет патч к функции process_images для включения live preview.
    """
    try:
        print("Применяем патч для включения live preview...")
        
        # Импортируем необходимые модули
        from modules import processing, shared, devices
        from modules.processing import StableDiffusionProcessing, process_images
        import torch
        
        # Сохраняем оригинальную функцию
        original_process_images = process_images
        
        # Определяем новую функцию с поддержкой live preview
        def patched_process_images(p: StableDiffusionProcessing, *args, **kwargs):
            """
            Патч для функции process_images, который вызывает колбэки на каждом шаге генерации.
            """
            try:
                print("Запуск patched_process_images с поддержкой live preview")
                
                # Проверяем наличие колбэков
                if not hasattr(shared, 'progress_callbacks') or not shared.progress_callbacks:
                    print("Предупреждение: shared.progress_callbacks не существует или пуст")
                    shared.progress_callbacks = []
                
                # Сохраняем оригинальный callback
                original_callback = p.callback
                
                # Создаем новый callback, который будет вызывать все зарегистрированные колбэки
                def callback_with_preview(step, total_steps):
                    try:
                        # Вызываем оригинальный callback, если он есть
                        if original_callback is not None:
                            original_callback(step, total_steps)
                        
                        # Получаем промежуточное изображение
                        preview_image = get_preview_image(p)
                        
                        # Вызываем все зарегистрированные колбэки
                        for callback in shared.progress_callbacks:
                            try:
                                callback(step, total_steps, preview_image)
                            except Exception as e:
                                print(f"Ошибка при вызове колбэка: {e}")
                                traceback.print_exc()
                    except Exception as e:
                        print(f"Ошибка в callback_with_preview: {e}")
                        traceback.print_exc()
                
                # Заменяем оригинальный callback на наш
                p.callback = callback_with_preview
                
                # Вызываем оригинальную функцию
                print("Вызываем оригинальную функцию process_images")
                result = original_process_images(p, *args, **kwargs)
                
                # Восстанавливаем оригинальный callback
                p.callback = original_callback
                
                return result
            except Exception as e:
                print(f"Критическая ошибка в patched_process_images: {e}")
                traceback.print_exc()
                # В случае ошибки возвращаем результат оригинальной функции
                return original_process_images(p, *args, **kwargs)
        
        # Заменяем оригинальную функцию на нашу
        processing.process_images = patched_process_images
        
        print("Патч для live preview успешно применен")
        return True
    except Exception as e:
        print(f"Ошибка при применении патча для live preview: {e}")
        traceback.print_exc()
        return False


def get_preview_image(p) -> Optional[str]:
    """
    Получает промежуточное изображение из текущего процесса генерации и возвращает его в формате base64.
    
    Args:
        p: Объект StableDiffusionProcessing
        
    Returns:
        Строка base64 с изображением или None, если изображение недоступно
    """
    try:
        # Импортируем необходимые модули
        from modules import shared, devices
        import torch
        import numpy as np
        
        # Проверяем наличие state
        if not hasattr(shared, 'state'):
            print("shared.state не существует")
            return None
        
        # Проверяем наличие промежуточного изображения
        if not hasattr(shared.state, 'current_latent'):
            print("shared.state.current_latent не существует")
            return None
        
        # Получаем текущий latent
        current_latent = shared.state.current_latent
        if current_latent is None:
            print("current_latent is None")
            return None
        
        # Декодируем latent в изображение
        try:
            # Получаем модель
            if hasattr(p, 'sd_model'):
                model = p.sd_model
            elif hasattr(shared, 'sd_model'):
                model = shared.sd_model
            else:
                print("Модель SD не найдена")
                return None
            
            # Переносим latent на устройство модели
            device = devices.get_device_for("unet")
            if current_latent.device != device:
                current_latent = current_latent.to(device)
            
            # Декодируем latent в изображение с помощью VAE
            with torch.no_grad():
                # Используем VAE для декодирования
                decoded = model.decode_first_stage(current_latent)
                
                # Преобразуем тензор в изображение
                decoded = decoded.cpu().numpy()
                
                # Нормализуем значения
                decoded = np.clip((decoded + 1.0) / 2.0, 0.0, 1.0)
                
                # Преобразуем в формат [batch, height, width, channel]
                if decoded.shape[1] == 3:  # [batch, channel, height, width]
                    decoded = np.transpose(decoded, (0, 2, 3, 1))
                
                # Берем первое изображение из батча
                if len(decoded.shape) == 4:
                    decoded = decoded[0]
                
                # Преобразуем в uint8
                decoded = (decoded * 255).astype(np.uint8)
                
                # Создаем изображение PIL
                image = Image.fromarray(decoded)
                
                # Преобразуем в base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Добавляем префикс data:image/png;base64,
                base64_image = f"data:image/png;base64,{base64_image}"
                
                return base64_image
        except Exception as e:
            print(f"Ошибка при декодировании latent: {e}")
            traceback.print_exc()
            
            # Попробуем альтернативный метод декодирования
            try:
                print("Пробуем альтернативный метод декодирования...")
                
                # Импортируем необходимые модули для декодирования
                from modules import sd_samplers
                
                # Декодируем latent в изображение
                decoded = sd_samplers.decode_first_stage(model, current_latent)
                if decoded is None:
                    print("Не удалось декодировать latent")
                    return None
                
                # Преобразуем тензор в изображение
                x_sample = 255.0 * decoded.cpu().numpy()
                x_sample = x_sample.clip(0, 255).astype(np.uint8)
                
                # Преобразуем формат из [batch, height, width, channel] в [height, width, channel]
                if len(x_sample.shape) == 4:
                    x_sample = x_sample[0]
                
                # Преобразуем из [channel, height, width] в [height, width, channel], если необходимо
                if x_sample.shape[0] == 3 and len(x_sample.shape) == 3:
                    x_sample = np.transpose(x_sample, (1, 2, 0))
                
                # Создаем изображение
                image = Image.fromarray(x_sample)
                
                # Преобразуем в base64
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Добавляем префикс data:image/png;base64,
                base64_image = f"data:image/png;base64,{base64_image}"
                
                return base64_image
            except Exception as e:
                print(f"Ошибка при альтернативном декодировании latent: {e}")
                traceback.print_exc()
                return None
    except Exception as e:
        print(f"Ошибка при получении промежуточного изображения: {e}")
        traceback.print_exc()
        return None