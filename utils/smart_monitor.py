
import os
import time
import logging
import psutil
import json
import traceback
from datetime import datetime, timezone
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, Any, Dict, Optional

logger = logging.getLogger("smart_monitor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class SystemDiagnostics:
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """Returns sophisticated system health metrics."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_usage_mb": round(mem_info.rss / 1024 / 1024, 2),
            "disk_usage": psutil.disk_usage('/').percent,
            "thread_count": process.num_threads(),
            "uptime_seconds": time.time() - process.create_time()
        }

class AnomalyDetector:
    # Thresholds based on typical agricultural data ranges
    THRESHOLDS = {
        "temperature": {"min": 0, "max": 50, "unit": "°C"},
        "rainfall": {"min": 0, "max": 5000, "unit": "mm"}, # Annual rainfall can be high, but monthly/daily varies. Assume input is relevant period.
        "humidity": {"min": 0, "max": 100, "unit": "%"},
        "ph": {"min": 0, "max": 14, "unit": ""},
        "nitrogen": {"min": 0, "max": 500, "unit": "kg/ha"}, # High bound
        "phosphorus": {"min": 0, "max": 500, "unit": "kg/ha"},
        "potassium": {"min": 0, "max": 500, "unit": "kg/ha"},
        "light_intensity": {"min": 0, "max": 200000, "unit": "lux"} # Sun can be ~100k lux
    }

    @classmethod
    def detect_anomalies(cls, data: Dict[str, Any]) -> list[str]:
        """Detects if input values are out of realistic bounds."""
        warnings = []
        for key, value in data.items():
            if key in cls.THRESHOLDS:
                try:
                    val = float(value)
                    rules = cls.THRESHOLDS[key]
                    if val < rules["min"] or val > rules["max"]:
                        warnings.append(
                            f"Anomali terdeteksi: {key} bernilai {val} {rules['unit']} "
                            f"(Normal: {rules['min']}-{rules['max']})"
                        )
                except (ValueError, TypeError):
                    continue # Should be handled by type validation
        return warnings

class SmartMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        path = request.url.path
        
        # 1. Smart Logging - Start
        # logger.info(f"Incoming Request: {request.method} {path}")

        try:
            response = await call_next(request)
            
            # 2. Performance Monitoring
            process_time = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            log_level = logging.INFO
            if process_time > 1000: # Slower than 1s
                log_level = logging.WARNING
                logger.warning(f"Slow Request Detected: {request.method} {path} took {process_time:.2f}ms")
            elif status_code >= 500:
                log_level = logging.ERROR
                logger.error(f"Server Error: {request.method} {path} returned {status_code}")
            
            # logger.log(log_level, f"Completed: {request.method} {path} - Status: {status_code} - Time: {process_time:.2f}ms")
            
            return response
            
        except Exception as e:
            # 3. Smart Error Catching & Recovery
            process_time = (time.time() - start_time) * 1000
            logger.critical(f"Unhandled Exception in {path}: {str(e)}")
            logger.critical(traceback.format_exc())
            
            # Re-raise to let FastAPI's exception handler handle the response format
            raise e 
