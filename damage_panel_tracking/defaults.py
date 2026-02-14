from __future__ import annotations

from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "camera": {
        "device": "/dev/video0",
        "capture": {"width": 800, "height": 600, "fps": 120, "fourcc": "MJPG"},
        "init_controls": {
            "brightness": 0,
            "contrast": 32,
            "saturation": 90,
            "hue": 0,
            "white_balance_automatic": 0,
            "gamma": 100,
            "gain": 0,
            "power_line_frequency": 1,
            "white_balance_temperature": 4600,
            "sharpness": 3,
            "backlight_compensation": 0,
            "auto_exposure": 1,
            "exposure_time_absolute": 4,
        },
    },
    "detection": {
        "kernel_sz": 3,
        "width_tol": 0.6,
        "min_h_overlap": 0.05,
        "min_v_gap": 1,
        "min_box_h": 1,
        "min_box_w": 4,
        "hsv": {
            "blue": {"H_low": 100, "H_high": 135, "S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
            "red1": {"H_low": 0, "H_high": 15},
            "red2": {"H_low": 165, "H_high": 179},
            "redSV": {"S_low": 180, "S_high": 255, "V_low": 120, "V_high": 255},
        },
    },
    "tracking": {
        "enabled": False,
        "backend": "motpy",  # motpy | distance | noop (future: sort)
        "min_steps_alive": 2,
        "history_len": 20,
        "color_bgr": [0, 255, 255],
        "motpy": {
            "order_pos": 2,
            "dim_pos": 2,
            "order_size": 0,
            "dim_size": 2,
            "q_var_pos": 5000.0,
            "r_var_pos": 0.1,
            "min_iou": None,
            "max_staleness": 4,
        },
        "distance": {
            "gate_px": 80.0,
            "normalize": "diag",
            "size_weight": 0.15,
            "use_prediction": True,
            "vel_alpha": 0.6,
            "max_speed_px_s": 8000.0,
            "max_age": 4,
        },
    },
    "publish": {
        "enabled": False,
        "publish_key": "damagepanel/target",
    },
    "subscribe": {
        "enabled": False,
        "subscribe_key": "damagepanel/color",
        "default_target": "blue",
    },
    "ui": {
        "window_name": "Panel (paired by same-color top & bottom)",
        "fps_ema_alpha": 0.2,
    },
    "logging": {
        "enabled": False,
        "path": None,
        "flush_every": 60,
    },
}
