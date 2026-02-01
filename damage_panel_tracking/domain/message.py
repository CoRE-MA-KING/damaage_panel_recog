from pydantic import BaseModel


class DamagePanelRecognition(BaseModel):
    """ロボットに送信するコマンド"""

    target_x: int = 640
    target_y: int = 360
    target_distance: int = 0
