from pydantic import BaseModel


class Target(BaseModel):
    x: int = 640
    y: int = 360
    distance: int = 0


class DamagePanelRecognition(BaseModel):
    """ロボットに送信するコマンド"""

    target: Target
