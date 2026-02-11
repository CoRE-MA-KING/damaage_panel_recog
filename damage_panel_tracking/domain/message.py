from pydantic import BaseModel


class Target(BaseModel):
    x: int
    y: int
    distance: int
    width: int
    height: int


class DamagePanelRecognition(BaseModel):
    """ロボットに送信するコマンド"""

    target: Target | None = None
