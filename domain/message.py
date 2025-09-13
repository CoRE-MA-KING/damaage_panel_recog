from pydantic import BaseModel


class RobotCommand(BaseModel):
    """ロボットに送信するコマンド"""

    target_x: int = 640
    target_y: int = 360
    target_distance: int = 0
    dummy: int = 0  # 未使用
