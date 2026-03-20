"""PlantNet API による植物種同定。"""

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# PlantNet API のステータスコードに対応するエラーメッセージ
_STATUS_MESSAGES: Dict[int, str] = {
    400: "不正なリクエストです。画像や器官の指定を確認してください。",
    401: "APIキーが無効です。正しいキーを設定してください。",
    404: "植物を特定できませんでした。別の画像を試してください。",
    429: "APIの呼び出し回数制限に達しました。しばらく待ってから再試行してください。",
    500: "PlantNet APIでサーバーエラーが発生しました。しばらく待ってから再試行してください。",
}


@dataclass
class PlantNetResult:
    """APIのレスポンスとメタデータをまとめるデータクラス"""

    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    remaining: Optional[int] = None
    limit: Optional[int] = None


class PlantNetClient:
    """PlantNet APIを呼び出すクライアントクラス"""

    API_BASE = "https://my-api.plantnet.org/v2/identify"
    MAX_IMAGES = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        project: str = "japan",
        lang: str = "ja",
    ):
        """クライアントを初期化する。

        Args:
            api_key: PlantNet API キー。省略時は環境変数から取得。
            project: PlantNet プロジェクト名。
            lang: レスポンスの言語コード。
        """
        self.api_key = api_key or os.getenv("PLANTNET_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API キーが必要です。環境変数 PLANTNET_API_KEY を設定してください。"
            )
        self.project = project
        self.lang = lang
        self.last_remaining: Optional[int] = None
        self.limit: Optional[int] = None

    def _update_quota_info(
        self, headers: requests.structures.CaseInsensitiveDict
    ) -> None:
        """ヘッダーから残り回数と最大回数を抽出して内部状態を更新する"""
        rl_header = headers.get("ratelimit")
        if rl_header:
            rem_match = re.search(r"r=(\d+)", rl_header)
            if rem_match:
                self.last_remaining = int(rem_match.group(1))

        policy_header = headers.get("ratelimit-policy")
        if policy_header:
            limit_match = re.search(r"q=(\d+)", policy_header)
            if limit_match:
                self.limit = int(limit_match.group(1))

    def _encode_image(self, img: np.ndarray, index: int) -> Optional[tuple]:
        """画像をJPEGバイナリにエンコードし、multipart用タプルを返す"""
        success, buf = cv2.imencode(".jpg", img)
        if not success:
            logger.warning(
                "画像 %d のエンコードに失敗しました。スキップします。", index
            )
            return None
        image_content = io.BytesIO(buf.tobytes())
        image_content.seek(0)
        return ("images", (f"image_{index}.jpg", image_content, "image/jpeg"))

    def _build_multipart(
        self, images: List[np.ndarray], organs: List[str]
    ) -> tuple:
        """画像リストからmultipartのfiles/dataを構築する"""
        files = []
        data = []
        for i, img in enumerate(images):
            encoded = self._encode_image(img, i)
            if encoded is not None:
                files.append(encoded)
                data.append(("organs", organs[i % len(organs)]))
        return files, data

    def _error_message_for(
        self, status_code: int, response: Optional[requests.Response] = None
    ) -> str:
        """ステータスコードに応じたエラーメッセージを返す"""
        if status_code in _STATUS_MESSAGES:
            return _STATUS_MESSAGES[status_code]
        if response is not None:
            try:
                body = response.json()
                if "message" in body:
                    return f"APIエラー: {body['message']}"
            except (ValueError, KeyError):
                pass
        return f"APIリクエストが失敗しました (HTTP {status_code})。"

    def identify(
        self, images: List[np.ndarray], organs: Optional[List[str]] = None
    ) -> PlantNetResult:
        """画像リストを送信して樹種を判別する"""
        if not images:
            return PlantNetResult(
                success=False,
                status_code=0,
                error_message="画像が指定されていません。",
            )

        if organs is None:
            organs = ["auto"]

        if len(images) > self.MAX_IMAGES:
            logger.warning(
                "画像は最大%d枚までです。最初の%d枚のみを使用します。",
                self.MAX_IMAGES,
                self.MAX_IMAGES,
            )
            images = images[: self.MAX_IMAGES]

        files, data = self._build_multipart(images, organs)
        if not files:
            return PlantNetResult(
                success=False,
                status_code=0,
                error_message="すべての画像のエンコードに失敗しました。",
            )

        url = f"{self.API_BASE}/{self.project}"
        params = {"api-key": self.api_key, "lang": self.lang}

        try:
            res = requests.post(
                url, params=params, files=files, data=data, timeout=30
            )
        except requests.exceptions.ConnectionError:
            logger.error("ネットワーク接続に失敗しました。")
            return PlantNetResult(
                success=False,
                status_code=0,
                error_message="ネットワーク接続に失敗しました。インターネット接続を確認してください。",
            )
        except requests.exceptions.Timeout:
            logger.error("リクエストがタイムアウトしました。")
            return PlantNetResult(
                success=False,
                status_code=0,
                error_message="リクエストがタイムアウトしました。しばらく待ってから再試行してください。",
            )
        except requests.exceptions.RequestException as e:
            logger.error("リクエストエラーが発生しました: %s", e)
            return PlantNetResult(
                success=False,
                status_code=0,
                error_message=f"リクエストエラー: {e}",
            )

        self._update_quota_info(res.headers)

        if res.status_code == 200:
            return PlantNetResult(
                success=True,
                status_code=200,
                data=res.json(),
                remaining=self.last_remaining,
                limit=self.limit,
            )

        error_msg = self._error_message_for(res.status_code, res)
        logger.error("API エラー (HTTP %d): %s", res.status_code, error_msg)
        return PlantNetResult(
            success=False,
            status_code=res.status_code,
            error_message=error_msg,
            remaining=self.last_remaining,
            limit=self.limit,
        )


def _format_results(data: Dict[str, Any], top_n: int = 10) -> list[str]:
    """API レスポンスから上位結果を整形したテキスト行のリストを返す。

    Args:
        data: PlantNet API のレスポンスデータ。
        top_n: 表示する上位件数。

    Returns:
        整形済みテキスト行のリスト。
    """
    lines = [f"=== 推論結果 (Top {top_n}) ==="]
    for i, match in enumerate(data.get("results", [])[:top_n], 1):
        species = match["species"]["scientificNameWithoutAuthor"]
        common_names = match["species"].get("commonNames")
        common = common_names[0] if common_names else "N/A"
        score = match["score"] * 100
        lines.append(f"{i}. {species} ({common}) - 確信度: {score:.1f}%")
    return lines


def _show_result_figure(img: np.ndarray, results_texts: list[str]) -> None:
    """画像と判定結果を Matplotlib で並列表示する。

    Args:
        img: BGR 形式の入力画像。
        results_texts: 表示するテキスト行のリスト。
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(img_rgb)
    ax1.axis("off")
    ax1.set_title("Input Image")

    ax2.axis("off")
    ax2.text(
        0.0,
        0.95,
        "\n".join(results_texts),
        fontsize=12,
        va="top",
        ha="left",
        transform=ax2.transAxes,
    )
    plt.tight_layout()
    plt.show()


def main(image_path: str = "tmp/image.png", no_display: bool = False) -> None:
    """画像を PlantNet API で判定し結果を表示する。

    Args:
        image_path: 入力画像のパス。
        no_display: True の場合、Matplotlib 表示をスキップ。
    """
    path = Path(image_path)
    if not path.exists():
        print(f"画像が見つかりません: {path}")
        raise SystemExit(1)

    img = cv2.imread(str(path))
    if img is None:
        print(
            "画像を読み込めませんでした。ファイルが壊れている可能性があります。"
        )
        raise SystemExit(1)

    client = PlantNetClient()
    print("PlantNet API に問い合わせ中...\n")
    result = client.identify([img])

    if not result.success or not result.data:
        print(f"判別に失敗しました: {result.error_message}")
        raise SystemExit(1)

    lines = _format_results(result.data)
    for line in lines:
        print(line)
    print(
        f"\n[状態確認] 残りAPI呼び出し回数: "
        f"{client.last_remaining}/{client.limit}"
    )

    if not no_display:
        _show_result_figure(img, lines)


if __name__ == "__main__":
    main()
