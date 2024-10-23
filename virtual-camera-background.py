import cv2
import numpy as np
import pyvirtualcam
from pyvirtualcam import PixelFormat
import mediapipe as mp
import sys
import logging
from datetime import datetime


class VirtualBackgroundCamera:
    def __init__(self,
                 background_image_path=None,
                 camera_index=0,
                 preview_mode=False,
                 blur_amount=7,
                 edge_blur=3,
                 feather_amount=10):
        """
        背景置換機能付き仮想カメラの初期化

        Args:
            background_image_path: 背景画像のパス（Noneの場合は青背景）
            camera_index: カメラデバイスのインデックス
            preview_mode: プレビュー表示の有無
            blur_amount: マスクのぼかし量（奇数、大きいほどぼける）
            edge_blur: エッジ検出のぼかし量
            feather_amount: 境界のフェザリング量
        """
        self.blur_amount = blur_amount
        self.edge_blur = edge_blur
        self.feather_amount = feather_amount

        # 通常の初期化処理
        self.setup_logging()

        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"カメラ（インデックス: {camera_index}）を開けませんでした")

            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.preview_mode = preview_mode

            self.logger.info(f"カメラの解像度: {self.width}x{self.height}")

            if background_image_path:
                self.background = cv2.imread(background_image_path)
                if self.background is None:
                    self.logger.warning("背景画像を読み込めませんでした。デフォルトの青背景を使用します")
                    self.background = np.full((self.height, self.width, 3), (255, 0, 0), dtype=np.uint8)
                else:
                    self.background = cv2.resize(self.background, (self.width, self.height))
            else:
                self.logger.info("デフォルトの青背景を使用します")
                self.background = np.full((self.height, self.width, 3), (255, 0, 0), dtype=np.uint8)

            self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
            self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

            self.frame_count = 0
            self.start_time = datetime.now()

        except Exception as e:
            self.logger.error(f"初期化中にエラーが発生しました: {str(e)}")
            raise


    def enhance_mask(self, mask):
        """マスクの品質を改善"""
        # マスクを浮動小数点に変換
        mask = mask.astype(np.float32)

        # ガウシアンブラーでマスクをぼかす
        blurred_mask = cv2.GaussianBlur(mask, (self.blur_amount, self.blur_amount), 0)

        # エッジ検出でマスクの境界を強調
        edges = cv2.Canny(
            (blurred_mask * 255).astype(np.uint8), 
            threshold1=30, 
            threshold2=100
        )
        dilated_edges = cv2.dilate(edges, None, iterations=2)
        edge_mask = cv2.GaussianBlur(
            dilated_edges.astype(np.float32), 
            (self.edge_blur, self.edge_blur), 
            0
        ) / 255.0

        # マスクとエッジをブレンド
        enhanced_mask = cv2.addWeighted(
            blurred_mask, 0.7,
            edge_mask, 0.3,
            0
        )

        # フェザリング（境界のソフト化）
        feathered_mask = cv2.GaussianBlur(
            enhanced_mask,
            (self.feather_amount * 2 + 1, self.feather_amount * 2 + 1),
            0
        )

        return np.clip(feathered_mask, 0, 1)


    def blend_images(self, frame, background, mask):
        """フレームと背景を自然にブレンド"""
        # マスクを3チャンネルに拡張
        mask_3channel = np.stack([mask] * 3, axis=-1)

        # アルファブレンディング
        blended = cv2.addWeighted(
            (mask_3channel * frame).astype(np.float32), 1,
            ((1 - mask_3channel) * background).astype(np.float32), 1,
            0
        ).astype(np.uint8)

        return blended


    def process_frame(self, frame):
        """1フレームの処理"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(rgb_frame)

            if results.segmentation_mask is None:
                self.logger.warning("セグメンテーションに失敗しました")
                return frame

            # マスクの品質改善
            enhanced_mask = self.enhance_mask(results.segmentation_mask)

            # 画像のブレンド
            output_frame = self.blend_images(frame, self.background, enhanced_mask)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_fps = self.get_fps()
                self.logger.debug(f"現在のFPS: {current_fps:.2f}")

            return output_frame

        except Exception as e:
            self.logger.error(f"フレーム処理中にエラーが発生しました: {str(e)}")
            return frame

    # 以下のメソッドは前回と同じなので省略（setup_logging, get_fps, create_preview, run, cleanup）
    # 必要な場合は前回のコードをそのまま使用してください


    def setup_logging(self):
        """ロギングの設定"""
        self.logger = logging.getLogger('VirtualCamera')
        self.logger.setLevel(logging.INFO)

        # 既存のハンドラをクリア
        if self.logger.handlers:
            self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # コンソールハンドラ
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)


    def get_fps(self):
        """現在のFPSを計算"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        return self.frame_count / elapsed_time if elapsed_time > 0 else 0


    def process_frame(self, frame):
        """1フレームの処理"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(rgb_frame)

            if results.segmentation_mask is None:
                self.logger.warning("セグメンテーションに失敗しました")
                return frame

            mask = results.segmentation_mask
            mask = np.stack((mask,) * 3, axis=-1) > 0.1
            output_frame = np.where(mask, frame, self.background)

            self.frame_count += 1
            if self.frame_count % 30 == 0:  # 30フレームごとにFPSを表示
                current_fps = self.get_fps()
                self.logger.debug(f"現在のFPS: {current_fps:.2f}")

            return output_frame

        except Exception as e:
            self.logger.error(f"フレーム処理中にエラーが発生しました: {str(e)}")
            return frame


    def create_preview(self, original_frame, processed_frame):
        """プレビュー用の画像を作成"""
        try:
            preview = np.hstack((original_frame, processed_frame))

            # FPS表示を追加
            current_fps = self.get_fps()
            cv2.putText(preview, f"FPS: {current_fps:.2f}", 
                      (10, self.height - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.putText(preview, "Original", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(preview, "Processed", (self.width + 10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return preview
        except Exception as e:
            self.logger.error(f"プレビュー作成中にエラーが発生しました: {str(e)}")
            return original_frame


    def run(self):
        """仮想カメラの実行"""
        try:
            with pyvirtualcam.Camera(width=self.width, height=self.height, fps=30, fmt=PixelFormat.BGR) as cam:
                self.logger.info(f'仮想カメラを起動しました: {cam.device}')
                self.logger.info(f'解像度: {self.width}x{self.height}, FPS: 30')

                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        self.logger.error("カメラからフレームを読み取れませんでした")
                        break

                    output_frame = self.process_frame(frame)

                    if self.preview_mode:
                        preview = self.create_preview(frame, output_frame)
                        cv2.imshow('Virtual Camera Preview', preview)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.logger.info("ユーザーによって終了されました")
                            break

                    cam.send(output_frame)
                    cam.sleep_until_next_frame()

        except Exception as e:
            self.logger.error(f"仮想カメラの実行中にエラーが発生しました: {str(e)}")
            raise
        finally:
            self.cleanup()


    def cleanup(self):
        """リソースの解放"""
        try:
            self.cap.release()
            cv2.destroyAllWindows()
            self.logger.info("カメラリソースを解放しました")
        except Exception as e:
            self.logger.error(f"クリーンアップ中にエラーが発生しました: {str(e)}")


if __name__ == "__main__":
    try:
        virtual_cam = VirtualBackgroundCamera(
            preview_mode=False,
            blur_amount=15,      # マスクのぼかし量（奇数値）
            edge_blur=6,        # エッジのぼかし量
            feather_amount=20,   # 境界のフェザリング量
            background_image_path="./images/background-example.jpg", # "path/to/your/background.jpg",
            camera_index=0,
        )
        virtual_cam.run()
    except KeyboardInterrupt:
        print("\nプログラムを終了します")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        sys.exit(1)
