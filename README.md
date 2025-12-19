## 全天球画像生成モデル（開発中）

## 開発環境
・Ubuntu 20.04
・CUDA12.9
・Python 3.10

## 必要なライブラリをインストール
	pip install -r requirements.txt
## 学習するデータセットの学習用VQGANコード作成
	python generate_dataset.py

## モデルのトレーニング
	python train.py

## inference
モデルを指定して生成
```
python image_generation.py \--model_path trained_modelpath
```

	
