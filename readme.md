### To run
example:
python -m src.viewer.viewer

### 依存関係
conda の environment.yml にはいろいろ書いてるけど、
今のとこ pdal は使ってないから pip だけでもよさそう

### やりたいこと
src/lib の中、segment フォルダ作りたい
名前の統一：Detector, Classifier 

### To Do
classifier/test.py で RANSAC を試している
2D DBSCAN したあとに RANSAC で円柱チェックをしたい