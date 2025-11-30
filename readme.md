### To run
example: \
python -m src.viewer.viewer \
python -m src.classifier.ground

### 依存関係
conda の environment.yml にはいろいろ書いてるけど、
今のとこ pdal は使ってないから pip だけでもよさそう

### やりたいこと
src/lib の中、segment フォルダ作りたい \
名前の統一：Detector, Classifier 

### To Do
いろいろ踏まえて、3dgs を 点群にしっかり標準化してからやったほうがいい気がしてきた \
今のところでかいガウスも小さいガウスも同じ重みで処理しているため