# shogi-human-policy

レーティング、残り持ち時間、局面履歴から、人間が次に指す手の確率を予測する将棋モデルです。
棋譜の途中局面を調べたり、入力Rを変えて着手傾向を比較したりできます。

現在はローカル準備版です。
GitHubリポジトリと重みの配布先はまだ作成していません。

## まず試す

Python 3.11〜3.13を使用します。
推論に学習リポジトリ、JAX、TPU、棋譜DBは不要です。
以下のコマンドは、このリポジトリのルートで実行します。

```console
uv sync
uv run shogi-human-policy predict --model models/preview-step-0605000 --position "startpos moves 7g7f 3c3d"
```

モデルディレクトリには、同じモデルの`manifest.json`と`policy.safetensors`を置きます。
読み込み時に重みのSHA-256と構造を照合します。
重みはGitに含めません。

## 入力できるもの

| 手元の情報 | 入力方法 |
|---|---|
| 現在局面のSFEN | `--position "<SFENの4フィールド>"` |
| SFENとその後の着手 | `--position "<SFEN> moves 7g7f 3c3d"` |
| 平手初期局面からの着手 | `--position "startpos moves 7g7f 3c3d"` |
| KIF / KIFU / KI2 / CSA | `--file game.kif` |
| 棋譜の途中局面 | `--file game.kif --ply 30` |

`--ply 30`は開始局面から30手指した後で、平手初期局面からなら31手目を予測します。
`0`は開始局面、省略時は末尾です。
履歴も予測に使うため、分かる場合はSFEN単体より棋譜や着手列を渡してください。

付属のサンプルで、2手指した後を試せます。

```console
uv run shogi-human-policy predict --model models/preview-step-0605000 --file examples/records/opening.kif --ply 2 --top-k 5
```

標準出力には、日本語表記、USI表記、確率を表示します。
機械処理には`--format json`を指定してください。
詳しくは[入力と出力](docs/inference.md)を参照してください。

## Pythonから使う

```python
from shogi_human_policy import Conditions, PolicyModel, Position

model = PolicyModel.from_directory("models/preview-step-0605000")
position = Position.from_file("examples/records/opening.kif", ply=2)
result = model.predict(
    position,
    Conditions(ratings=(1500, 1800)),  # 現在の手番側R、相手R。時計は不明。
    top_k=5,
)
for row in result.as_rows(position):
    print(row["notation"], row["usi"], f"{row['probability']:.2%}")
```

`Position.from_sfen(sfen, moves=["7g7f", "3c3d"])`や、`Position.from_usi("startpos moves ...")`も使えます。
`Conditions()`ならRも時計も不明として推論します。
時計を指定する場合は、`clocks=(自分の利用可能秒数, 相手の利用可能秒数)`を渡します。
本時間と秒読みがある場合は合算し、当該手に実際に使った時間を混ぜないでください。
Rと時計は各組を2人分まとめて指定または省略します。
棋譜の名前や時間ヘッダから自動推定はしません。

## Colabで試す

[デモノートブック](notebooks/demo.ipynb)をColabへアップロードして実行できます。
SFENの貼り付け、棋譜アップロード、途中局面の選択、盤面表示、日本語の上位候補表示に対応しています。
先手と後手のRを指定でき、手番に応じた自他の振り分けも行います。

現時点では、ローカルで`uv build --wheel`を実行して作ったwheelと、モデルの2ファイルをノートブックからアップロードします。
標準のCPUランタイムで使用できます。
[実行手順](docs/colab.md)も参照してください。

## 予測性能

学習途中の605,000更新EMAを、固定した自然頻度のvalidation 50,000局面で評価した値です。
未使用testは使っていません。

| 人間の着手との一致率 | policy NLL |
|---:|---:|
| 58.75% | 1.2677 |

![実Rと入力Rごとの正解率](docs/assets/rating-input-accuracy.png)

両軸は200刻みで、丸は各実R帯で最良だった入力帯です。
この図は各R帯2,000局面で、20手刻みの手数帯の内訳を揃えた別のvalidation標本です。
[入力Rの効果](docs/rating-response.md)と[評価結果](docs/evaluation.md)に、条件の省略、履歴、手数帯ごとの比較を掲載しています。

## モデルと利用範囲

最大8局面の履歴と、R、時計、条件の既知性を入力するAdaLN付きTransformerです。
学習時には条件の省略と履歴のランダム短縮を行っています。
確率は全合法手を分母としており、上位候補だけを表示した場合も再正規化しません。
指定Rでの対局棋力を保証するものではありません。
valueは今回の人間棋譜で学習していないため、APIには含めていません。

[モデルの来歴](docs/model-card.md) / [入力仕様](docs/inference.md) / [モデル形式](docs/model-format.md)

## 開発

```console
uv sync --group dev --group analysis
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

CPU float32で検証しています。
R帯比較ではCUDA float32（TF32無効）も使用し、CPUとの数値比較を行っています。
図の再生成には日本語フォントが必要です。
公開前に確定する項目は[モデルの来歴](docs/model-card.md)に記載しています。
