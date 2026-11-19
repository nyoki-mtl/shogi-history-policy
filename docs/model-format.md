# 配布形式

1モデルを`manifest.json`と`policy.safetensors`の2ファイルで保存します。
学習用checkpointのoptimizer、EMA更新状態、未学習のvalue系headは含めません。

manifestにはschema `shogi-human-policy-v1`、構造、EMAの更新数、1496ラベル、8履歴、入力正規化、履歴padding、geometryと重みのSHA-256を記録します。
読み込み時にこれらと全パラメータ名、shape、有限性を検査します。
pickleは読み込みません。
geometryはパッケージに同梱した定数で、モデルとラベル順が一致することをSHA-256で照合します。
未知の構造や入力契約は受け付けません。

現在の準備版のmanifestは[model-manifest.json](model-manifest.json)を参照してください。
重み自体と棋譜データはGitに含めません。
学習リポジトリからの変換とcheckpoint取得は、本リポジトリの推論APIの担当外です。
