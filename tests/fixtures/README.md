# 数値比較用fixture

`jax-reference.npz`は小型のランダム初期化モデルと、手書きの合法着手列から作成した48入力のfloat32出力です。
棋譜DBの局面や公開予定モデルの学習済み重みは含みません。
`param:`はpolicyに必要なパラメータ、`input:`は入力、`expected`は1496 logitです。
構造は`config.json`に記録しています。

参照元はrsshogi-az-trainerの`aztrainer.aligned.model.ShogiTransformer`、commit `f3d29c8908c0c238ff7786608639be37941bd9af`です。
JAX 0.12.0.1、Flax 0.12.12、CPU float32、初期化seed 317を使用しました。
AdaLNとpromotionのパラメータにはNumPyのseed 317で標準偏差0.1の正規乱数を設定し、ゼロ初期化によって条件経路が未検証になることを避けています。
QK RMS正規化も有効にしています。

入力は`startpos`、`7g7f`、`7g7f 3c3d`、`7g7f 3c3d 8h2b+ 3a2b B*4e`の各着手列に、履歴長0/3/7と4種類の条件を組み合わせています。
条件は「両方省略」「R=(300,2700)」「時計=(0,60)」「R=(1500,2100), 時計=(10,300)」です。
テストの実行にJAXとFlaxは不要です。
