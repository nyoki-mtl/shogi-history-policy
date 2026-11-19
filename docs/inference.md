# 入力仕様

## 局面と履歴

### SFENとUSI着手列

```python
position = Position.from_sfen(sfen)
position = Position.from_sfen(sfen, moves=["7g7f", "3c3d"])
position = Position.from_usi("startpos moves 7g7f 3c3d")
position = Position.from_usi(sfen + " moves 7g7f 3c3d")
```

`from_usi`は、先頭の`position`や`sfen`の有無を受け付けます。
`from_sfen`には、盤面、手番、持ち駒、手数の4フィールドを渡します。
着手列はUSI形式で、SFENの局面から進める順番に指定します。
`Position()`は平手初期局面です。

### 棋譜ファイル

```python
position = Position.from_file("game.kif", ply=30)
position = Position.from_file("game.kifu")
position = Position.from_kif(kif_text, ply=30)
```

KIF/KIFU、KI2、CSAの読み込みは`rsshogi`を使用します。
`.sfen`と`.usi`は、1局面分のSFENまたはUSI position文字列を保存したテキストとして読み込みます。
文字コードはUTF-8（BOMありを含む）を先に試し、デコードできない場合はCP932（Shift-JIS）を使います。
必要なら`encoding="utf-16"`などで明示できます。

`ply`は入力の開始局面から適用する着手数です。
30なら開始局面から30手指した後を使い、次の着手を予測します。
0なら開始局面、`None`または省略なら末尾です。
SFENに含まれる絶対手数とは区別します。
例えば手数100のSFENから2手進めた局面の絶対手数は102です。
`position.at_ply(30)`でも同じ切り出しができます。
範囲外の手数はエラーになります。

変化を含む棋譜は本譜だけを使用します。
投了や時間切れなどの終局記号は着手数に含めず、末尾を選ぶと終局記号の直前の盤面になります。
棋譜のヘッダにある名前、Rらしき文字列、持ち時間から推論条件を自動設定することはありません。

### 履歴と再利用

棋譜や着手列を渡すと、途中局面でもそこまでの履歴を保持します。
現在局面と最大7手前までを使い、足りない履歴は盤面0、持ち駒の符号化値31で埋めます。
SFEN単体からは、それ以前の着手や反復を復元できません。

入力は現在の手番側から見た向きに統一します。
後手番では盤面を180度回転して駒色を反転し、持ち駒の自他を入れ替えます。
SFENの手数から絶対手数を引き継ぎます。
320は事前学習と共通の手数正規化の基準で、320手を超える局面も利用できます。

`history_plies=0`なら現在局面だけ、既定の7なら最大8局面です。
短縮時には残した履歴内で反復回数を数え直します。
`position.to_usi()`は履歴を含むposition文字列、`position.to_sfen()`は選んだ現在局面だけのSFENを返します。
`position.to_board()`で独立した`rsshogi`の盤面を取得できます。

### CLI

```console
shogi-human-policy predict --model models/preview-step-0605000 --position "startpos" --moves 7g7f 3c3d
shogi-human-policy predict --model models/preview-step-0605000 --file game.kif --ply 30
shogi-human-policy predict --model models/preview-step-0605000 --file game.kif --ply 30 --format json
```

`--position`と`--file`はどちらか一方です。
両方を省略すると平手初期局面になります。
`--moves`は`--position`に着手を追加し、`--ply`はその後に適用します。
ファイルと追加の着手列は混ぜません。
`--encoding`はファイル入力だけで使用できます。

## レーティングと時計

| 引数 | 順序と意味 | 不明時 |
|---|---|---|
| `Conditions.ratings` | `(手番側R, 相手R)` | `None` |
| `Conditions.clocks` | `(手番側の利用可能秒数, 相手の利用可能秒数)` | `None` |

有限な非負の数を指定します。
レーティングは学習元の将棋倶楽部24の尺度を想定し、他サービスの値への換算は検証していません。
自他片方だけが分からない場合、その組全体を省略してください。
モデルは個人識別情報を入力しません。
駒落ちで先後の値を指定する場合は、黒を下手、白を上手として扱います。

時計は着手前の状態です。
当該手に実際に使った時間は入力しません。
通常の秒読みなら本時間とその手で使える秒読みを合計し、切れ負けなら残り本時間を使います。
フィッシャーは現在の残り時間を使い、その手を完了した後の加算を含めません。
考慮時間など特殊な時計では、そのルールに従って当該手で使える量を算出します。
相手についても現在観測している時計状態を使い、未来の消費時間を予測して差し引きません。
USIのミリ秒を使う場合は秒へ変換します。
このAPIはルール文字列からの時計復元を行いません。

Rは`(R - 1500) / 1500`、時計は`log1p(秒数) / log1p(3600)`へ変換します。
不明な組は正規化後に0へ置換し、別の既知性maskも渡します。
0秒を不明扱いする処理は入れません。

## 出力

`Prediction.moves`は確率降順の合法手です。
`probability`は全合法手に対する確率、`label`は1496次元の内部ラベルです。
`result.as_rows(position)`でUSI、日本語表記、確率の辞書列を取得できます。
日本語表記は、その予測を行った同じ局面に対して生成してください。
`entropy`は全合法手分布のエントロピー（自然対数）、`legal_count`は合法手数です。
合法手がない局面では空の着手列とentropy 0を返します。
温度は既定1で、正の有限値を指定できます。
評価には必ず温度1を使います。

`predict_batch(positions, conditions=None)`で複数局面を処理できます。
条件を渡す場合は局面数と同じ長さの`Conditions`列を渡します。
CLIはスレッド数を既定4に設定し、ライブラリAPIは呼び出し元の設定を維持します。

このライブラリは着手分布を予測します。
探索、投了判断、時間配分を行うUSIエンジンは含みません。
