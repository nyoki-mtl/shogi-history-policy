# 実Rと入力Rの対応

両軸とも0以上3000未満の200刻み。丸は、各実R帯で最良だった入力帯です。
入力には各帯の中央（100、300、…、2900）を使っています。

![実Rと入力Rごとの正解率](assets/rating-input-accuracy.png)

[拡大用SVG](assets/rating-input-accuracy.svg)

## 1500固定からの改善

青は改善、赤は悪化です。

![1500固定からの正解率の変化](assets/rating-input-accuracy-delta.png)

[拡大用SVG](assets/rating-input-accuracy-delta.svg)

## 損失で見る

青は損失の減少、赤は増加です。

![1500固定からの損失の変化](assets/rating-input-nll.png)

[拡大用SVG](assets/rating-input-nll.svg)

605,000更新EMA、検証用12,611局面。同じ局面で自分の入力Rだけを変えています。
丸は標本上の最良値で、細かなRの優劣を確定するものではありません。

[数値表と測定条件](rating-response-details.md) / [集計JSON](metrics/preview-step-0605000-rating-matrix.json)
