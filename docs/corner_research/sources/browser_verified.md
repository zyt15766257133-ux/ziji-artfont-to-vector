# 浏览器补核验摘录

核验日期：2026-10-17。以下是读取公开网页后的关键摘录，不是完整页面存档。

## Adobe Illustrator

URL: https://helpx.adobe.com/illustrator/desktop/manage-objects/traces-mockups-symbols/image-trace-panel-options.html

页面标题：Image Trace panel options。页面显示 Last updated on Feb 26, 2026。

> Paths: Controls the distance between the traced shape and the original pixel shape. Lower values create looser path fitting, while higher values create tighter path fitting.

> Corners: Specifies the emphasis on corners and the likelihood that a sharp bend will turn into a corner point. A higher value results in more corners.

> Noise: Specifies an area in pixels that is ignored while tracing. A higher value results in less noise. For high-res images, set to 20–50. For low-res, use 1–10.

> Snap Curves To Lines: This function replaces slightly curved lines with straight lines and snaps lines near 0 or 90 degrees to absolute 0 or 90 degrees.

Shapes 一项同时说明 isolated perfect circles/squares 才能识别，稍有旋转的矩形可能较不准确。

## Potrace 手册

URL: https://potrace.sourceforge.net/potrace.1.html

> -a n, --alphamax n: set the corner threshold parameter. The default value is 1. The smaller this value, the more sharp corners will be produced. If this parameter is 0, then no smoothing will be performed and the output is a polygon. If this parameter is greater than 4/3, then all corners are suppressed and the output is completely smooth.

> -O n, --opttolerance n: set the curve optimization tolerance. The default value is 0.2. Larger values allow more consecutive Bezier curve segments to be joined together in a single segment, at the expense of accuracy.

## PolyVector Fields

URL: https://arxiv.org/html/1801.01922v2

标题：Vectorization of Line Drawings via PolyVector Fields。Mikhail Bessmeltsev、Justin Solomon。页面标注 05 Sep 2018。

Figure 1 说明方向场在尖角、X/T 交点处叠加多个方向，再提取拓扑和矢量曲线。

正文结果表列出的实现用时为 23～81 秒；Parameters and Processing Time 说明历史测试机器为 4-core Intel i7-6700 @ 3.4GHz、32GB RAM。此为作者历史环境，非本次本机测量。

Limitations 说明阴影、极低分辨率、多重涂抹线条会形成挑战。Figure 20 和参数段落说明提高方向场平滑权重可让交点更锐利，但可能损失细节。
