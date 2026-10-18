# Notes for an app

What an app that reads and solves a photographed sudoku would take, starting from this repository.

## The pipeline as it stands

`solve_image.read_and_solve(image)` runs the whole chain on one image:

1. **Cut**: a 272 × 272 image (pixels in [0, 1], white ink on black, 2-pixel grid lines) into 81 cells of 28 × 28.
2. **Classify**: the CNN (seed 0) gives 10 logits per cell; class 0 is a blank cell.
3. **Repair**: MaxSAT picks the most probable reading that some valid grid satisfies, then the uniqueness loop rejects readings with several solutions (at most 20 MaxSAT calls, 10 s).
4. **Solve**: SAT completes the repaired reading.

It returns the argmax reading, the repaired reading, the solution, a status (`solved`, `timeout` or `flagged`) and the time. A grid that is not `solved` has no answer: the app should say it could not read the grid and ask for a new photo.

On the 1,000 test grids (CNN, 3 seeds): 99.0% right, 0.97% wrong, no timeouts; 0.07 s per grid on average, 0.28 s at most, on a laptop CPU.

## The classifier in ONNX

`export_onnx.py` writes `models/cnn_seed0.onnx` and checks it against PyTorch (`results/onnx_check.csv`).

- Input `cells`: float32, shape (N, 1, 28, 28), any N (81 for a grid). Same scaling as MNIST: [0, 1], white ink on black, no mean/std normalization.
- Output `logits`: float32, shape (N, 10). Softmax gives the probabilities; the repair uses their logarithms.
- Check on the 10,022 test cells: largest logit difference from PyTorch 1.1 × 10⁻⁵, same argmax on every cell. The file is 1.6 MB, and ONNX Runtime takes 3.5 ms per grid of 81 cells on a laptop CPU.
- The export uses PyTorch's TorchScript exporter. The newer default (`dynamo=True`) relies on `torch.export`, which fails on Python 3.14 with torch 2.10.
- ONNX Runtime runs it on a phone or in a browser (onnxruntime-web), so the classifier needs no Python.

## The solver side

- The repair needs a MaxSAT solver. Here it is PySAT's RC2, a Python wrapper around C++ SAT solvers. The simplest app keeps it on a small Python server: the phone sends the 81 × 10 log-probabilities (or the image), the server returns the solution.
- For a fully offline app, the encoding would have to be ported: 1,539 variables, about 16,000 hard clauses, and one soft clause per cell and class with weight max(1, round(−1000 ln p)). Each cell's weights are shifted by their minimum. Any MaxSAT solver that accepts clauses added between calls can run the uniqueness loop.
- The time budget is wall-clock. On a slow phone, the same grid may time out where the laptop does not.

## What a real photo adds

The models only ever saw synthetic grids: clean MNIST digits pasted into perfect cells. A photo needs, before step 1:

- find the grid's outer square and warp it to 272 × 272 (perspective transform);
- convert to grayscale and invert (dark ink on white paper → white on black), then scale to [0, 1];
- center each digit in its cell the way MNIST does (bounding box to 20 × 20, centered by center of mass), and remove leftover grid lines from the cell borders.

The classifier would also need training on real cells: printed digits (most published sudokus) and handwriting from a phone camera. Errors on photos would no longer be independent across cells (shared lighting, blur, handwriting), so the error rates measured here are optimistic.

## Wrong answers

About 1% of the CNN's answers are wrong while looking certain: the misreading turns the grid into another valid puzzle with a unique solution. No check on the grid alone can detect it. The app can:

- show the reading next to the photo, so the user can check the digits;
- flag answers with a small margin, i.e. those for which another answer's reading is almost as probable (`undetectable.margin`). With the threshold chosen on validation grids (2.65 nats), this caught 8 of the 10 wrong answers on 1,000 test grids, at the cost of 34 right answers (3.4%) flagged. Each check takes up to two more MaxSAT searches (1 s each here).
