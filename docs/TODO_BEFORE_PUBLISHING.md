# Before you publish

1. Copy the code from the original project folder into the repo root:
   ```bash
   cp ~/Downloads/swapdetect/{detectors,evaluate_cv,summarize_seeds,test_detectors,test_real_pipeline}.py .
   ```
   Also copy any other `.py` files those import.
2. Copy the final CSVs and regenerate figures:
   ```bash
   cp ~/Downloads/swapdetect/results_final/results_cv_*_s[012].csv results_final/
   python make_results.py
   ```
3. Replace `requirements.txt` with the output of `pip freeze` from `.venv311`.
4. Fill in: dataset name, source and licence; voice model; how a swap is simulated; how the quality score is defined; CUSUM parameters and fusion model type. Add them to `docs/EXPERIMENT.md`.
5. Put your name in `LICENSE` and add author and course details to the README.
6. Check the dataset licence allows publishing anything derived from it. Do not commit raw recordings.
7. Do not commit large caches (`.gitignore` already excludes them).
8. Enable GitHub Pages on the `/docs` folder to host `docs/index.html`.
