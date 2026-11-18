# GPVS-Faults literature: how the train/test splits were made

Dataset: GPVS-Faults, Bakdi, Guichi, Mekhilef, Bounoua, Mendeley Data 2020, doi:10.17632/n76t439f65.1.
Compiled 2026-11-17. Every quote below comes from the paper's own text (PDF or PMC full text), not from a summary.
"Not stated" means I searched the full text and the paper does not describe the split.

## Summary table

| # | Paper | Model | Headline metric | Split (as described) | Shuffled? | Per-class recall? |
|---|-------|-------|-----------------|----------------------|-----------|-------------------|
| 1 | Bakdi, Bounoua, Guichi, Mekhilef, *IJEPES* 125:106457 (2021), [doi:10.1016/j.ijepes.2020.106457](https://doi.org/10.1016/j.ijepes.2020.106457) (accepted manuscript: [hdl:10852/84159](https://hdl.handle.net/10852/84159)) | Online PCA + KDE + KL divergence, semi-supervised detection | <1% false alarms; detected/not-detected per fault (Table 2). No accuracy reported | Time-ordered. Trained on the first 3.3 s of normal operation, plus separate fault-free validation runs; tested on whole 10–15 s experiments | **No** | Binary detected/not per fault; detection rate is explicitly *not* computed |
| 2 | Wali & Khan, "Explainable Signature-based ML Approach for Identification of Faults in Grid-Connected PV Systems", [arXiv:2112.14842](https://arxiv.org/abs/2112.14842) (2021) | Random Forest on statistics of 200-sample batches (mean/std/max/min); KNN and SVM baselines | 100% accuracy (MPPT files, 8 classes) | "split into a train-test ratio of 70:30". How the rows were assigned is not stated | **Not stated** (likely random, see note) | **Yes**: per-class accuracy, precision, recall and F1 (Table II) |
| 3 | Devi & Srivenkatesh, "CNNs for Fault Detection in Grid-Connected PV Panels", *Ingénierie des Systèmes d'Information* 28(6):1619–1625 (2023), [doi:10.18280/isi.280619](https://doi.org/10.18280/isi.280619) | "Deep sequential model" (text mixes CNN/LSTM/MLP); RF, KNN, SVM and ensemble baselines | 99.947% accuracy | "Data is split into train data, test data, and validation data." No ratio, no method | **Not stated** | No. Confusion-matrix figure only; Table 1 reports aggregate metrics |
| 4 | Teta et al., "FDD of grid-connected PV systems using energy valley optimizer based lightweight CNN and wavelet transform", *Sci. Rep.* 14:18907 (2024), [doi:10.1038/s41598-024-69890-7](https://doi.org/10.1038/s41598-024-69890-7) | EVO-tuned lightweight 2D-CNN on CWT scalograms of non-overlapping windows (MPPT only) | 100% acc/F1 (balanced, 8-class); 99.86% (unbalanced); 93.75% (with noise) | "70% for training and the remaining 30% equally split between validation and testing, with 720 images each." Assignment method not stated | **Not stated** | Confusion matrices (Figs. 13–15); tables report aggregate metrics only |
| 5 | Zhang, Jing, Jiao, Yao, "Advanced Multi-Scale CNN-BiLSTM for Robust PV Fault Detection", *Sensors* 25(14):4474 (2025), [doi:10.3390/s25144474](https://doi.org/10.3390/s25144474) | Multi-scale CNN-BiLSTM with attention; CNN, BiLSTM, RF, SVM and XGBoost baselines | 100% acc ("Medium"); 95.7% ("Easy"); 83.3% ("Hard") | Overlapping windows (60 steps, stride 2), *then* a stratified 60/20/20 split of the windows | **Yes (clearly implied)**: stratified sampling of overlapping windows | No. Table 1 is aggregate only; no confusion matrix in the text |
| 6 | Jorry, Duma, Reinikainen, Haario, Roininen, "An unsupervised kernel norm monitoring for fault detection in a time series PV system", [arXiv:2607.01541](https://arxiv.org/abs/2607.01541) (2026) | Kernel-norm monitoring (KNM) on FFT window features; OCSVM, iForest and LOF baselines | 99.1% (F2L) and 98.3% (F4L) detection accuracy, IPPT only, 2 faults | Time-ordered. The baseline is the first windows of normal operation; later windows are monitored | **No** | Binary TPR/FPR per fault scenario (detection only, no diagnosis) |

Not included: Harrou et al., *Energy Convers. Manage.* (2024), doi:10.1016/j.enconman.2024.118665 (VAE, reported 92.90% MPPT / 92.99% IPPT). The full text is paywalled and returned 403, so its split is **unverified**. Several other 2025–26 GPVS papers (IJEPES 2025 hybrid CNN-AE, Energies 2026 fuzzy kNN, Energy Reports 2026 GAN augmentation, Int. J. Energy Res. 2026 few-shot) could not be fetched because of bot walls, so they are also **unverified**.

## Quotes on the split, per paper

### 1. Bakdi et al. 2021 (dataset authors)
> "The reference training data is collected for both MPPT and IPPT modes where each set spans an initial operation period of 3.3 seconds (… = 3.3 × 10^4) … during normal operation conditions while test experiments are of 10 to 15 seconds (… 100 000×7 to … 150 000×7)."

> "Validation fault-free data are collected separately from training data without condition variations and used in the validation stage to tune the control limits."

> "Unfortunately, the detection rate and detection delay cannot be assessed precisely without a reference method. Table 2 lists the highest false alarms rate, successful detection of the seven GPV system faults, and average offline computational time…"

### 2. Wali & Khan 2021
> "all thirteen signals were divided into batches of 200 consecutive samples … each individual fault sample was labeled at this stage and the resulting dataset of each CSV was concatenated to generate a final dataset"

> "The GPVS signature dataset is first normalized at the data transformation stage, then it is split into a train-test ratio of 70:30."

> "the randomized search was utilized with 5-fold cross-validation" (this was for hyperparameter tuning)

Note: the paper does not say the split was shuffled. The frame is a concatenation ordered by class, and Table II reports test results for every class. A plain sequential 70:30 cut would have left some classes out of the test set entirely, so the split must have been random or done per class. Random is likely (scikit-learn is used and shuffles by default), but the paper does not confirm it. Normalization was also fitted before the split.

### 3. Devi & Srivenkatesh 2023
> "Data is split into train data, test data, and validation data [17]."

No windowing is described; the model appears to work on individual rows. Table 1 also reports a standard deviation across "subsets", which suggests cross-validation, but the fold scheme is not stated.

### 4. Teta et al. 2024
> "A non-overlapping sliding window is employed to retain temporal information and optimize computational efficiency."

> "The dataset comprises a total of 4800 images, which are allocated as follows: 70% for training and the remaining 30% equally split between validation and testing, with 720 images each."

> "It is important to note that the proposed diagnostic approach is validated using the dataset under MPPT mode."

The paper never says how images were assigned to train, validation and test (random, by time, or by experiment). Its own numbers also disagree: Table 10 lists 510 train+validation and 90 test images per class, which is about 85/15, not 70/15/15. The window width W is never given numerically (Table 8's (100, 13) input suggests W = 100). The binary task adds time-shift augmentation of the healthy class, and the paper does not say whether that happened before or after the split.

### 5. Zhang et al. 2025
> "The continuous time-series data was segmented into sequences of 60 time steps with a stride of 2, which significantly increased the number of training instances available to the model"

> "Subsequently, the time-series data is transformed into overlapping sequences, as previously described. Finally, the sequential data is split using a stratified sampling method into training (60%), validation (20%), and testing (20%) sets, ensuring that the class distribution is maintained across all subsets"

> "containing 4272 carefully curated samples distributed across 16 distinct fault categories"

Adjacent windows share 58 of their 60 rows, and the windows are split by stratified sampling. A near-copy of almost every test window is therefore in the training set. Min-max scaling was also applied before windowing and splitting.

### 6. Jorry et al. 2026
> "The first n_init windows is used as a baseline (e.g., first 6 s/Ts) for threshold computation."

> "They are trained using only fault-free data and are applied to cluster the test data into normal observations or anomalies."

> "these results are obtained without any labeled fault data at any stage of training or parameter selection, relying exclusively on normal operation data from the initial baseline window."

## Side note: sampling rate
The Mendeley page says "average sampling T_s=9.9989 μs" (about 100 kHz), and Jorry et al. repeat about 100 kHz. Bakdi et al. state Ts = 100 µs and give sample counts that match it: 3.3 s = 3.3×10^4 samples, and 10–15 s = 100k–150k samples. Wali & Khan likewise say 200 samples equal one cycle. The sample counts support **about 10 kHz**, so the "μs" on the Mendeley page looks like a typo. Check the `Time` column of the CSVs to settle it.

## Conclusion
1. The very high diagnosis accuracies (99.9–100%: Wali & Khan, Devi & Srivenkatesh, Teta et al., Zhang et al.) all come from a single pooled random-style split of windows/rows. Only Zhang et al. states it clearly (stratified split of stride-2 overlapping windows, which is leaky). The other three **do not state** how they split, so they cannot be confirmed as shuffled.
2. None of the six papers holds out whole experiments, and none tests cross-mode (MPPT↔IPPT). The only time-ordered evaluations (Bakdi 2021; Jorry 2026) are detection-only, report detection-style metrics, and do not claim 8-class diagnosis accuracy.
3. Verdict: the near-perfect diagnosis numbers are **consistent with, but mostly not proven to come from**, leaky within-recording splits. Per-class recall is reported only by Wali & Khan, so most headline numbers hide per-class failures.
