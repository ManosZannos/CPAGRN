# CPA-GRN: CPA-Aware Graph Recurrent Network for Vessel Trajectory Prediction

Κώδικας για τη διπλωματική εργασία **"CPA-GRN: Πρόβλεψη Τροχιάς Πλοίων με Ενσωμάτωση CPA σε Μηχανισμό Αραιής Προσοχής Γράφου"**.

Το CPA-GRN είναι ένα μοντέλο πρόβλεψης τροχιάς πολλαπλών πλοίων που ενσωματώνει τα μεγέθη **TCPA/DCPA** (Time/Distance to Closest Point of Approach) ως διαφορίσιμα χαρακτηριστικά ακμής μέσα σε μηχανισμό αραιής προσοχής γράφου, ενσωματωμένα σε κάθε χρονικό βήμα ενός GRU-based αναδρομικού κωδικοποιητή.

## Μοντέλα

| Μοντέλο | Περιγραφή |
|---|---|
| **LSTM** | Baseline χωρίς κανέναν μηχανισμό αλληλεπίδρασης μεταξύ πλοίων |
| **No-CPA v4** | Ablation: ίδιος μηχανισμός προσοχής γράφου, αλλά με αμιγώς γεωμετρικά χαρακτηριστικά ακμής (απόσταση, bearing, σχετική πορεία) — χωρίς TCPA/DCPA |
| **CPA-GRN v4** | Πλήρες προτεινόμενο μοντέλο, με TCPA/DCPA ενσωματωμένα per-step |
| **SMCHN** | Αναπαραγωγή του Sparse Multi-graph Convolutional Hybrid Network [Wang et al., 2023] ως εξωτερικό σημείο αναφοράς |

## Dataset

- **Πηγή:** NOAA AIS δεδομένα, Δεκέμβριος 2021, περιοχή San Diego
- **Προεπεξεργασία:** `preprocess_dec2021_merged.py` — συνενώνει τα 31 ημερήσια αρχεία σε συνεχή μηνιαία τροχιά ανά πλοίο, επαναδειγματοληψία σε σταθερό βήμα 1 λεπτού με γραμμική παρεμβολή (έως 5 λεπτά κενού), διάσπαση τροχιάς σε κενά >10 λεπτών
- **Διαχωρισμός:** χρονολογικός (19/6/6 ημέρες train/val/test), για αποφυγή χρονικής διαρροής πληροφορίας

## Δομή Αρχείων

```
├── preprocess_dec2021_merged.py   # προεπεξεργασία AIS δεδομένων (NOAA → clean trajectories)
├── dataset.py                     # PyTorch Dataset classes, sliding-window sampling
│
├── model_lstm.py                  # LSTM baseline
├── model_cpagrn.py                # CPA-GRN v4 (EDGE_DIM=7: dist, bearing, dhdg, TCPA, DCPA, ...)
├── model_cpagrn_nocpa.py          # No-CPA v4 ablation (EDGE_DIM=5, χωρίς TCPA/DCPA)
│
├── train_lstm.py                  # εκπαίδευση LSTM (--seed για αναπαραγωγιμότητα)
├── train_cpagrn.py                # εκπαίδευση CPA-GRN v4 / No-CPA v4
├── evaluate_lstm.py               # αξιολόγηση LSTM (ADE/FDE ανά χρονικό βήμα)
├── evaluate_cpagrn.py             # αξιολόγηση CPA-GRN v4 / No-CPA v4
├── evaluate_cpagrn.py             # αξιολόγηση CPA-GRN v4 / No-CPA v4
│
├── generate_thesis_figures.py     # παράγει Εικόνα 3.2, Διαγράμματα 4.2-4.4
├── plot_trajectories.py           # εντοπίζει σενάρια σύγκλισης και παράγει το Διάγραμμα 4.1
```
## Εγκατάσταση

```bash
conda create -n trajpred python=3.10
conda activate trajpred
pip install -r requirements.txt
```

## Χρήση

**Προεπεξεργασία δεδομένων:**
```bash
python preprocess_dec2021_merged.py
```

**Εκπαίδευση:**
```bash
python train_cpagrn.py --obs_len 10 --pred_len 10 --d_model 64 \
    --epochs 200 --lr 0.001 --batch_size 16 \
    --tag CPAGRN_obs10_pred10_s42 --seed 42 --gpu_num 0
```

**Αξιολόγηση:**
```bash
python evaluate_cpagrn.py --tag CPAGRN_obs10_pred10_s42 \
    --obs_len 10 --pred_len 10 --split test --gpu_num 0
```

Το `--tag` καθορίζει το checkpoint directory (`checkpoints/<tag>/val_best.pth`) και πρέπει να είναι μοναδικό ανά συνδυασμό `obs_len`/`pred_len`/`seed`.

## Παραγωγή Εικόνων/Διαγραμμάτων

```bash
# Εικόνα 2.2, Διαγράμματα 4.2-4.4 (βάσει ήδη υπολογισμένων αποτελεσμάτων)
python generate_thesis_figures.py

# Διάγραμμα 4.1 (σενάριο σύγκλισης, απαιτεί ήδη εκπαιδευμένα checkpoints)
python plot_trajectories.py --obs_len 10 --pred_len 10 --seed 42 --gpu_num 0
```

Οι εικόνες αποθηκεύονται στον φάκελο `figures/` (`plot_trajectories.py`) και `thesis_figures/` (`generate_thesis_figures.py`). Το `plot_trajectories.py` σαρώνει το test set και προτείνει πολλαπλά υποψήφια σενάρια σύγκλισης· το τελικά επιλεγμένο για τη διπλωματία (Διάγραμμα 4.1) πρέπει να επιλεγεί χειροκίνητα από τα παραγόμενα candidates βάσει του πίνακα εξόδου του script.

## Κύρια Αποτελέσματα (ADE σε μοίρες, test set)

| Ορίζοντας | LSTM | No-CPA v4 | CPA-GRN v4 | SMCHN |
|---|---|---|---|---|
| 5 λεπτά | 0.000823 | 0.001807 | 0.000818 | 0.001617 |
| 10 λεπτά | 0.001486 | 0.001185 | 0.001137 | 0.003764 |
| 20 λεπτά | 0.001775 | 0.001777 | 0.001890 | 0.002206 |
| 30 λεπτά | 0.002504 | 0.002668 | 0.002527 | 0.003691 |

Πλήρη αποτελέσματα, ablation study, και ανάλυση ευρημάτων στη διπλωματική εργασία.

## Αναπαραγωγιμότητα

Όλα τα training scripts δέχονται `--seed` για έλεγχο τυχαιότητας. Το LSTM baseline έχει επαληθευτεί σε 3 seeds (42, 123, 456)· το CPA-GRN v4 σε multi-seed επαλήθευση στον κύριο ορίζοντα (10 λεπτά).

## Αναφορά

Αν χρησιμοποιήσετε αυτόν τον κώδικα, παρακαλώ αναφέρατε:

> Εμμανουήλ-Λουκάς Ζάννος , "CPA-GRN: Πρόβλεψη Τροχιάς Πλοίων με Ενσωμάτωση CPA σε Μηχανισμό Αραιής Προσοχής Γράφου", Διπλωματική Εργασία, Τμήμα Μηχανικών Η/Υ και Πληροφορικής, 2026.

## Άδεια Χρήσης

MIT License

Copyright (c) 2026 Manos Zannos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
