==============================================================
  ZK-SNARK BENCHMARK  (10 samples)
==============================================================

── Sample 1/10 ────────────────────────────────────────
   FP32 prob   : 0.9963 → Disease  (0.742 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.246s)
   Circuit out : 9757924843555150431866349988235210396498134346161559635078454249265299456.0000  |  Δ=9757924843555150431866349988235210396498134346161559635078454249265299456.000000  |  label match=True
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

── Sample 2/10 ────────────────────────────────────────
   FP32 prob   : 0.0768 → No Disease  (0.314 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.189s)
   Circuit out : 3788561292301324371090358175727671851264092376700316238602417282324889600.0000  |  Δ=3788561292301324371090358175727671851264092376700316238602417282324889600.000000  |  label match=False
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

── Sample 3/10 ────────────────────────────────────────
   FP32 prob   : 0.7219 → Disease  (0.298 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.222s)
   Circuit out : -1286594747896449268976118080373599583786752786554101449820375139869523968.0000  |  Δ=1286594747896449268976118080373599583786752786554101449820375139869523968.000000  |  label match=False
   [ZKP] Generating proof …
[worker] START  verb=prove
[worker] SUCCESS verb=prove
   ✓  Proof    (2.340s,  20.05 KB)
   [ZKP] Verifying proof …
[worker] START  verb=verify
   ✓  VALID  (0.3406s)
   Overhead    : 8605x  vs plain inference

── Sample 4/10 ────────────────────────────────────────
   FP32 prob   : 0.9090 → Disease  (0.359 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.351s)
   Circuit out : 1220962064661918175020879779260945033947678996608017446524220300966494208.0000  |  Δ=1220962064661918175020879779260945033947678996608017446524220300966494208.000000  |  label match=True
   [ZKP] Generating proof …
[worker] START  verb=prove
[worker] SUCCESS verb=prove
   ✓  Proof    (2.638s,  20.06 KB)
   [ZKP] Verifying proof …
[worker] START  verb=verify
   ✓  VALID  (0.2965s)
   Overhead    : 8337x  vs plain inference

── Sample 5/10 ────────────────────────────────────────
   FP32 prob   : 0.1628 → No Disease  (0.378 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.321s)
   Circuit out : 310728172238891788538455921793696026897795877382688512740399650300755968.0000  |  Δ=310728172238891788538455921793696026897795877382688512740399650300755968.000000  |  label match=False
   [ZKP] Generating proof …
[worker] START  verb=prove
[worker] SUCCESS verb=prove
   ✓  Proof    (2.615s,  20.00 KB)
   [ZKP] Verifying proof …
[worker] START  verb=verify
   ✓  VALID  (0.2729s)
   Overhead    : 7777x  vs plain inference

── Sample 6/10 ────────────────────────────────────────
   FP32 prob   : 0.9325 → Disease  (0.473 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.330s)
   Circuit out : 9205353776665230918634854847011656046211267434077168061505754653597368320.0000  |  Δ=9205353776665230918634854847011656046211267434077168061505754653597368320.000000  |  label match=True
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

── Sample 7/10 ────────────────────────────────────────
   FP32 prob   : 0.1099 → No Disease  (0.378 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.190s)
   Circuit out : 4616986533489529230700886215321611062603614550516142949801769018062798848.0000  |  Δ=4616986533489529230700886215321611062603614550516142949801769018062798848.000000  |  label match=False
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

── Sample 8/10 ────────────────────────────────────────
   FP32 prob   : 0.6397 → Disease  (0.350 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.209s)
   Circuit out : 3958085436944367594119224366594851288939922740829251358363596783571959808.0000  |  Δ=3958085436944367594119224366594851288939922740829251358363596783571959808.000000  |  label match=True
   [ZKP] Generating proof …
[worker] START  verb=prove
[worker] SUCCESS verb=prove
   ✓  Proof    (2.287s,  20.02 KB)
   [ZKP] Verifying proof …
[worker] START  verb=verify
   ✓  VALID  (0.2585s)
   Overhead    : 7140x  vs plain inference

── Sample 9/10 ────────────────────────────────────────
   FP32 prob   : 0.0819 → No Disease  (0.430 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.362s)
   Circuit out : 6107548064822953803668436145452752403287664991349566121137025421521453056.0000  |  Δ=6107548064822953803668436145452752403287664991349566121137025421521453056.000000  |  label match=False
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

── Sample 10/10 ────────────────────────────────────────
   FP32 prob   : 0.9711 → Disease  (0.306 ms)
   [ZKP] Generating witness …
[worker] START  verb=witness
[worker] SUCCESS verb=witness
   ✓  Witness  (0.250s)
   Circuit out : 1055751511485619054359160310807688737179630575586688818360017377359298560.0000  |  Δ=1055751511485619054359160310807688737179630575586688818360017377359298560.000000  |  label match=True
   [ZKP] Generating proof …
   [!] Proof generation failed or timed out (Likely a Field Overflow). Skipping.

==============================================================
  RESULTS SUMMARY
==============================================================

  [ ZKP TIMING ]
  Avg Witness Time     : 0.2759 s
  Avg Proof Time       : 2.4702 s  (±0.1820)
  Avg Verify Time      : 0.2921 s
  Avg Total ZKP Time   : 3.0383 s

  [ PROOF SIZE ]
  Avg Proof Size       : 20.03 KB
  Std Dev of Size      : 0.027538 KB
  Constant Size?       : NO — investigate

  [ QUANTIZATION FIDELITY ]
  Avg Quant Error      : 1694092605435406718923633863932883850259813817545988160812060946146000896.00000000
  Max Quant Error      : 3958085436944367594119224366594851288939922740829251358363596783571959808.00000000
  Label Match Rate     : 50.0%

  [ PRIVACY OVERHEAD ]
  Avg FP32 Latency     : 0.3459 ms
  Avg Overhead Ratio   : 7965x  vs plain inference

  [ VALIDITY ]
  Proof Validity Rate  : 100.0%

  Results → C:\Users\Swayam Bansal\.vscode\PBL Project\privacy_ml_zkp\benchmark_results.csv
  Summary → C:\Users\Swayam Bansal\.vscode\PBL Project\privacy_ml_zkp\benchmark_summary.json
==============================================================


Train Results

[1/5] Loading & preprocessing data …
      Dataset shape: (920, 13)  |  Positive rate: 55.33%
      Features (13): ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
[2/5] Building model …
[3/5] Training …
  Epoch  25/150 | Loss: 0.3701 | Acc: 0.8315 | F1: 0.8531
  Epoch  50/150 | Loss: 0.3270 | Acc: 0.8152 | F1: 0.8411
  Epoch  75/150 | Loss: 0.2858 | Acc: 0.8152 | F1: 0.8396
  Epoch 100/150 | Loss: 0.2379 | Acc: 0.8315 | F1: 0.8531
  Epoch 125/150 | Loss: 0.2017 | Acc: 0.8261 | F1: 0.8505
  Epoch 150/150 | Loss: 0.1723 | Acc: 0.8207 | F1: 0.8451
[4/5] Final evaluation on test set …
               precision    recall  f1-score   support

   No Disease       0.84      0.77      0.80        82
Heart Disease       0.83      0.88      0.85       102

     accuracy                           0.83       184
    macro avg       0.83      0.83      0.83       184
 weighted avg       0.83      0.83      0.83       184