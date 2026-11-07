# Dự báo rủi ro vỡ nợ tín dụng – LightGBM, Hồi quy Logistic và Stacked Ensemble

[English](README.md) · **Tiếng Việt**

[![CI](https://github.com/kiezu01/credit-default-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/kiezu01/credit-default-prediction/actions/workflows/ci.yml)

So sánh ba cách chấm điểm tín dụng trên hai bộ dữ liệu, có xử lý mất cân bằng lớp, đánh giá độ hiệu chuẩn (calibration) và giải thích mô hình bằng SHAP. Đây là bài tập nhóm của chương trình Thạc sĩ Hệ thống thông tin, đặt trong khung phân loại phân tích dữ liệu của Lepenioti và cộng sự (2020).

## Phương pháp

| # | Mô hình | Nhóm (Lepenioti 2020) | Thư viện |
|---|---|---|---|
| 1 | Cây tăng cường gradient | Học máy / Khai phá dữ liệu | LightGBM |
| 2 | Hồi quy Logistic | Phân tích thống kê | scikit-learn |
| 3 | Stacked Ensemble (LightGBM + LR → LR meta-learner) | Cải tiến | scikit-learn + imbalanced-learn |

- **Chia train/test 80/20 có phân tầng.** Tập test chỉ dùng để tính chỉ số cuối cùng.
- **SMOTE** cho biến mục tiêu mất cân bằng (tỷ lệ vỡ nợ 7–8%). Đây là cách hiệu chỉnh mất cân bằng duy nhất; các mô hình không tăng trọng số lớp thêm lần nữa.
- **Early stopping cho LightGBM** trên 15% dữ liệu train tách ra làm tập validation, không bao giờ dùng tập test.
- **Stacking out-of-fold 5 fold**, SMOTE chạy *bên trong* từng fold để điểm tổng hợp không lọt sang fold kiểm tra; meta-learner học trên tỷ lệ lớp thật.
- **Chỉ số:** AUC-ROC, AUC-PR (chỉ số chính vì dữ liệu mất cân bằng), Gini, KS, F1 ở ngưỡng 0,5 và Brier score để đo độ hiệu chuẩn.
- **SHAP** summary plot cho các mô hình LightGBM.

## Kết quả

Tập test, 2.000 hồ sơ mỗi bộ dữ liệu:

| Mô hình | Bộ dữ liệu | AUC-ROC | AUC-PR | Gini | KS | F1 | Brier ↓ |
|---|---|---|---|---|---|---|---|
| LightGBM | Home Credit | 0,998 | 0,984 | 0,997 | 0,970 | 0,927 | 0,009 |
| Hồi quy Logistic | Home Credit | **0,999** | **0,990** | **0,998** | **0,982** | 0,921 | 0,010 |
| Stacked Ensemble | Home Credit | **0,999** | **0,990** | **0,998** | 0,980 | **0,938** | **0,008** |
| LightGBM | Give Me Some Credit | 0,851 | 0,382 | 0,702 | 0,555 | **0,345** | 0,054 |
| Hồi quy Logistic | Give Me Some Credit | **0,884** | **0,489** | **0,768** | 0,602 | 0,338 | 0,148 |
| Stacked Ensemble | Give Me Some Credit | **0,884** | 0,462 | 0,767 | **0,604** | 0,291 | **0,048** |

**Nhận xét**

- Trên bộ Give Me Some Credit (khó hơn), Hồi quy Logistic và Stacked Ensemble xếp hạng rủi ro tốt như nhau (AUC 0,884, KS ≈ 0,60). LightGBM kém hơn hai mô hình này.
- Stacked Ensemble hiệu chuẩn tốt nhất với khoảng cách lớn: Brier 0,048 so với 0,148 của LR. LR học trên dữ liệu đã cân bằng bằng SMOTE nên xác suất dự báo cao hơn nhiều so với tỷ lệ vỡ nợ thực 7%. Meta-learner học trên tỷ lệ lớp thật nên sửa được sai lệch này. Độ hiệu chuẩn quan trọng mỗi khi điểm số được dùng để định giá, trích lập dự phòng hay thu hồi nợ, không chỉ để duyệt/từ chối.
- F1 ở ngưỡng 0,5 của Ensemble thấp nhất *chính vì* mô hình đã hiệu chuẩn: khi tỷ lệ nền là 7%, rất ít hồ sơ có P(vỡ nợ) > 0,5. Trong thực tế, ngưỡng nên được chọn theo chi phí kinh doanh của việc duyệt nhầm so với từ chối nhầm.

Toàn bộ số liệu nằm trong [`results/results_summary.json`](results/results_summary.json), hình ảnh trong [`figures/`](figures/):

| Hình | Nội dung |
|---|---|
| `fig1_roc_pr_curves.png` | Đường ROC và precision-recall, mọi mô hình × cả hai bộ dữ liệu |
| `fig2_metrics_comparison.png` | Biểu đồ cột AUC-ROC, AUC-PR, F1 và Brier |
| `fig3a/b_shap_*.png` | SHAP summary plot của LightGBM |
| `fig4_confusion_matrices.png` | Ma trận nhầm lẫn ở ngưỡng 0,5 |
| `fig5_stacking_architecture.png` | Kiến trúc Stacked Ensemble |

> **Về dữ liệu.** Cả hai bộ dữ liệu (mỗi bộ 10.000 dòng) là dữ liệu *mô phỏng*, bám theo cấu trúc và phân phối của bộ Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) và [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit). Dữ liệu kiểu Home Credit gần như tách được tuyến tính, nên điểm gần tuyệt đối (AUC ≈ 0,999) không phản ánh hiệu năng thực tế. Kết quả trên dữ liệu kiểu Give Me Some Credit mới là phép so sánh có ý nghĩa.

### Các lỗi phương pháp đã sửa (tháng 11/2026)

Phiên bản đầu của repo có ba lỗi phương pháp. Các lỗi đã được sửa và bảng kết quả ở trên lấy từ pipeline đã sửa:

1. **Rò rỉ tập test.** Early stopping của LightGBM theo dõi tập test. Giờ nó theo dõi tập validation tách từ dữ liệu train.
2. **Hiệu chỉnh mất cân bằng hai lần.** Dữ liệu đã được SMOTE cân bằng 50/50 *mà* mô hình vẫn tăng trọng số lớp thêm lần nữa (`scale_pos_weight` = 11–13, `class_weight='balanced'`). Trên Give Me Some Credit, điều này khiến loss validation của LightGBM tăng ngay sau cây đầu tiên, nên kết quả LightGBM báo cáo trước đây thực chất là mô hình 1 cây (AUC 0,780, Brier 0,255).
3. **Stacking bị rò dữ liệu.** Ensemble được cross-validate trên dữ liệu sau SMOTE, nên điểm tổng hợp sinh từ các láng giềng ở fold kiểm tra bị lọt giữa các fold.

Tác động trên Give Me Some Credit: LightGBM AUC 0,780 → 0,851, Brier 0,255 → 0,054; Stacked Ensemble AUC 0,847 → 0,884, Brier 0,087 → 0,048. Các báo cáo trong [`docs/`](docs/) mô tả phiên bản ban đầu.

## Cấu trúc repo

```
├── data/
│   ├── dataset1_homecredit.csv     # Kiểu Home Credit, 13 biến + TARGET
│   └── dataset2_givemecredit.csv   # Kiểu Give Me Some Credit, 10 biến + SeriousDlqin2yrs
├── src/credit_default_experiment.py  # Đọc dữ liệu → huấn luyện → đánh giá → vẽ hình
├── tests/test_experiment.py          # Bộ test pytest (~20 giây)
├── figures/                          # Hình được sinh ra
├── results/results_summary.json      # Chỉ số được sinh ra
└── docs/                             # Báo cáo môn học (tiếng Việt, định dạng IEEE)
```

## Cách chạy

Cần Python 3.10+ (đã kiểm thử trên 3.12).

```bash
pip install -r requirements.txt
python src/credit_default_experiment.py               # chạy đầy đủ, dưới 1 phút
python src/credit_default_experiment.py --skip-shap   # bỏ qua hình SHAP
python src/credit_default_experiment.py --data-dir path/to/csvs --out-dir runs/exp1
```

Test và lint:

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check --select F src tests
```

## Công nghệ

Python · LightGBM · scikit-learn · imbalanced-learn · SHAP · pandas · matplotlib · pytest · GitHub Actions

## Tài liệu tham khảo

Lepenioti, K., Bousdekis, A., Apostolou, D., & Mentzas, G. (2020). Prescriptive analytics: Literature review and research challenges. *International Journal of Information Management*, 50, 57–70. https://doi.org/10.1016/j.ijinfomgt.2019.04.003
