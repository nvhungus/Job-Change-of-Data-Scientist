import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# Cấu hình giao diện chung cho các biểu đồ
sns.set_theme(style = "whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

def plot_numerical_distribution(data: np.ndarray, column_name: str) -> None:
    # Vẽ histogram và box plot để xem phân phối của một cột số
    if column_name not in data.dtype.names:
        logger.error(f"Không tìm thấy cột '{column_name}' trong dữ liệu.")
        return

    fig, axes = plt.subplots(1, 2, figsize = (18, 7))
    fig.suptitle(f'Phân phối của {column_name}', fontsize = 16)

    sns.histplot(data[column_name], kde = True, bins = 30, ax = axes[0])
    axes[0].set_title('Histogram và Biểu đồ mật độ')

    sns.boxplot(x = data[column_name], ax = axes[1])
    axes[1].set_title('Box Plot (Phát hiện ngoại lai)')

    plt.tight_layout(rect = [0, 0.03, 1, 0.95])
    plt.show()

def plot_categorical_distribution(data: np.ndarray, column_name: str) -> None:
    # Vẽ biểu đồ cột để xem phân phối của một cột categorical
    if column_name not in data.dtype.names:
        logger.error(f"Không tìm thấy cột '{column_name}' trong dữ liệu.")
        return

    column_data = data[column_name]
    unique_vals, counts = np.unique(column_data, return_counts=True)
    unique_vals_str = unique_vals.astype(str)
    
    # Sắp xếp dữ liệu dựa trên số lượng giảm dần
    sorted_indices = np.argsort(-counts)
    x_data = unique_vals_str[sorted_indices] # Dùng dữ liệu đã được làm sạch
    y_data = counts[sorted_indices]
    
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=x_data, y=y_data, hue=x_data, palette='viridis', legend=False)

    # Hiển thị số lượng trên mỗi cột
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', fontsize = 11, color = 'black', xytext = (0, 5), textcoords = 'offset points')
    
    plt.title(f'Phân phối của {column_name}')
    plt.ylabel('Số lượng')
    plt.xticks(rotation = 45, ha = 'right')    
    plt.tight_layout()
    plt.show()
    logger.info(f"Đã tạo biểu đồ cột cho '{column_name}'.")

def plot_missing_values(data: np.ndarray) -> None:
    # Vẽ biểu đồ hiển thị tỷ lệ phần trăm giá trị thiếu của mỗi cột
    missing_counts = {}
    total_rows = len(data)

    for col in data.dtype.names:
        num_missing = np.sum(data[col].astype(str) == '')
        if num_missing > 0:
            missing_counts[col] = num_missing

    if not missing_counts:
        logger.info("Không tìm thấy giá trị thiếu trong dữ liệu.")
        return

    cols_with_missing = list(missing_counts.keys())
    percent_missing = [(count / total_rows) * 100 for count in missing_counts.values()]

    plt.figure(figsize = (14, 8))
    sns.barplot(x = cols_with_missing, y = percent_missing)
    plt.title('Tỷ lệ phần trăm giá trị thiếu của mỗi cột')
    plt.ylabel('Tỷ lệ (%)')
    plt.xticks(rotation = 45, ha = 'right')
    plt.tight_layout()
    plt.show()

def plot_pie_chart(data: np.ndarray, column_name: str, title: str = None) -> None:
    # Vẽ biểu đồ tròn để thể hiện tỷ lệ của các danh mục
    if column_name not in data.dtype.names:
        logger.error(f"Không tìm thấy cột '{column_name}' trong dữ liệu.")
        return
    
    unique_vals, counts = np.unique(data[column_name], return_counts = True)
    
    plt.figure(figsize = (8, 8))
    plt.pie(counts, labels = unique_vals, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette('pastel'))
    
    final_title = title if title else f'Tỷ lệ của {column_name}'
    plt.title(final_title)
    plt.ylabel('') # Xóa nhãn trục y
    plt.show()

def plot_correlation_heatmap(data: np.ndarray, numerical_cols: list) -> None:
    # Vẽ heatmap biểu diễn ma trận tương quan giữa các cột số
    numerical_data = np.array([data[col] for col in numerical_cols]).T.astype(float)
    corr_matrix = np.corrcoef(numerical_data, rowvar = False)
    
    plt.figure(figsize = (10, 8))
    sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', xticklabels = numerical_cols, yticklabels=numerical_cols)
    plt.title('Heatmap tương quan của các đặc trưng số')
    plt.show()

def plot_trend_by_category(data: np.ndarray, category_col: str, target_col: str, title: str) -> None:
    # Vẽ biểu đồ đường thể hiện xu hướng của một biến mục tiêu theo từng danh mục
    categories = np.unique(data[category_col])
    rates = []
    
    # Sắp xếp các giá trị của 'experience' theo thứ tự logic
    if category_col == 'experience':
        cat_numeric = [int(str(c).replace('<', '0').replace('>', '21')) for c in categories]
        sorted_indices = np.argsort(cat_numeric)
        categories = categories[sorted_indices]

    for cat in categories:
        targets_for_cat = data[target_col][data[category_col] == cat]
        # Tỷ lệ của nhãn 1 chính là giá trị trung bình của mảng 0 và 1
        rate = np.mean(targets_for_cat)
        rates.append(rate)
        
    plt.figure(figsize = (14, 7))
    sns.pointplot(x = categories, y = rates, linestyles = "--", markers = "o", color = 'b')
    plt.title(title)
    plt.xlabel(category_col)
    plt.ylabel(f'Tỷ lệ thay đổi công việc (Giá trị trung bình của {target_col})')
    plt.xticks(rotation = 45, ha = 'right')
    plt.grid(True)
    plt.show()

def plot_distribution_by_category(data: np.ndarray, num_col: str, cat_col: str, target_col: str) -> None:
    # Vẽ violin plot so sánh phân phối của biến số theo từng danh mục và biến mục tiêu
    plt.figure(figsize = (16, 8))
    # Trích xuất dữ liệu ra trước để tương thích với Seaborn
    x_data = data[cat_col]
    y_data = data[num_col]
    hue_data = data[target_col]
    sns.violinplot(x = x_data, y = y_data, hue = hue_data, split = True, palette = {0: "lightgrey", 1: "skyblue"}, inner="quart")
    plt.title(f'Phân phối của {num_col} theo {cat_col} và Trạng thái thay đổi công việc')
    plt.xticks(rotation = 45, ha = 'right')
    plt.show()
    
def plot_numerical_vs_categorical(data: np.ndarray, num_col: str, cat_col: str) -> None:
    # Vẽ box plot so sánh phân phối của biến số theo từng danh mục
    plt.figure(figsize = (16, 8))
    # Trích xuất dữ liệu ra trước để tương thích với Seaborn
    x_data = data[cat_col]
    y_data = data[num_col]
    sns.boxplot(x = x_data, y = y_data, hue = x_data, palette = "coolwarm", legend = False)
    plt.title(f'Phân phối của {num_col} theo {cat_col}')
    plt.xticks(rotation = 45, ha = 'right')
    plt.show()

def plot_categorical_vs_categorical(data: np.ndarray, cat_col_1: str, cat_col_2: str) -> None:
    # Vẽ count plot thể hiện mối quan hệ giữa hai biến phân loại
    plt.figure(figsize = (16, 8))
    # Trích xuất dữ liệu ra trước để tương thích với Seaborn
    x_data = data[cat_col_1]
    hue_data = data[cat_col_2]
    sns.countplot(x = x_data, hue = hue_data, palette = "viridis")
    plt.title(f'Mối quan hệ giữa {cat_col_1} và {cat_col_2}')
    plt.xticks(rotation = 45, ha = 'right')
    plt.legend(title = cat_col_2)
    plt.show()

def plot_scatter_numerical(data: np.ndarray, num_col_1: str, num_col_2: str) -> None:
    # Vẽ scatter plot thể hiện mối quan hệ giữa hai biến số
    plt.figure(figsize = (10, 7))
    # Trích xuất dữ liệu ra trước để tương thích với Seaborn
    x_data = data[num_col_1]
    y_data = data[num_col_2]
    sns.scatterplot(x = x_data, y = y_data, alpha = 0.5)
    plt.title(f'Biểu đồ Scatter: {num_col_1} vs {num_col_2}')
    plt.grid(True)
    plt.show()

def _roc_curve_points(y_true, y_probas):
    # Hàm hỗ trợ tính các điểm (FPR, TPR) cho đường cong ROC
    thresholds = np.sort(np.unique(y_probas))[::-1]
    tprs = np.zeros(len(thresholds) + 1)
    fprs = np.zeros(len(thresholds) + 1)

    for i, thresh in enumerate(thresholds):
        y_pred = (y_probas >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tprs[i+1] = tp / (tp + fn) if (tp + fn) > 0 else 0
        fprs[i+1] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
    return fprs, tprs

def plot_roc_curve(model, X, y) -> float:
    # Vẽ đường cong ROC và tính/trả về điểm AUC
    y_probas = model.predict_proba(X)
    fprs, tprs = _roc_curve_points(y, y_probas)
    
    # Tính AUC bằng quy tắc hình thang (trapezoidal rule)
    auc_score = np.trapz(tprs, fprs)
    
    plt.figure(figsize = (8, 8))
    plt.plot(fprs, tprs, color = 'darkorange', lw = 2, label = f'Đường cong ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--')
    plt.xlabel('Tỷ lệ dương tính giả (False Positive Rate)')
    plt.ylabel('Tỷ lệ dương tính thật (True Positive Rate)')
    plt.title('Đường cong Receiver Operating Characteristic (ROC)')
    plt.legend(loc = "lower right")
    plt.show()
    return auc_score

def plot_probability_distribution(model, X):
    # Vẽ biểu đồ phân phối xác suất dự đoán của mô hình
    y_probas = model.predict_proba(X)
    plt.figure(figsize = (10, 6))
    sns.histplot(y_probas, kde = True, bins = 50)
    plt.title('Phân phối xác suất dự đoán của mô hình')
    plt.xlabel('Xác suất dự đoán')
    plt.ylabel('Tần suất')
    plt.show()