import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> np.ndarray:
    # Đọc dữ liệu từ file CSV vào một structured array của NumPy
    try:
        data = np.genfromtxt(
            file_path,
            delimiter=',',
            dtype=None,      # Tự động nhận diện kiểu dữ liệu cho mỗi cột
            names=True,      # Dòng đầu tiên là tên cột
            encoding='utf-8'
        )
        logger.info(f"Tải dữ liệu thành công từ '{file_path}'.")
        return data
    except FileNotFoundError:
        logger.error(f"Lỗi: Không tìm thấy file dữ liệu tại '{file_path}'.")
        return np.array([])


def handle_missing_categorical(data: np.ndarray, columns: list) -> np.ndarray:
    # Xử lý giá trị thiếu cho các cột categorical bằng cách điền 'Unknown'
    processed_data = np.copy(data)
    for col_name in columns:
        column = processed_data[col_name]
        # Giá trị thiếu (đọc từ file là chuỗi rỗng) được thay thế bằng 'Unknown'
        missing_mask = (column.astype(str) == '')
        column[missing_mask] = 'Unknown'
    logger.info("Hoàn tất xử lý giá trị thiếu cho các cột categorical.")
    return processed_data


def encode_ordinal_features(data: np.ndarray) -> tuple[np.ndarray, dict]:
    # Mã hóa các đặc trưng có thứ tự (ordinal) sang dạng số
    processed_data = np.copy(data)
    ordinal_mappings = {}

    # Mã hóa 'relevent_experience'
    processed_data['relevent_experience'] = np.array(
        [1 if 'Has' in val else 0 for val in processed_data['relevent_experience']]
    )
    ordinal_mappings['relevent_experience'] = {'Has relevent experience': 1, 'No relevent experience': 0}

    # Mã hóa 'education_level'
    edu_map = {'Unknown': 0, 'Primary School': 1, 'High School': 2, 'Graduate': 3, 'Masters': 4, 'Phd': 5}
    processed_data['education_level'] = np.array([edu_map.get(val, 0) for val in processed_data['education_level']])
    ordinal_mappings['education_level'] = edu_map
    
    # Mã hóa 'company_size'
    size_map = {'Unknown': 0, '<10': 1, '10-49': 2, '50-99': 3, '100-500': 4, '500-999': 5, '1000-4999': 6, '5000-9999': 7, '10000+': 8}
    processed_data['company_size'] = np.array([size_map.get(val, 0) for val in processed_data['company_size']])
    ordinal_mappings['company_size'] = size_map
    
    logger.info("Hoàn tất mã hóa ordinal.")
    return processed_data, ordinal_mappings


def encode_one_hot_features(data: np.ndarray, columns: list) -> tuple[np.ndarray, dict]:
    # Mã hóa các đặc trưng không có thứ tự (nominal) bằng One-Hot Encoding
    one_hot_matrices = []
    one_hot_mappings = {}
    
    for col_name in columns:
        column = data[col_name]
        unique_vals = np.unique(column)
        
        # Thao tác vector hóa: so sánh mỗi giá trị trong cột với mảng các giá trị duy nhất
        one_hot_matrix = (column[:, np.newaxis] == unique_vals).astype(int)
            
        one_hot_matrices.append(one_hot_matrix)
        one_hot_mappings[col_name] = {'columns': unique_vals.astype(str)}
        logger.info(f"Đã mã hóa '{col_name}' thành {len(unique_vals)} features.")

    final_one_hot_matrix = np.hstack(one_hot_matrices)
    logger.info(f"Hoàn tất mã hóa one-hot. Ma trận cuối cùng có shape: {final_one_hot_matrix.shape}")
    return final_one_hot_matrix, one_hot_mappings


def process_and_scale_numerical(data: np.ndarray) -> tuple[np.ndarray, dict]:
    # Xử lý và chuẩn hóa các cột dữ liệu số bằng Z-score
    processed_data = np.copy(data)
    scaling_params = {}

    # Xử lý cột 'experience' trước khi chuẩn hóa
    exp_col = processed_data['experience'].astype(str)
    exp_col[exp_col == '<1'] = '0'
    exp_col[exp_col == '>20'] = '21'
    exp_numeric = np.array([np.nan if x == '' or x == 'nan' else float(x) for x in exp_col])
    median_exp = np.nanmedian(exp_numeric)
    exp_numeric[np.isnan(exp_numeric)] = median_exp
    processed_data['experience'] = exp_numeric

    # Chuẩn hóa Z-score
    numerical_cols = ['city_development_index', 'training_hours', 'experience']
    for col_name in numerical_cols:
        col_data = processed_data[col_name].astype(float)
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)

        scaling_params[col_name] = {'mean': mean_val, 'std': std_val}

        if std_val > 0:
            processed_data[col_name] = (col_data - mean_val) / std_val
        else:
            processed_data[col_name] = 0

    logger.info("Hoàn tất xử lý và chuẩn hóa các cột số.")
    return processed_data, scaling_params


def create_new_features(data: np.ndarray) -> np.ndarray:
    # Tạo các đặc trưng mới (Feature Engineering)
    logger.info("Bắt đầu tạo đặc trưng mới...")
    
    # 1. Xử lý outliers cho 'training_hours' bằng cách giới hạn ở phân vị thứ 99
    hours_col = data['training_hours'].astype(float)
    percentile_99 = np.percentile(hours_col, 99)
    hours_col[hours_col > percentile_99] = percentile_99
    
    # 2. Tạo đặc trưng 'is_major_city' dựa trên chỉ số phát triển thành phố
    cdi_col = data['city_development_index'].astype(float)
    is_major_city = (cdi_col >= 0.9).astype(int)
    
    # 3. Thêm cột mới vào structured array
    new_dtype = data.dtype.descr + [('is_major_city', '<i4')]
    new_data = np.empty(data.shape, dtype=new_dtype)
    
    for name in data.dtype.names:
        if name == 'training_hours':
            new_data[name] = hours_col # Sử dụng cột đã xử lý outliers
        else:
            new_data[name] = data[name]
    new_data['is_major_city'] = is_major_city
    
    logger.info("Hoàn tất Feature Engineering.")
    return new_data

def handle_outliers_iqr(data: np.ndarray, column_name: str, method: str = 'clip') -> np.ndarray:
    # Xử lý outliers cho một cột số bằng phương pháp IQR
    
    column_data = data[column_name].astype(float)
    
    # Tính toán Q1, Q3 và IQR
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    iqr = q3 - q1
    
    # Xác định ngưỡng trên và ngưỡng dưới
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    
    if method == 'clip':
        # Phương pháp 1: Giới hạn các giá trị ngoại lai (khuyến khích)
        processed_data = np.copy(data)
        clipped_column = np.clip(column_data, lower_bound, upper_bound)
        processed_data[column_name] = clipped_column
        return processed_data
        
    elif method == 'remove':
        # Phương pháp 2: Xóa các hàng chứa giá trị ngoại lai (không khuyến khích vì làm mất dữ liệu)
        outlier_mask = (column_data < lower_bound) | (column_data > upper_bound)
        return data[~outlier_mask]
        
    else:
        logger.warning("Phương pháp không hợp lệ. Trả về dữ liệu gốc.")
        return data