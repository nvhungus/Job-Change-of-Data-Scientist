import logging
import sys

# Khai báo phiên bản của package
__version__ = "1.0.0"

# Cấu hình Logging tập trung
logging.basicConfig(
    level = logging.INFO,
    format = "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    stream = sys.stdout
)

logger = logging.getLogger(__name__)
logger.info(f"SRC package phiên bản {__version__} đã được khởi tạo")