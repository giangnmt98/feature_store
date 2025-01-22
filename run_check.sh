đi #!/bin/bash

# Constants
MAX_LINES=500
MAX_CHANGE_LINES=200
PYTHON_EXEC=python3
CODE_DIRECTORY=featurestore

# Functions
echo_error() {
  local RED='\033[0;31m'
  local NC='\033[0m' # No Color
  echo -e "${RED}$1${NC}"
}

echo_separator() {
  echo "================================"
}

# Start of script
echo_separator
echo "Running checks before commit"
echo ""
echo_separator
echo "Running code style checking with Flake8"
$PYTHON_EXEC -m flake8 ./$CODE_DIRECTORY
echo ""
echo_separator
echo "Running type checking with MyPy"
$PYTHON_EXEC -m mypy ./$CODE_DIRECTORY
echo ""
echo_separator
echo "Running docstrings checking Pylint"
$PYTHON_EXEC -m pylint ./$CODE_DIRECTORY
echo ""

# File line count check
check_file_line_count() {
  error_files=""  # Chuỗi để lưu danh sách các file vi phạm

  # Tìm tất cả các file trong thư mục CODE_DIRECTORY (bỏ qua thư mục con, nếu cần thêm điều kiện)
  for file in $(find "$CODE_DIRECTORY" -type f); do
    if echo "$file" | grep -vq ".pylintrc"; then  # Bỏ qua file .pylintrc
      line_count=$(wc -l < "$file")  # Đếm số dòng trong file
      if [ "$line_count" -gt "$MAX_LINES" ]; then
        error_files="${error_files}\n$file: $line_count lines"  # Thêm file vi phạm vào danh sách chuỗi lỗi
      fi
    fi
  done

  # Kiểm tra mảng lỗi và in ra nếu có bất kỳ file nào vi phạm
  if [ -n "$error_files" ]; then
    echo_error "The following files exceed the $MAX_LINES lines threshold:"
    echo -e "$error_files"  # In danh sách file vi phạm
    exit 1  # Thoát với lỗi
  fi
}

# Change line count check
check_change_line_count() {
  local changes=$(git diff main --numstat | awk '{added+=$1; deleted+=$2} END {print added+deleted}')
  if [ "$changes" -gt "$MAX_CHANGE_LINES" ]; then
    echo_error "Too many changes: $changes lines. Maximum allowed: $MAX_CHANGE_LINES lines."
    exit 1  # Exit if changes exceed the limit
  else
    echo "Number of changed lines: $changes"
  fi
  echo_separator
  echo "Change line check completed"
}

# Execution
check_file_line_count
echo_separator
echo "Line count check completed"
echo ""

IS_CHECK_CHANGE_LINE=True
while [ "$#" -gt 0 ]; do
  case $1 in
    check_change_line) IS_CHECK_CHANGE_LINE=true ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

[ "$IS_CHECK_CHANGE_LINE" = true ] && check_change_line_count
