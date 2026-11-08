# Setup TrafficLLM Audit

# Pytorch
https://docs.pytorch.org/get-started/locally/

# Gdown
pip install gdown
gdown "https://drive.google.com/uc?id=1-MVfxyHdQeUguBmYrIIw1jhMVSqxXQgO" -O data/raw/trafficllm_reference/trafficllm_data.zip

# Requirements
pip install -r requirements.txt

# Chia sẻ artifact (chủ repo làm một lần, người khác không cần)
# 1. GitHub > Releases > "Draft a new release", tag: p3-artifacts
# 2. Upload đúng 2 file, giữ nguyên tên (mỗi file < 2 GB):
#      outputs/checkpoints/ustc_gpt2_seed42.pt   (475 MB)
#      outputs/features/ustc_seed42.npz          (575 MB)
# 3. Publish. Link tải có dạng:
#      https://github.com/k1w106/TrafficLLM/releases/download/p3-artifacts/<tên file>
# Không dùng Git LFS (free 1 GB, pull một lần là hết quota).
# Kiểm tra toàn vẹn bằng sha256 ghi trong README.md.

# Kiểm tra nhanh sau khi clone
PYTHONPATH=src pytest -q
PYTHONPATH=src python3 -c "import trafficllm.openset.scoring, trafficllm.evaluation.metrics; print('import ok')"