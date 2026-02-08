"""
Phân tích tần suất cập nhật orderbook từ dữ liệu LOBSTER

Script này sẽ tính toán:
- Khoảng thời gian trung bình giữa các messages
- Tần suất cập nhật orderbook (messages/giây)
- Phân phối thời gian giữa các messages
"""

import numpy as np
import matplotlib.pyplot as plt

# Load data
data_path = r"f:\OneDrive - 7t12kn\Documents\GitHub\ViT-MARL\saved_npz\lobster_AMZN_2012-06-21_10_fixed_time_1800_60_100_34200_57600.npz"
print(f"Đang load data từ: {data_path}")
data = np.load(data_path, allow_pickle=True)

msgs = data['msgs']
print(f"\nShape của messages: {msgs.shape}")
print(f"Total messages: {msgs.shape[0]:,}")

# Trích xuất thời gian (cột -2: time_s, cột -1: time_ns)
time_s = msgs[:, -2]
time_ns = msgs[:, -1]

# Chuyển sang thời gian tuyệt đối (giây + phần nano)
time_absolute = time_s + time_ns / 1e9

# Lọc bỏ các messages padding (time = 0 hoặc time không hợp lệ)
valid_mask = time_s > 0
time_valid = time_absolute[valid_mask]

print(f"\nSố messages hợp lệ (non-padding): {len(time_valid):,}")
print(f"Thời gian bắt đầu: {time_s[valid_mask][0]} giây (từ nửa đêm)")
print(f"Thời gian kết thúc: {time_s[valid_mask][-1]} giây (từ nửa đêm)")

# Chuyển đổi sang giờ đọc được
start_hour = time_s[valid_mask][0] // 3600
start_min = (time_s[valid_mask][0] % 3600) // 60
end_hour = time_s[valid_mask][-1] // 3600
end_min = (time_s[valid_mask][-1] % 3600) // 60

print(f"  → Từ {start_hour:02d}:{start_min:02d} đến {end_hour:02d}:{end_min:02d}")

# Tính khoảng thời gian giữa các messages liên tiếp
time_diffs = np.diff(time_valid)

# Lọc bỏ các giá trị âm hoặc 0 (nếu có)
time_diffs_positive = time_diffs[time_diffs > 0]

print(f"\n" + "="*60)
print("PHÂN TÍCH TỐC ĐỘ CẬP NHẬT ORDERBOOK")
print("="*60)

# Tính các thống kê
mean_interval = np.mean(time_diffs_positive)
median_interval = np.median(time_diffs_positive)
min_interval = np.min(time_diffs_positive)
max_interval = np.max(time_diffs_positive)

# Tính tần suất
mean_frequency = 1.0 / mean_interval if mean_interval > 0 else 0
median_frequency = 1.0 / median_interval if median_interval > 0 else 0

print(f"\n📊 KHOẢNG THỜI GIAN GIỮA CÁC MESSAGES:")
print(f"   Trung bình:  {mean_interval*1000:.3f} ms  ({mean_interval:.6f} giây)")
print(f"   Trung vị:    {median_interval*1000:.3f} ms  ({median_interval:.6f} giây)")
print(f"   Nhỏ nhất:    {min_interval*1000:.6f} ms  ({min_interval:.9f} giây)")
print(f"   Lớn nhất:    {max_interval*1000:.3f} ms  ({max_interval:.6f} giây)")

print(f"\n⚡ TẦN SUẤT CẬP NHẬT ORDERBOOK:")
print(f"   Trung bình:  {mean_frequency:.2f} messages/giây  ({mean_frequency*60:.0f} messages/phút)")
print(f"   Trung vị:    {median_frequency:.2f} messages/giây  ({median_frequency*60:.0f} messages/phút)")

# Tính tần suất trong các khoảng thời gian khác nhau
total_time = time_valid[-1] - time_valid[0]
print(f"\n📈 TỔNG QUAN:")
print(f"   Tổng thời gian: {total_time:.2f} giây ({total_time/60:.2f} phút)")
print(f"   Tổng messages:  {len(time_valid):,}")
print(f"   Tần suất tổng:  {len(time_valid)/total_time:.2f} messages/giây")

# Phân tích theo percentiles
percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"\n📉 PHÂN PHỐI KHOẢNG THỜI GIAN (milliseconds):")
for p in percentiles:
    val = np.percentile(time_diffs_positive, p)
    print(f"   P{p:2d}: {val*1000:8.3f} ms")

# Tính số config hiện tại
print(f"\n" + "="*60)
print("CẤU HÌNH HIỆN TẠI VS THỰC TẾ")
print("="*60)
current_n_per_step = 100
print(f"\n⚙️  Cấu hình hiện tại: n_data_msg_per_step = {current_n_per_step}")
print(f"   → Mỗi step xử lý {current_n_per_step} messages")
print(f"   → Thời gian trung bình mỗi step: {current_n_per_step * mean_interval:.3f} giây")
print(f"   → Tần suất step: {1/(current_n_per_step * mean_interval):.3f} steps/giây")

# Visualization
plt.figure(figsize=(14, 8))

# Subplot 1: Histogram of time intervals
plt.subplot(2, 2, 1)
plt.hist(time_diffs_positive * 1000, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Khoảng thời gian giữa messages (ms)')
plt.ylabel('Số lượng')
plt.title('Phân phối khoảng thời gian giữa messages')
plt.axvline(mean_interval * 1000, color='red', linestyle='--', label=f'Mean: {mean_interval*1000:.2f} ms')
plt.axvline(median_interval * 1000, color='green', linestyle='--', label=f'Median: {median_interval*1000:.2f} ms')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Log scale histogram
plt.subplot(2, 2, 2)
plt.hist(time_diffs_positive * 1000, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Khoảng thời gian (ms)')
plt.ylabel('Số lượng')
plt.title('Phân phối (Log scale)')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Subplot 3: Cumulative distribution
plt.subplot(2, 2, 3)
sorted_diffs = np.sort(time_diffs_positive * 1000)
cumulative = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs) * 100
plt.plot(sorted_diffs, cumulative)
plt.xlabel('Khoảng thời gian (ms)')
plt.ylabel('Phần trăm tích lũy (%)')
plt.title('Phân phối tích lũy')
plt.grid(True, alpha=0.3)
plt.xlim(0, np.percentile(time_diffs_positive * 1000, 95))

# Subplot 4: Time series of message rate
plt.subplot(2, 2, 4)
# Chia thành các bins theo thời gian (mỗi bin 60 giây)
bin_size = 60  # seconds
bins = np.arange(time_valid[0], time_valid[-1], bin_size)
counts, _ = np.histogram(time_valid, bins=bins)
bin_centers = bins[:-1] + bin_size / 2
bin_hours = (bin_centers - 34200) / 3600  # Convert to hours from 9:30 AM

plt.plot(bin_hours, counts / bin_size, linewidth=1)
plt.xlabel('Thời gian (giờ từ 9:30 AM)')
plt.ylabel('Messages/giây')
plt.title(f'Tần suất messages theo thời gian (window {bin_size}s)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('orderbook_frequency_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Đã lưu biểu đồ: orderbook_frequency_analysis.png")

plt.show()

print(f"\n" + "="*60)
print("KẾT LUẬN")
print("="*60)
print(f"""
Orderbook được cập nhật với tần suất:
  • Trung bình: ~{mean_frequency:.1f} lần/giây
  • Khoảng thời gian giữa các lần cập nhật: ~{mean_interval*1000:.2f} ms
  
Với cấu hình n_data_msg_per_step = {current_n_per_step}:
  • Mỗi step tương ứng ~{current_n_per_step * mean_interval:.2f} giây dữ liệu thực
  • Tần suất step: ~{1/(current_n_per_step * mean_interval):.2f} lần/giây
""")
