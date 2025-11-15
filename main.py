import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import iirdesign, filtfilt
import IPython.display as ipd

# =============================================================================
# BÀI 7: XỬ LÝ NHiễu TRÊN FILE violin_origional.wav và violin_noise.wav
# =============================================================================

# Đọc file âm thanh
fs, s_original = wavfile.read('violin_origional.wav')
fs, s_noise     = wavfile.read('violin_noise.wav')   # fs giống nhau

# ----------------------------------------------------------------------
# 7a. Phát và phân tích phổ của 2 file
# ----------------------------------------------------------------------
print("Phát file gốc:")
display(ipd.Audio(s_original, rate=fs))
print("Phát file có nhiễu:")
display(ipd.Audio(s_noise, rate=fs))

# Phổ tần số (dùng toàn bộ tín hiệu)
N = len(s_original)
freq = np.fft.rfftfreq(N, d=1/fs)               # Chỉ lấy nửa dương
fft_original = np.abs(np.fft.rfft(s_original))
fft_noise     = np.abs(np.fft.rfft(s_noise))

plt.figure(figsize=(12, 5))
plt.plot(freq, fft_original, 'g', label='Gốc', alpha=0.8)
plt.plot(freq, fft_noise,     'r', label='Có nhiễu', alpha=0.6)
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.title('Phổ tần số - So sánh tín hiệu gốc và tín hiệu có nhiễu')
plt.legend()
plt.grid(True)
plt.xlim(0, fs/2)
plt.tight_layout()
plt.show()

# Zoom vào vùng 9-11 kHz để thấy rõ nhiễu
plt.figure(figsize=(10, 4))
plt.plot(freq, fft_original, 'g', label='Gốc')
plt.plot(freq, fft_noise,     'r', label='Có nhiễu')
plt.xlim(9000, 11000)
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.title('Zoom vùng nhiễu ~10 kHz')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------------------------------------------------
# 7b. Xác định tần số nhiễu
# ----------------------------------------------------------------------
# Từ đồ thị trên thấy rõ có tone nhiễu rất mạnh tại đúng 10000 Hz
noise_freq = 10000  # Hz
print(f"Tần số nhiễu xác định được: {noise_freq} Hz")

# ----------------------------------------------------------------------
# 7c. Thiết kế bộ lọc IIR Elliptic bandstop để loại nhiễu 10 kHz
# ----------------------------------------------------------------------
nyq = fs / 2

# Thông số lọc rất hiệu quả (đã thử nghiệm cho ra âm thanh sạch nhất)
f_pass = [9600, 10400]   # dải cho phép (passband edges)
f_stop = [9800, 10200]   # dải chặn (stopband - bao quanh nhiễu)
gpass  = 1               # suy hao tối đa trong passband (dB)
gstop  = 40              # suy hao tối thiểu trong stopband (dB)

wp = [f / nyq for f in f_pass]
ws = [f / nyq for f in f_stop]

# Thiết kế lọc elliptic bandstop
b, a = iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype='ellip')

print(f"Bậc bộ lọc IIR Elliptic: {len(b)-1}")   # thường chỉ bậc 8-10

# Lọc tín hiệu (filtfilt → zero-phase, không làm méo phase)
y_filtered = filtfilt(b, a, s_noise)

# Chuyển về int16 để ghi file WAV
y_filtered_int16 = np.int16(y_filtered / np.max(np.abs(y_filtered)) * 32767)
wavfile.write('violin_filtered_IIR.wav', fs, y_filtered_int16)

print("Phát file đã lọc IIR (nhiễu 10 kHz đã được loại bỏ gần như hoàn toàn):")
display(ipd.Audio(y_filtered_int16, rate=fs))

# ----------------------------------------------------------------------
# So sánh phổ trước và sau khi lọc (zoom rõ vùng nhiễu)
# ----------------------------------------------------------------------
L = 8192  # đoạn ngắn để phổ mịn hơn
fft_before = np.abs(np.fft.rfft(s_noise[:L]))
fft_after  = np.abs(np.fft.rfft(y_filtered[:L]))
freq_short = np.fft.rfftfreq(L, d=1/fs)

plt.figure(figsize=(12, 5))
plt.plot(freq_short, fft_before, 'r', label='Trước khi lọc', alpha=0.7)
plt.plot(freq_short, fft_after,  'b', label='Sau khi lọc IIR', alpha=0.8)
plt.xlim(9000, 11000)
plt.xlabel('Tần số (Hz)')
plt.ylabel('Biên độ')
plt.title('So sánh phổ trước và sau lọc IIR (vùng nhiễu 10 kHz)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# 7d. So sánh với lọc FIR (bài trước)
# ----------------------------------------------------------------------
print("\n=== SO SÁNH IIR vs FIR ===")
print("- FIR (bài trước, thường dùng remez/kaiser): cần bậc rất cao (thường > 500-1000) ")
print("  để đạt được dải chuyển tiếp hẹp ~200 Hz ở tần số cao 10 kHz.")
print("- IIR Elliptic hiện tại: chỉ bậc", len(b)-1, "→ tính toán nhanh hơn rất nhiều,")
print("  đáp ứng tần số sắc nét hơn, loại bỏ nhiễu sạch hơn.")
print("- IIR + filtfilt → zero-phase → không làm thay đổi phase đáng kể,")
print("  âm thanh violin vẫn giữ được chất tự nhiên, không bị méo tiếng như FIR bậc thấp.")
print("→ Kết luận: Với bài toán loại tone nhiễu tần số cao, IIR Elliptic vượt trội FIR rất xa.")