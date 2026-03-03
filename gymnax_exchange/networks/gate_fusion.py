import jax
import jax.numpy as jnp
from flax import linen as nn

class EMASmoothing(nn.Module):
    """
    Bước 1: Tiền xử lý và Khử nhiễu.
    Làm mượt dữ liệu định lượng o_t theo trục thời gian để giữ lại xu hướng thanh khoản thực tế.
    """
    alpha: float = 0.5  # Hệ số làm mượt (smoothing factor) tương đương (1 - beta)

    @nn.compact
    def __call__(self, o_t):
        # o_t có shape: (batch_size, seq_len, features)
        
        # Hàm xử lý từng bước thời gian cho jax.lax.scan
        def ema_step(carry, x_t):
            # \tilde{o}_t = \alpha * x_t + (1 - \alpha) * \tilde{o}_{t-1}
            next_carry = self.alpha * x_t + (1.0 - self.alpha) * carry
            return next_carry, next_carry

        # Khởi tạo trạng thái ban đầu (carry) là zero cho bước t=0
        init_carry = jnp.zeros_like(o_t[:, 0, :])
        
        # Đổi trục để đưa seq_len lên đầu phục vụ cho jax.lax.scan: (seq_len, batch_size, features)
        o_t_time_major = jnp.swapaxes(o_t, 0, 1)
        _, o_t_smoothed = jax.lax.scan(ema_step, init_carry, o_t_time_major)
        
        # Trả lại shape ban đầu: (batch_size, seq_len, features)
        return jnp.swapaxes(o_t_smoothed, 0, 1)


class StableGatedCrossAttention(nn.Module):
    """
    Bước 2, 3 & 4: Linear Cross-Attention, Stable Gating và MLP Aggregation.
    """
    d_model: int = 128  # Chiều không gian tiềm ẩn chung (Common Latent Space)
    
    @nn.compact
    def __call__(self, o_t_smoothed, z_t):
        # o_t_smoothed: Số liệu LOB đã làm mượt (Primary Modality) - Shape: (batch, seq_len, d_o)
        # z_t: Đặc trưng ảnh (Secondary Modality) - Shape: (batch, spatial_len, d_z)
        
        # ------------------------------------------------------------------
        # BƯỚC 2: HỢP NHẤT SƠ BỘ TỐC ĐỘ CAO (LINEAR CROSS-ATTENTION)
        # ------------------------------------------------------------------
        # Ánh xạ 2 phương thức khác kích thước vào không gian tiềm ẩn chung (d_model)
        Q = nn.Dense(self.d_model, name='W_Q')(o_t_smoothed) # Query từ Số (batch, seq_len, d_model)
        K = nn.Dense(self.d_model, name='W_K')(z_t)          # Key từ Ảnh (batch, spatial_len, d_model)
        V = nn.Dense(self.d_model, name='W_V')(z_t)          # Value từ Ảnh (batch, spatial_len, d_model)
        
        # Hàm xấp xỉ phi tuyến \phi(x) = elu(x) + 1 cho Linear Attention
        def phi(x):
            return nn.elu(x) + 1.0
        
        Q_phi = phi(Q)
        K_phi = phi(K)
        
        # Tính toán Linear Attention O(N) thông qua hàm einsum
        # 1. Tính Tích K^T * V -> Shape: (batch, d_model, d_model)
        KV = jnp.einsum('bsd,bse->bde', K_phi, V)
        # 2. Tử số: Q * (K^T * V) -> Shape: (batch, seq_len, d_model)
        numerator = jnp.einsum('btd,bde->bte', Q_phi, KV)
        
        # 3. Mẫu số: Q * sum(K) dọc theo spatial_len
        K_sum = jnp.sum(K_phi, axis=1)  # (batch, d_model)
        denominator = jnp.einsum('btd,bd->bt', Q_phi, K_sum)[..., None] # Thêm trục để chia: (batch, seq_len, 1)
        
        # Đặc trưng hợp nhất sơ bộ (tránh chia cho 0)
        H_unstable = numerator / (denominator + 1e-6) 
        
        # ------------------------------------------------------------------
        # BƯỚC 3: KIỂM DUYỆT AN TOÀN (STABLE GATING)
        # ------------------------------------------------------------------
        # Dùng số liệu định lượng làm cổng kiểm duyệt
        gate_logits = nn.Dense(self.d_model, name='W_g')(o_t_smoothed)
        g_t = nn.sigmoid(gate_logits) # Shape: (batch, seq_len, d_model)
        
        # Triệt tiêu ảo giác thị giác bằng phép nhân Hadamard (element-wise)
        H_stable = g_t * H_unstable 
        
        # ------------------------------------------------------------------
        # BƯỚC 4: NÉN VÀ TỔNG HỢP (MLP AGGREGATION)
        # ------------------------------------------------------------------
        x = nn.Dense(self.d_model, name='MLP_L1')(H_stable)
        x = nn.relu(x)
        # Nén số chiều xuống một nửa để đưa vào RNN-IPPO cho nhẹ
        H_compact = nn.Dense(self.d_model // 2, name='MLP_L2')(x) 
        H_compact = nn.relu(H_compact)
        
        return H_compact

if __name__ == "__main__":
    batch_size = 32
    seq_len = 10      
    spatial_len = 64 
    d_o = 20          
    d_z = 256 

    rng = jax.random.PRNGKey(0)
    o_t_raw = jax.random.normal(rng, (batch_size, seq_len, d_o))
    z_t = jax.random.normal(rng, (batch_size, spatial_len, d_z))
    
    ema_module = EMASmoothing(alpha=0.5)
    variables_ema = ema_module.init(rng, o_t_raw)
    o_t_smoothed = ema_module.apply(variables_ema, o_t_raw)
    
    fusion_module = StableGatedCrossAttention(d_model=128)
    variables_fusion = fusion_module.init(rng, o_t_smoothed, z_t)
    H_compact = fusion_module.apply(variables_fusion, o_t_smoothed, z_t)
    
    print(f"Shape đầu vào o_t (Số): {o_t_raw.shape}")
    print(f"Shape đầu vào z_t (Ảnh): {z_t.shape}")
    print(f"Shape đầu ra H_compact (để đưa vào RNN-IPPO): {H_compact.shape}")