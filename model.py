"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // TODO: implement warp-level sum reduction using shuffle intrinsics
    int arr[] = {16, 8, 4, 2, 1};
    for (int num : arr){
        val += __shfl_xor_sync(0xffffffff, val, num);
    }
    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // TODO: implement warp-level max reduction using shuffle intrinsics
    int arr[] = {16, 8, 4, 2, 1};
    #pragma unroll
    for (int offset : arr){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

# Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // TODO: block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_sum(val);

    if (lane == 0){
        shared[warp_id] = val;
    }

    __syncthreads();

    // Use the first warp to reduce the gathered warp sums
    // If block size is < 1024, fill empty slots with 0.0f
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane < num_warps) ? shared[lane] : 0.0f;

    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

# Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    // TODO: block-wide max via warp_reduce_max + shared memory
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    val = warp_reduce_max(val);

    if (lane == 0){
        shared[warp_id] = val;
    }

    __syncthreads();

    // Use the first warp to call warp_reduce_max the gathered warp sums
    int num_warps = (blockDim.x + 31) / 32;
    val = (lane < num_warps) ? shared[lane] : -INFINITY;

    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (; i < n; i+=blockDim.x * gridDim.x){
      out[i] = x[i] + residual[i];
  }
  (void)x; (void)residual; (void)out; (void)n;
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i+=blockDim.x * gridDim.x){
        float v = x[i];
        float v3 = v*v*v;
        float gelu = 0.5 * v * (1.0f + tanh(sqrtf(2.0f / M_PI) * (v + 0.044715f * v3)));
        out[i] = gelu;
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    for (; i < n; i+=blockDim.x * gridDim.x){
        out[i] = x[i] / (1.0f + exp(-x[i]));
    }
    (void)x; (void)out; (void)n;
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (; i < n; i+=blockDim.x * gridDim.x){
        float v = gate[i] / (1.0f + exp(-gate[i]));
        out[i] = v * up[i];
    }
    (void)gate; (void)up; (void)out; (void)n;
}

# Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    // TODO: Apply RMSNorm per row (one block per row)
    
    const float* x_row = x + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;
    extern __shared__ float shared[];
    float sum_sq = 0.0f;

    
    for (int i = threadIdx.x; i < n; i += blockDim.x){
        sum_sq += x_row[i] * x_row[i];
    }

    float sum_block_reduce = block_reduce_sum(sum_sq, shared);

    __shared__ float s_rms;

    if (threadIdx.x == 0){
        s_rms = rsqrtf(sum_block_reduce / n + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x){
        out_row[i] = x_row[i] * s_rms * weight[i];
    }
    
}

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // TODO: per-row LayerNorm using block_reduce_sum for mean and variance
    const float* x_row = x + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;
    extern __shared__ float shared[];
    float sum = 0.0f;
    float tmp = 0.0f;

    for (int i = threadIdx.x; i < n; i+= blockDim.x){
        sum += x_row[i];
    }
    float sum_block_reduce = block_reduce_sum(sum, shared);
    __shared__ float row_mean;
    if (threadIdx.x == 0){
        row_mean = sum_block_reduce / n;
    }
    __syncthreads();
    

    for (int i = threadIdx.x; i < n; i+= blockDim.x){
        tmp += (x_row[i] - row_mean) * (x_row[i] - row_mean);
    }
    float tmp_reduce = block_reduce_sum(tmp, shared);
    __shared__ float row_var;
    if (threadIdx.x == 0){
        row_var = tmp_reduce / n;
    }
    __syncthreads();
    

    for (int i = threadIdx.x; i < n; i += blockDim.x){
        out_row[i] = (x_row[i] - row_mean) * rsqrt(row_var + eps) * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // TODO: fuse residual addition with RMSNorm (one block per row)
    const float* x_ptr = x + blockIdx.x * n;
    const float* residual_ptr = residual + blockIdx.x * n;
    float* residual_out_ptr = residual_out + blockIdx.x * n;
    float* out_ptr = out + blockIdx.x * n;
    float sum_sqs = 0.0f;
    extern __shared__ float shared[];
    __shared__ float inv_rms;

    // residual addition and sum of squares
    for (int i = threadIdx.x; i < n; i+= blockDim.x){
        float r = x_ptr[i] + residual_ptr[i];
        residual_out_ptr[i] = r;

        sum_sqs += r * r;
    }

    float sum_sqs_reduce = block_reduce_sum(sum_sqs, shared);

    if (threadIdx.x == 0){
        inv_rms = rsqrtf(sum_sqs_reduce / n + eps);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < n; i += blockDim.x){
        out_ptr[i] = residual_out_ptr[i] * inv_rms * weight[i];
    }
}

# Step 12 - softmax_row_kernel
__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax (one block per row)
    //Each row contains cols elements
    const float* x_ptr = x + blockIdx.x * cols;
    float* out_ptr = out + blockIdx.x * cols;
    extern __shared__ float shared[];
    float local_max = -INFINITY; float expf_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        if (x_ptr[i] > local_max){
            local_max = x_ptr[i];
        }
    }

    float local_max_reduce = block_reduce_max(local_max, shared);

    __shared__ float row_max;

    if (threadIdx.x == 0){
        row_max = local_max_reduce;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        expf_sum += expf(x_ptr[i] - row_max);
    }

    float expf_sum_reduce = block_reduce_sum(expf_sum, shared);

    __shared__ float row_expf_sum;

    if (threadIdx.x == 0){
        row_expf_sum = expf_sum_reduce;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        out_ptr[i] = expf(x_ptr[i] - row_max) / row_expf_sum;
    }

}

# Step 13 - causal_softmax_kernel
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: numerically stable causal softmax (one block per row);
    //       mask columns c > row to 0; use block_reduce_max / block_reduce_sum
    const float* x_ptr = x + blockIdx.x * cols;
    float* out_ptr = out + blockIdx.x * cols;
    extern __shared__ float shared[];
    float local_max = -INFINITY; float expf_sum = 0.0f;
    int valid_len = min(blockIdx.x + 1, cols); // only columns c <= r contribute to the softmax

    for (int i = threadIdx.x; i < valid_len; i += blockDim.x){
        if (x_ptr[i] > local_max) local_max = x_ptr[i];
    }

    float local_max_reduce = block_reduce_max(local_max, shared);

    __shared__ float row_max;

    if (threadIdx.x == 0){
        row_max = local_max_reduce;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < valid_len; i += blockDim.x){
        expf_sum += expf(x_ptr[i] - row_max);
    }

    float expf_sum_reduce = block_reduce_sum(expf_sum, shared);

    __shared__ float row_expf_sum;

    if (threadIdx.x == 0){
        row_expf_sum = expf_sum_reduce;
    }

    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x){
        if (i < valid_len) {
            out_ptr[i] = expf(x_ptr[i] - row_max) / row_expf_sum;
        }
        else{
            out_ptr[i] = 0.0f;
        }
    }

}

# Step 14 - embedding_lookup_kernel
__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    // TODO: gather embedding vectors for each token id into out
    //const int* token_ids_ptr
    int total = seq_len * embed_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    // grid-stride implementation to have one thread handle multiple output elements
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int token_pos = i / embed_dim;
        int dim = i % embed_dim;
        int token_id = token_ids[token_pos];

        out[i] = weight[token_id * embed_dim + dim];
    }
    
}

# Step 15 - rope_kernel
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
   // TODO: apply RoPE rotation in-place to every even/odd pair of q and k
   int total = seq_len * n_heads * (head_dim/2);
   int pairs = head_dim / 2;
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x * gridDim.x;

   if (idx >= total) return;

   for (int i = idx; i < total; i+=stride){
      int pair_i = i % pairs;
      int h = (i / pairs) % n_heads;
      int t = i / (n_heads * pairs);

      int base = (t * n_heads + h) * head_dim;
      int even = base + 2*pair_i;
      int odd = even + 1;

      // load tables
      float cos = cos_table[t * head_dim/2 + pair_i];
      float sin = sin_table[t * head_dim/2 + pair_i];

      // load even and odd values for q and k
      float q0 = q[even];
      float q1 = q[odd];
      float k0 = k[even];
      float k1 = k[odd];

      //apply rotation transformation
      q[even] = cos * q0 - sin * q1;
      q[odd] = sin * q0 + cos * q1;
      k[even] = cos * k0 - sin * k1;
      k[odd] = sin * k0 + cos * k1;

   }
}

# Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // TODO: compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    // int total = M * N;
    // for (int i = threadIdx.x; i < total; i += blockDim.x){
    //     for (int m = 0; m < M; m++){
    //         for (int n = 0; n < N; n++){
    //             if (m >= M || n >= N) return;
    //             float counter = 0.f;
    //             for (int k = 0; k < K; k++){
    //                 counter += x[m*K + k] * weight[n*K + k];
    //             }
    //             if (bias != nullptr) counter += bias[n];
    //             out[m*N + n] = counter;
    //         }
    //     }
    // }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;

    const float* x_vec = &x[row * K];
    const float* weight_vec = &weight[col * K];

    for (int k = 0; k < K; k++) {
        sum += x_vec[k] * weight_vec[k];
    }

    if (bias != nullptr) sum += bias[col];

    out[row * N + col] = sum;
}

# Step 17 - fused_linear_bias_gelu_kernel
__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // TODO: fuse matmul, bias add, and GELU tanh approx into one kernel
    (void)x; (void)weight; (void)bias; (void)out; (void)M; (void)N; (void)K;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N; int n = idx % N;
    //float* sum = new float[K];
    float sum = 0.0f;

    const float* x_vec = &x[m * K];
    const float* weight_vec = &weight[n * K];
    
    for (int k = 0; k < K; k++){
        sum += x_vec[k] * weight_vec[k];
    }
    sum += bias[n];

    float v = sum;
    float v3 = v*v*v;
    float gelu = 0.5 * v * (1.0f + tanh(sqrtf(2.0f / M_PI) * (v + 0.044715f * v3)));
    out[idx] = gelu;
}

# Step 18 - mlp_swiglu_forward
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // TODO: allocate temps, run gate/up linears, swiglu, then down projection
    (void)x; (void)w_gate; (void)w_up; (void)w_down; (void)out;
    (void)M; (void)hidden_dim; (void)intermediate_dim;

    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int length = M * intermediate_dim;

    // cudaMalloc three temps
    float* temps;
    cudaMalloc((void**)&temps, 3 * length * sizeof(float));
    float* d_gate = temps;
    float* d_up = temps + length;
    float* d_act = temps + 2*length;

    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    linear_kernel<<<blocks, threads>>>(x, w_gate, nullptr, d_gate, M, intermediate_dim, hidden_dim);
    linear_kernel<<<blocks, threads>>>(x, w_up, nullptr, d_up, M, intermediate_dim, hidden_dim);
    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_act, M * intermediate_dim);
    linear_kernel<<<blocks, threads>>>(d_act, w_down, nullptr, out, M, hidden_dim, intermediate_dim);

    cudaDeviceSynchronize();


    cudaFree(temps);
}

# Step 19 - rmsnorm_residual_block
void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    // TODO: launch fused_add_rmsnorm_kernel for the pre-norm residual+RMSNorm block
    int threads = 256;
    fused_add_rmsnorm_kernel<<<rows, threads>>>(x, residual, weight, out, residual_out, n, eps);
}

# Step 20 - run_transformer_ffn
void run_transformer_ffn(const float* x, const float* residual,
                         const float* norm_weight, const float* w_gate,
                         const float* w_up, const float* w_down, float* out,
                         int M, int hidden_dim, int intermediate_dim,
                         float eps) {
  // TODO: residual+RMSNorm, SwiGLU MLP, then residual add into out
  int threads = 256;
  int length = M * hidden_dim;
  int blocks = (length + threads - 1)/threads;

  float* temps;
  cudaMalloc((void**)&temps, 2 * length * sizeof(float));

  float* r = temps;
  float* x_hat = temps + length;

  rmsnorm_residual_block(x, residual, norm_weight, x_hat, r, M, hidden_dim, eps);
  
  mlp_swiglu_forward(x_hat, w_gate, w_up, w_down, out, M, hidden_dim, intermediate_dim);

  add_residual_kernel<<<blocks, threads>>>(r, out, out, length);


  cudaDeviceSynchronize();
  cudaFree(temps);
  
}

