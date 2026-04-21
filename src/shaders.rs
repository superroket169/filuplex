// =============================
//            Shaders
// =============================

pub mod cs_add {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] + b[i];
            }
        ",
    }
}
pub mod cs_sub {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] - b[i];
            }
        ",
    }
}
pub mod cs_mul {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                c[i] = a[i] * b[i];
            }
        ",
    }
}
pub mod cs_div {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                if (b[i] == 0.0) { c[i] = 0.0; return; }
                c[i] = a[i] / b[i];
            }
        ",
    }
}
pub mod cs_mod {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                if (b[i] == 0.0) { c[i] = 0.0; return; }
                c[i] = mod(a[i], b[i]);
            }
        ",
    }
}
pub mod cs_matris_mul {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 8, local_size_y = 8) in;

            layout(set = 0, binding = 0) buffer MatA { float a[]; };  // M x K
            layout(set = 0, binding = 1) buffer MatB { float b[]; };  // K x N
            layout(set = 0, binding = 2) buffer MatC { float c[]; };  // M x N
            layout(set = 0, binding = 3) uniform Meta { uint M; uint K; uint N; };

            void main() {
                uint row = gl_GlobalInvocationID.x;
                uint col = gl_GlobalInvocationID.y;
                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint k = 0; k < K; k++) {
                    sum += a[row * K + k] * b[k * N + col];
                }
                c[row * N + col] = sum;
            }
        ",
    }
}
pub mod cs_relu {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer MatA { float a[]; };
            layout(set = 0, binding = 1) buffer MatB { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                b[i] = (a[i] < 0.0) ? 0 : a[i];
            }
        ",
    }
}
pub mod cs_softmax {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer Output { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;

                float max_val = a[0];
                for (uint k = 1; k < length; k++) {
                    max_val = max(max_val, a[k]);
                }

                float sum_exp = 0.0;
                for (uint k = 0; k < length; k++) {
                    sum_exp += exp(a[k] - max_val);
                }

                b[i] = exp(a[i] - max_val) / sum_exp;
            }
        ",
    }
}
pub mod cs_sigmoid {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;
            layout(set = 0, binding = 0) buffer Input { float a[]; };
            layout(set = 0, binding = 1) buffer Output { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                
                // Sigmoid hesaplama
                b[i] = 1.0 / (1.0 + exp(-a[i]));
            }
        ",
    }
}
pub mod cs_sigmoid_derivative {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;
            layout(set = 0, binding = 0) buffer Input { float a[]; };
            layout(set = 0, binding = 1) buffer Output { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                
                float s = a[i];
                b[i] = s * (1.0 - s);
            }
        ",
    }
}
pub mod cs_transpose {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            // Matris işlemleri olduğu için x ve y eksenli 2B grid kullanıyoruz (cs_matris_mul'daki gibi)
            layout(local_size_x = 8, local_size_y = 8) in;
            
            layout(set = 0, binding = 0) buffer Input { float a[]; };
            layout(set = 0, binding = 1) buffer Output { float b[]; };
            layout(set = 0, binding = 2) uniform Meta {
                uint rows;
                uint cols;
            };

            void main() {
                uint c = gl_GlobalInvocationID.x;
                uint r = gl_GlobalInvocationID.y;
                
                if (r >= rows || c >= cols) return;
                uint in_idx = r * cols + c;
                uint out_idx = c * rows + r;
                b[out_idx] = a[in_idx];
            }
        ",
    }
}
pub mod cs_matris_mul_transposed_b {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 8, local_size_y = 8) in;
            
            layout(set = 0, binding = 0) buffer InputA { float a[]; };
            layout(set = 0, binding = 1) buffer InputB { float b[]; };
            layout(set = 0, binding = 2) buffer Output { float c[]; };
            layout(set = 0, binding = 3) uniform Meta {
                uint M;
                uint K;
                uint N;
            };

            void main() {
                uint col = gl_GlobalInvocationID.x;
                uint row = gl_GlobalInvocationID.y;

                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint i = 0; i < K; i++) {
                    sum += a[row * K + i] * b[col * K + i];
                }

                c[row * N + col] = sum;
            }
        ",
    }
}
pub mod cs_sqrt {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;
            layout(set = 0, binding = 0) buffer Input { float a[]; };
            layout(set = 0, binding = 1) buffer Output { float b[]; };
            layout(set = 0, binding = 2) uniform Meta { uint length; };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;
                
                b[i] = sqrt(max(a[i], 0.0));
            }
        ",
    }
}
pub mod cs_matmul_relu {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 8, local_size_y = 8) in;

            layout(set = 0, binding = 0) buffer MatA { float a[]; };
            layout(set = 0, binding = 1) buffer MatB { float b[]; };
            layout(set = 0, binding = 2) buffer MatC { float c[]; };
            layout(set = 0, binding = 3) uniform Meta { uint M; uint K; uint N; };

            void main() {
                uint row = gl_GlobalInvocationID.x;
                uint col = gl_GlobalInvocationID.y;
                if (row >= M || col >= N) return;

                float sum = 0.0;
                for (uint k = 0; k < K; k++) {
                    sum += a[row * K + k] * b[k * N + col];
                }
                
                c[row * N + col] = max(0.0, sum);
            }
        ",
    }
}
pub mod cs_rmsprop_update {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 450
            layout(local_size_x = 64) in;

            layout(set = 0, binding = 0) buffer Weight { float w[]; };
            layout(set = 0, binding = 1) buffer Cache { float c[]; };
            layout(set = 0, binding = 2) buffer Grad { float dw[]; };
            layout(set = 0, binding = 3) uniform Meta { 
                uint length; 
                float learning_rate; 
                float decay; 
                float epsilon; 
            };

            void main() {
                uint i = gl_GlobalInvocationID.x;
                if (i >= length) return;

                c[i] = decay * c[i] + (1.0 - decay) * (dw[i] * dw[i]);
                w[i] -= (learning_rate * dw[i]) / (sqrt(c[i]) + epsilon);
            }
        ",
    }
}
