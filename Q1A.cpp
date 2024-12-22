#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>  // For AVX2 intrinsics
#include <intrin.h>     // For __cpuid

#define N 256 // Input size of the matrix

// Declare matrices A, B, C, and the vectorized result matrix
float A[N][N], B[N][N], C[N][N];
float C_vectorized[N][N];

// Function prototypes
void init();
void q1();
void q1_vectorized();
void check_correctness();
int check_avx_support();

int main() {
    double start_1, end_1;

    init(); // Initialize matrices

    // Measure time for original q1 function (non-vectorized)
    start_1 = omp_get_wtime();
    q1();
    end_1 = omp_get_wtime();
    printf("Time for original q1 (non-vectorized): %f seconds\n", end_1 - start_1);

    // Check if AVX2 is supported
    if (check_avx_support()) {
        printf("AVX2 is supported. Using vectorized function.\n");

        // Measure time for new vectorized q1 function (using AVX/AVX2)
        start_1 = omp_get_wtime();
        q1_vectorized();
        end_1 = omp_get_wtime();
        printf("Time for vectorized q1 (AVX/AVX2): %f seconds\n", end_1 - start_1);
    }
    else {
        printf("AVX2 is not supported. Skipping vectorized function.\n");

        // Fallback to the original (non-vectorized) matrix multiplication
        start_1 = omp_get_wtime();
        q1();
        end_1 = omp_get_wtime();
        printf("Time for fallback (original q1): %f seconds\n", end_1 - start_1);
    }

    // Check correctness of vectorized matrix multiplication
    check_correctness();

    system("pause"); // Prevents the output window from closing immediately
    return 0;
}

// Function to check if the CPU supports AVX2
int check_avx_support() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);  // Get CPU info
    return (cpuInfo[2] & (1 << 5)) != 0;  // Check if the AVX2 bit (bit 5) is set in the CPU info
}

// Function to initialize matrices A, B, and C
void init() {
    float e = 0.1234f, p = 0.7264f; // Constant values for matrix initialization

    // Initialize matrices A and B, and set matrix C to zero
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            A[i][j] = ((i - j) % 9) + p; // Assign values to matrix A
            B[i][j] = ((i + j) % 11) + e; // Assign values to matrix B
            C[i][j] = 0.0f; // Set all elements of matrix C to zero
            C_vectorized[i][j] = 0.0f; // Initialize vectorized result matrix
        }
    }
}

// Function to perform matrix multiplication (non-vectorized)
void q1() {
    // Matrix multiplication: C = A * B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j]; // Compute the dot product for each element of C
            }
        }
    }
}

// Function to perform vectorized matrix multiplication using AVX2
void q1_vectorized() {
    // Matrix multiplication: C = A * B using AVX2 (faster vectorized approach)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 c_val = _mm256_setzero_ps(); // Initialize a 256-bit register to zero

            for (int k = 0; k < N; k += 8) {
                // Load 8 elements from A[i][k...k+7] into a 256-bit register
                __m256 a_vals = _mm256_loadu_ps(&A[i][k]);

                // Load 8 elements from B[k...k+7][j] into a 256-bit register
                __m256 b_vals = _mm256_loadu_ps(&B[k][j]);

                // Perform element-wise multiplication and accumulate into c_val
                c_val = _mm256_fmadd_ps(a_vals, b_vals, c_val); // Fused multiply-add (A*B + C)
            }

            // Store the result of the vectorized multiplication into the C matrix
            float temp[8];
            _mm256_storeu_ps(temp, c_val);

            for (int idx = 0; idx < 8; idx++) {
                C_vectorized[i][j] += temp[idx]; // Accumulate the results for each element
            }
        }
    }
}

// Function to check the correctness of the vectorized matrix multiplication
void check_correctness() {
    int correct = 1;

    // Compare the original C matrix and the vectorized C_matrix to ensure correctness
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C[i][j] - C_vectorized[i][j]) > 1e-6) {  // Check if the difference is larger than a small threshold
                correct = 0;  // If a mismatch is found, set correct to 0
                break;
            }
        }
        if (!correct) break;  // If any mismatch is found, exit the loop early
    }

    if (correct) {
        printf("The vectorized matrix multiplication is correct!\n");
    }
    else {
        printf("The vectorized matrix multiplication is incorrect!\n");
    }
}
