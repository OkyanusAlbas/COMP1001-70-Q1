#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h> // This header is needed for AVX2 intrinsics

#define N 256 // input size

float A[N][N], B[N][N], C[N][N];

void init();
void q1();
void q1_vectorized();
void check_correctness(float C_serial[N][N], float C_vectorized[N][N]);

int main() {
    double start_1, end_1;

    init(); // initialize the arrays

    start_1 = omp_get_wtime(); // start the timer
    q1(); // original routine
    end_1 = omp_get_wtime(); // end the timer
    printf("Original q1() execution time: %f seconds\n", end_1 - start_1);

    // Initialize C to zero again for the vectorized version
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }

    start_1 = omp_get_wtime(); // start the timer for vectorized version
    q1_vectorized(); // vectorized routine using AVX2
    end_1 = omp_get_wtime(); // end the timer
    printf("Vectorized q1() execution time: %f seconds\n", end_1 - start_1);

    // Check correctness of the vectorized version
    check_correctness(C, C); // compare the results

    return 0;
}

void init() {
    float e = 0.1234f, p = 0.7264f;
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            A[i][j] = ((i - j) % 9) + p;
            B[i][j] = ((i + j) % 11) + e;
            C[i][j] = 0.0f;
        }
    }
}

void q1() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void q1_vectorized() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum = _mm256_setzero_ps(); // Initialize an AVX2 register to zero
            for (int k = 0; k < N; k += 8) { // Process 8 elements at a time
                // Load 8 elements from row i of A and column j of B
                __m256 a_vals = _mm256_loadu_ps(&A[i][k]);
                __m256 b_vals = _mm256_loadu_ps(&B[k][j]);

                // Multiply A[i][k..k+7] with B[k..k+7][j] and accumulate the result
                sum = _mm256_fmadd_ps(a_vals, b_vals, sum);
            }

            // Horizontal sum of the 8 elements in the AVX register to get the final result
            float result[8];
            _mm256_storeu_ps(result, sum);
            for (int l = 0; l < 8; l++) {
                C[i][j] += result[l];
            }
        }
    }
}

void check_correctness(float C_serial[N][N], float C_vectorized[N][N]) {
    int correct = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C_serial[i][j] - C_vectorized[i][j]) > 1e-6) {
                printf("Mismatch at C[%d][%d]: Serial = %f, Vectorized = %f\n", i, j, C_serial[i][j], C_vectorized[i][j]);
                correct = 0;
            }
        }
    }
    if (correct) {
        printf("Both routines produced the same result.\n");
    }
    else {
        printf("There were mismatches between the serial and vectorized results.\n");
    }
}
