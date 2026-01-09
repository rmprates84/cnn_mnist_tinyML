#include <stdio.h>
#include <math.h>
#include "pico/stdlib.h"

#include "tflm_wrapper.h"
#include "mnist_sample.h"

static int argmax_i8(const int8_t* v, int n) {
    int best = 0;
    int8_t bestv = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bestv) { bestv = v[i]; best = i; }
    }
    return best;
}

static int8_t quantize_f32_to_i8(float x, float scale, int zp) {
    // q = round(x/scale) + zp
    long q = lroundf(x / scale) + zp;
    if (q < -128) q = -128;
    if (q >  127) q = 127;
    return (int8_t)q;
}

int main() {
    stdio_init_all();
    sleep_ms(1500);
    printf("\n=== MNIST CNN INT8 no Pico W (main em C) ===\n");

    int rc = tflm_init();
    if (rc != 0) {
        printf("tflm_init falhou: %d\n", rc);
        while (1) tight_loop_contents();
    }

    printf("Arena usada (bytes): %d\n", tflm_arena_used_bytes());

    int in_bytes = 0;
    int8_t* in = tflm_input_ptr(&in_bytes);

    int out_bytes = 0;
    int8_t* out = tflm_output_ptr(&out_bytes);

    if (!in || !out) {
        printf("Erro: ponteiro input/output nulo\n");
        while (1) tight_loop_contents();
    }

    printf("Input bytes: %d | Output bytes: %d\n", in_bytes, out_bytes);

    // Quantização do tensor de entrada
    float in_scale = tflm_input_scale();
    int   in_zp    = tflm_input_zero_point();

    float out_scale = tflm_output_scale();
    int   out_zp    = tflm_output_zero_point();

    printf("IN:  scale=%f zp=%d\n", in_scale, in_zp);
    printf("OUT: scale=%f zp=%d\n", out_scale, out_zp);

    // Esperado: 28*28 = 784 bytes
    if (in_bytes < 28*28) {
        printf("Erro: input menor que 784 bytes\n");
        while (1) tight_loop_contents();
    }

    // Pré-processamento igual ao treino: x = pixel/255.0 (float), depois quantiza p/ int8
    for (int i = 0; i < 28*28; i++) {
        float x = (float)mnist_sample_28x28[i] / 255.0f;
        in[i] = quantize_f32_to_i8(x, in_scale, in_zp);
    }

    // Inferência
    rc = tflm_invoke();
    if (rc != 0) {
        printf("Invoke falhou: %d\n", rc);
        while (1) tight_loop_contents();
    }

    // Saída: 10 classes (MNIST)
    int pred = argmax_i8(out, 10);

    printf("Label esperado: %d\n", mnist_sample_label);
    printf("Predito: %d\n", pred);

    // Mostrar scores aproximados (dequant)
    for (int c = 0; c < 10; c++) {
        int8_t q = out[c];
        float y = (float)(q - out_zp) * out_scale;
        printf("c%d: q=%d y~=%f\n", c, (int)q, y);
    }

    while (1) sleep_ms(1000);
}
