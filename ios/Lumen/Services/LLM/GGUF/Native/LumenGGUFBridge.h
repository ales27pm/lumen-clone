#ifndef LUMEN_GGUF_BRIDGE_H
#define LUMEN_GGUF_BRIDGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 Future C ABI boundary for the Lumen GGUF backend.

 This header intentionally has no implementation in this patch. The next native
 backend patch can wire these declarations to llama.cpp without changing the
 Swift-side lifecycle and streaming contracts.
 */

typedef struct lumen_gguf_context lumen_gguf_context_t;

typedef struct {
    const char *model_path;
    int32_t context_tokens;
    int32_t batch_size;
    int32_t thread_count;
    int32_t gpu_layer_count;
    bool use_metal;
    bool use_memory_mapping;
} lumen_gguf_load_config_t;

typedef struct {
    double temperature;
    double top_p;
    int32_t top_k;
    double repeat_penalty;
    uint64_t seed;
    bool has_seed;
    int32_t max_tokens;
    const char *const *stop_sequences;
    size_t stop_sequence_count;
} lumen_gguf_sampling_config_t;

typedef void (*lumen_gguf_token_callback_t)(const char *token, void *user_data);
typedef void (*lumen_gguf_error_callback_t)(const char *message, void *user_data);

lumen_gguf_context_t *lumen_gguf_create_context(void);
void lumen_gguf_destroy_context(lumen_gguf_context_t *context);
int32_t lumen_gguf_load_model(lumen_gguf_context_t *context, const lumen_gguf_load_config_t *config);
void lumen_gguf_unload_model(lumen_gguf_context_t *context);
int32_t lumen_gguf_generate(
    lumen_gguf_context_t *context,
    const char *prompt,
    const lumen_gguf_sampling_config_t *sampling,
    lumen_gguf_token_callback_t token_callback,
    lumen_gguf_error_callback_t error_callback,
    void *user_data
);
void lumen_gguf_cancel(lumen_gguf_context_t *context);

#ifdef __cplusplus
}
#endif

#endif
