// qwen3_asr_batch_truncation.cpp - real-model gated regression test for Qwen3-ASR's
// audio-proportional generation budget in single-shot and batch decode.
//
// The old fixed max_new=256 path truncated love-loss.wav even though its audio
// tokens and natural transcript fit easily within Qwen's 65536-token context.
// The long clip must now reach EOS in both decode paths, while jfk.wav guards
// the unchanged short-input path. This test fails against the old code.
//
// Batch makeup:
//   row 0 = jfk.wav (~11 s)        -> 256-token minimum budget -> OK
//   row 1 = love-loss.wav (~197 s) -> audio-scaled budget      -> OK
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) builds it.
//   - At runtime, TRANSCRIBE_QWEN3_ASR_GGUF points at either variant's GGUF.
//     The variant-specific 1.7B/0.6B variables are fallback aliases. If no
//     model or sample is available, exits 77 ("skipped").

#include "transcribe.h"
#include "wav.h"

#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool load_sample(const std::string & name, std::vector<float> & pcm) {
    const std::string path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name;
    if (!file_exists(path)) {
        std::fprintf(stderr, "skipping: %s missing\n", path.c_str());
        return false;
    }
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err) || pcm.empty()) {
        std::fprintf(stderr, "failed to load %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const char * model_path = std::getenv("TRANSCRIBE_QWEN3_ASR_GGUF");
    if (model_path == nullptr || *model_path == '\0') {
        model_path = std::getenv("TRANSCRIBE_QWEN3_ASR_1_7B_GGUF");
    }
    if (model_path == nullptr || *model_path == '\0') {
        model_path = std::getenv("TRANSCRIBE_QWEN3_ASR_0_6B_GGUF");
    }
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr, "skipping: no Qwen3-ASR GGUF env var is set to an existing model\n");
        return 77;
    }

    std::vector<float> pcm_short, pcm_long;
    if (!load_sample("jfk.wav", pcm_short)) {
        return 77;
    }
    if (!load_sample("love-loss.wav", pcm_long)) {
        return 77;
    }

    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "model load failed: %s\n", model_path);
        return 1;
    }

    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    struct transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "session init failed\n");
        transcribe_model_free(model);
        return 1;
    }

    // ---- Single-shot: long-form must finish rather than hit the old 256 cap.
    {
        const transcribe_status rl = transcribe_run(s, pcm_long.data(), (int) pcm_long.size(), nullptr);
        CHECK(rl == TRANSCRIBE_OK);
        CHECK(transcribe_was_truncated(s) == false);
        const char * t = transcribe_full_text(s);
        CHECK(t != nullptr && std::strlen(t) > 1000);
    }
    {
        const transcribe_status rs = transcribe_run(s, pcm_short.data(), (int) pcm_short.size(), nullptr);
        CHECK(rs == TRANSCRIBE_OK);
        CHECK(transcribe_was_truncated(s) == false);  // reset + completed
    }

    // ---- Batch parity: both utterances must reach EOS under the same policy.
    {
        const float * pcms[2] = { pcm_short.data(), pcm_long.data() };
        const int     lens[2] = { (int) pcm_short.size(), (int) pcm_long.size() };

        CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
        CHECK_EQ_INT(transcribe_batch_n_results(s), 2);

        CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_OK);
        CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_OK);

        // Both rows keep their complete transcript; the long row must be well
        // past the amount of text the old 256-token cap could return.
        for (int i = 0; i < 2; ++i) {
            const char * text = transcribe_batch_full_text(s, i);
            CHECK(text != nullptr && text[0] != '\0');
        }
        const char * long_text = transcribe_batch_full_text(s, 1);
        CHECK(long_text != nullptr && std::strlen(long_text) > 1000);

        CHECK(transcribe_was_truncated(s) == false);
    }

    transcribe_session_free(s);
    transcribe_model_free(model);

    if (g_failures > 0) {
        std::fprintf(stderr, "qwen3_asr_long_form: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "qwen3_asr_long_form: ok\n");
    return EXIT_SUCCESS;
}
