// llama-chat: standalone low-RAM chat engine
//
// Copies how llama.cpp does chat (model chat template via common_chat_templates,
// classic conversation REPL, common_sampler defaults) but links no server, no
// fit probe, no prompt cache and no context checkpoints. Each turn renders the
// message list with the model template and feeds only the new delta into the KV,
// so RAM stays at one context + one KV instead of the server layers.

#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "chat.h"

#include <chrono>
#include <clocale>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#endif

static volatile sig_atomic_t g_interrupted = 0;
static bool g_generating = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_generating) {
            // stop the current reply, keep the chat alive
            g_interrupted = 1;
        } else {
            console::cleanup();
            LOG("\nInterrupted by user\n");
            _exit(130);
        }
    }
}
#endif

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n  chat:  %s -m your_model.gguf -c 8192 --ultra-low\n", argv[0]);
    LOG("\n");
}

// satisfies -Wmissing-declarations
int llama_chat(int argc, char ** argv);

int llama_chat(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    if (params.embedding) {
        LOG_ERR("llama-chat: embedding mode is not supported, this tool only does chat\n");
        return 1;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("llama-chat: minimum context size is 8, using minimum size\n");
        params.n_ctx = 8;
    }

    // this tool only does chat, raw completion lives in llama-completion
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_DISABLED) {
        LOG_ERR("llama-chat: conversation mode was disabled (-no-cnv), this tool only does chat\n");
        return 1;
    }

    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    mparams.no_alloc = false;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (!model) {
        LOG_ERR("llama-chat: unable to load model from %s\n", params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = common_context_params_to_llama(params);

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        LOG_ERR("llama-chat: unable to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    {
        int nt = params.cpuparams.n_threads;
        if (nt <= 0) {
            nt = std::max(1u, std::thread::hardware_concurrency());
        }
        int ntb = params.cpuparams_batch.n_threads <= 0 ? nt : params.cpuparams_batch.n_threads;
        llama_set_n_threads(ctx, nt, ntb);
    }

    common_sampler * smpl = common_sampler_init(model, params.sampling);

    auto tmpls = common_chat_templates_init(model, params.chat_template);
    const bool has_tmpl = common_chat_templates_was_explicit(tmpls.get());
    if (!has_tmpl) {
        LOG_WRN("llama-chat: model has no chat template, using fallback, replies may be worse\n");
    } else {
        LOG_INF("llama-chat: chat template example:\n%s\n",
            common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());
    }

    LOG_INF("\n%s\n\n", common_params_get_system_info(params).c_str());

    llama_perf_context_reset(ctx);

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    {
        struct sigaction sa;
        sa.sa_handler = sigint_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        sigaction(SIGINT, &sa, NULL);
    }
#endif

    // pin the template time so renders stay prefix-stable across turns
    const auto fixed_now = std::chrono::system_clock::now();

    std::vector<common_chat_msg> msgs;
    if (!params.system_prompt.empty()) {
        common_chat_msg m;
        m.role    = "system";
        m.content = params.system_prompt;
        msgs.push_back(m);
    }

    // rendered text of msgs (gen prompt off) as currently stored in the KV
    std::string prev_render;

    auto render = [&](bool gen_prompt) {
        common_chat_templates_inputs inputs;
        inputs.messages               = msgs;
        inputs.use_jinja              = params.use_jinja;
        inputs.add_generation_prompt  = gen_prompt;
        inputs.force_pure_content     = params.force_pure_content_parser;
        inputs.chat_template_kwargs   = params.default_template_kwargs;
        inputs.now                    = fixed_now;
        return common_chat_templates_apply(tmpls.get(), inputs).prompt;
    };

    auto n_kv = [&]() -> int {
        return (int) llama_memory_seq_pos_max(mem, 0) + 1;
    };

    const int n_ctx = llama_n_ctx(ctx);
    const int limit = n_ctx - 4;

    std::function<bool(const std::string &)> do_turn = [&](const std::string & user_text) {
        common_chat_msg um;
        um.role    = "user";
        um.content = user_text;
        msgs.push_back(um);

        // render and tokenize: feed only the new delta when the render is
        // prefix-stable, else clear the KV and re-encode (trim oldest msgs)
        std::vector<llama_token> toks;
        bool reset = false;
        while (true) {
            const std::string full = render(true);
            const int cur_kv = n_kv();
            if (cur_kv > 0 && full.size() >= prev_render.size() &&
                    full.compare(0, prev_render.size(), prev_render) == 0) {
                toks = common_tokenize(ctx, full.substr(prev_render.size()), false, true);
                if (cur_kv + (int) toks.size() <= limit) {
                    break;
                }
            }
            toks = common_tokenize(ctx, full, true, true);
            if ((int) toks.size() <= limit) {
                reset = true;
                break;
            }
            auto it = msgs.begin();
            if (it != msgs.end() && it->role == "system") {
                ++it;
            }
            if (it == msgs.end() || msgs.size() <= 1) {
                LOG_ERR("llama-chat: single turn does not fit in context\n");
                msgs.pop_back();
                return false;
            }
            LOG("\n<<context full, dropped oldest message>>\n");
            msgs.erase(it);
        }
        if (reset) {
            llama_memory_clear(mem, true);
        }

        common_sampler_reset(smpl);

        size_t off = 0;
        while (off < toks.size()) {
            const size_t n = std::min((size_t) params.n_batch, toks.size() - off);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(toks.data()) + off, (int) n);
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("llama-chat: decode failed\n");
                return false;
            }
            for (size_t i = off; i < off + n; i++) {
                common_sampler_accept(smpl, toks[i], false);
            }
            off += n;
        }

        std::string response;

        g_generating  = true;
        g_interrupted = 0;

        bool ok = true;
        int  n_gen = 0;
        while (true) {
            if (g_interrupted) {
                // close the assistant turn with an EOT so history stays valid
                llama_token eot = llama_vocab_eot(vocab);
                if (eot == LLAMA_TOKEN_NULL) {
                    eot = llama_vocab_eos(vocab);
                }
                llama_batch b = llama_batch_get_one(&eot, 1);
                if (llama_decode(ctx, b) != 0) {
                    ok = false;
                }
                LOG("\n<<interrupted>>\n");
                break;
            }
            if (params.n_predict >= 0 && n_gen >= params.n_predict) {
                break;
            }
            const llama_token id = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, id, true);
            // decode the sampled token too, so the KV matches the rendered
            // history (this includes the EOG token closing the turn)
            llama_token nid = id;
            llama_batch b = llama_batch_get_one(&nid, 1);
            if (llama_decode(ctx, b) != 0) {
                LOG_ERR("llama-chat: decode failed\n");
                ok = false;
                break;
            }
            if (llama_vocab_is_eog(vocab, id)) {
                break;
            }
            const std::string piece = common_token_to_piece(ctx, id, false);
            LOG("%s", piece.c_str());
            response += piece;
            n_gen++;
        }

        g_generating = false;

        common_chat_msg am;
        am.role    = "assistant";
        am.content = response;
        msgs.push_back(am);
        prev_render = render(false);

        LOG("\n");
        return ok;
    };

    if (!params.prompt.empty()) {
        if (!do_turn(params.prompt)) {
            return 1;
        }
        if (params.single_turn) {
            common_perf_print(ctx, smpl);
            common_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 0;
        }
    }

    LOG("== chat started, /exit to quit, /clear to reset ==\n\n");

    while (true) {
        LOG("> ");
        console::set_display(DISPLAY_TYPE_USER_INPUT);

        std::string buffer;
        std::string line;
        bool another = true;
        do {
            another = console::readline(line, params.multiline_input);
            buffer += line;
        } while (another);

        console::set_display(DISPLAY_TYPE_RESET);

        if (buffer.empty()) {
            LOG("EOF by user\n");
            break;
        }
        if (buffer.back() == '\n') {
            buffer.pop_back();
        }
        if (buffer.empty()) {
            continue;
        }
        if (buffer == "/exit" || buffer == "/quit") {
            break;
        }
        if (buffer == "/clear") {
            msgs.clear();
            prev_render.clear();
            llama_memory_clear(mem, true);
            if (!params.system_prompt.empty()) {
                common_chat_msg m;
                m.role    = "system";
                m.content = params.system_prompt;
                msgs.push_back(m);
            }
            common_sampler_reset(smpl);
            continue;
        }

        if (!do_turn(buffer)) {
            return 1;
        }
    }

    LOG("\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
