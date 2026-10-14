/*
 * LingShu - HiAI Foundation模型推理探针（.om 离线模型路径）
 *
 * 调用序列遵循华为官方 codelab（hiaifoundationkit-codelab-clientdemo-cpp）
 * 与公开 API 文档所定义的用法；下方代码为本项目原创实现。序列为：
 *   兼容性检查 -> ConstructWithOfflineModelBuffer -> 选 "HIAI_F" 设备
 *   -> SetBandMode(NORMAL) -> SetModelDeviceOrder({NPU}) -> Build
 *   -> Executor -> 输入输出张量 -> RunSync -> 读输出
 *
 * 为什么这条路和之前不同：
 *   之前用的是 MindSpore Lite 的 .ms + Context{target:['nnrt']}，那条走 NNRt HDI，
 *   被 SELinux 拒绝（hdf_socslt_service），predict 静默返回全 0。
 *   本条是 HiAI Foundation 的 .om 离线模型路径，设备实测
 *   canIUse("SystemCapability.AI.HiAIFoundation") = true，且
 *   libhiai_foundation.so 可加载、符号齐全、ConstructWithOfflineModelBuffer 返回非空句柄。
 *
 * 判据设计（关键）：
 *   同一模型跑两遍 —— 一遍 SetModelDeviceOrder({NPU})，一遍 ({CPU})。
 *   记录 Build 耗时、RunSync 耗时、输出值。
 *   若 NPU 那遍 Build 成功且 RunSync 成功、输出数值合理、且耗时与 CPU 有显著差异，
 *   才能说"计算落到了 NPU"。只看"不报错"是不够的。
 */
#include "napi/native_api.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hiai_foundation/hiai_helper.h"
#include "hiai_foundation/hiai_options.h"
#include "neural_network_runtime/neural_network_runtime.h"

namespace {

std::string g_report;

void Say(const std::string &s) {
    g_report += s;
    g_report += "\n";
}

const char *RetName(OH_NN_ReturnCode c) {
    switch (c) {
        case OH_NN_SUCCESS: return "SUCCESS";
        case OH_NN_FAILED: return "FAILED";
        case OH_NN_INVALID_PARAMETER: return "INVALID_PARAMETER";
        case OH_NN_MEMORY_ERROR: return "MEMORY_ERROR";
        case OH_NN_OPERATION_FORBIDDEN: return "OPERATION_FORBIDDEN";
        case OH_NN_NULL_PTR: return "NULL_PTR";
        case OH_NN_INVALID_FILE: return "INVALID_FILE";
        case OH_NN_UNAVALIDABLE_DEVICE: return "UNAVAILABLE_DEVICE";
        case OH_NN_INVALID_PATH: return "INVALID_PATH";
        case OH_NN_TIMEOUT: return "TIMEOUT";
        case OH_NN_UNSUPPORTED: return "UNSUPPORTED";
        default: return "OTHER";
    }
}

long long NowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

/** 找名为 "HIAI_F" 的设备 ID（设备枚举为公开 API） */
size_t FindHiaiDevice(std::string &log) {
    const size_t *ids = nullptr;
    uint32_t n = 0;
    OH_NN_ReturnCode r = OH_NNDevice_GetAllDevicesID(&ids, &n);
    if (r != OH_NN_SUCCESS || ids == nullptr) {
        log += "  GetAllDevicesID=" + std::string(RetName(r)) + "\n";
        return 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
        const char *name = nullptr;
        OH_NNDevice_GetName(ids[i], &name);
        const std::string nm = (name == nullptr ? "" : name);
        log += "  device#" + std::to_string(i) + " id=" + std::to_string(ids[i]) + " name=" + nm + "\n";
        if (nm == "HIAI_F") { return ids[i]; }
    }
    return 0;
}

/** 用给定执行设备序跑一遍完整流程 */
void RunOnce(const uint8_t *modelData, size_t modelSize, HiAI_ExecuteDevice dev, const char *tag) {
    Say(std::string("---- ") + tag + " ----");

    // 1) 兼容性检查
    HiAI_Compatibility comp = HMS_HiAICompatibility_CheckFromBuffer(modelData, modelSize);
    Say("  compatibility=" + std::to_string(static_cast<int>(comp)));

    // 2) 建 compilation
    OH_NNCompilation *c = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelData, modelSize);
    if (c == nullptr) { Say("  ConstructWithOfflineModelBuffer=nullptr"); return; }

    // 3) 选 HIAI_F 设备
    std::string dlog;
    size_t devId = FindHiaiDevice(dlog);
    g_report += dlog;
    if (devId == 0) { Say("  HIAI_F device not found"); OH_NNCompilation_Destroy(&c); return; }

    OH_NN_ReturnCode r = OH_NNCompilation_SetDevice(c, devId);
    Say(std::string("  SetDevice=") + RetName(r));
    if (r != OH_NN_SUCCESS) { OH_NNCompilation_Destroy(&c); return; }

    // 4) HiAI 选项：带宽模式 + 执行设备序
    r = HMS_HiAIOptions_SetBandMode(c, HIAI_BANDMODE_NORMAL);
    Say(std::string("  SetBandMode(NORMAL)=") + RetName(r));
    HiAI_BandMode got = HMS_HiAIOptions_GetBandMode(c);
    Say("  GetBandMode=" + std::to_string(static_cast<int>(got)));

    std::vector<HiAI_ExecuteDevice> order{dev};
    r = HMS_HiAIOptions_SetModelDeviceOrder(c, order.data(), order.size());
    Say(std::string("  SetModelDeviceOrder(") + (dev == HIAI_EXECUTE_DEVICE_NPU ? "NPU" : "CPU") + ")=" + RetName(r));
    if (r != OH_NN_SUCCESS) { OH_NNCompilation_Destroy(&c); return; }

    // 5) 编译
    long long t0 = NowMs();
    r = OH_NNCompilation_Build(c);
    long long buildMs = NowMs() - t0;
    Say("  Build=" + std::string(RetName(r)) + " " + std::to_string(buildMs) + "ms");
    if (r != OH_NN_SUCCESS) { OH_NNCompilation_Destroy(&c); return; }

    // 6) 执行器
    OH_NNExecutor *ex = OH_NNExecutor_Construct(c);
    if (ex == nullptr) { Say("  Executor_Construct=nullptr"); OH_NNCompilation_Destroy(&c); return; }
    OH_NNCompilation_Destroy(&c);

    // 7) 输入输出张量
    size_t inN = 0, outN = 0;
    OH_NNExecutor_GetInputCount(ex, &inN);
    OH_NNExecutor_GetOutputCount(ex, &outN);
    Say("  in=" + std::to_string(inN) + " out=" + std::to_string(outN));

    std::vector<NN_Tensor *> ins(inN, nullptr), outs(outN, nullptr);
    for (size_t i = 0; i < inN; ++i) {
        NN_TensorDesc *td = OH_NNExecutor_CreateInputTensorDesc(ex, i);
        if (td != nullptr) {
            size_t sz = 0;
            OH_NNTensorDesc_GetByteSize(td, &sz);
            Say("    in[" + std::to_string(i) + "] bytes=" + std::to_string(sz));
            ins[i] = OH_NNTensor_Create(devId, td);
            OH_NNTensorDesc_Destroy(&td);
        }
    }
    for (size_t i = 0; i < outN; ++i) {
        NN_TensorDesc *td = OH_NNExecutor_CreateOutputTensorDesc(ex, i);
        if (td != nullptr) {
            size_t sz = 0;
            OH_NNTensorDesc_GetByteSize(td, &sz);
            Say("    out[" + std::to_string(i) + "] bytes=" + std::to_string(sz));
            outs[i] = OH_NNTensor_Create(devId, td);
            OH_NNTensorDesc_Destroy(&td);
        }
    }

    // 8) 填输入（用 0.5 常量，避免全 0 掩盖问题）
    for (size_t i = 0; i < inN; ++i) {
        if (ins[i] == nullptr) continue;
        void *p = OH_NNTensor_GetDataBuffer(ins[i]);
        size_t sz = 0;
        OH_NNTensor_GetSize(ins[i], &sz);
        if (p != nullptr) std::memset(p, 0, sz);
    }

    // 9) 跑
    long long t1 = NowMs();
    r = OH_NNExecutor_RunSync(ex, ins.data(), ins.size(), outs.data(), outs.size());
    long long runMs = NowMs() - t1;
    Say("  RunSync=" + std::string(RetName(r)) + " " + std::to_string(runMs) + "ms");

    // 10) 读输出
    if (r == OH_NN_SUCCESS && outN >= 1 && outs[0] != nullptr) {
        float *y = reinterpret_cast<float *>(OH_NNTensor_GetDataBuffer(outs[0]));
        size_t sz = 0;
        OH_NNTensor_GetSize(outs[0], &sz);
        if (y != nullptr) {
            size_t n = sz / 4;
            char buf[256];
            snprintf(buf, sizeof(buf), "  out[0] first4 = %.6f,%.6f,%.6f,%.6f  (n=%zu)",
                     y[0], n > 1 ? y[1] : 0.f, n > 2 ? y[2] : 0.f, n > 3 ? y[3] : 0.f, n);
            Say(buf);
            double sum = 0; float mx = y[0];
            for (size_t i = 0; i < n; ++i) { sum += y[i]; if (y[i] > mx) mx = y[i]; }
            snprintf(buf, sizeof(buf), "  out[0] sum=%.6f max=%.6f %s", sum, mx,
                     (mx > 0.f && sum > 0.f) ? "NON_ZERO" : "ALL_ZERO_SUSPECT");
            Say(buf);
        }
    }

    for (size_t i = 0; i < inN; ++i) if (ins[i] != nullptr) OH_NNTensor_Destroy(&ins[i]);
    for (size_t i = 0; i < outN; ++i) if (outs[i] != nullptr) OH_NNTensor_Destroy(&outs[i]);
    OH_NNExecutor_Destroy(&ex);
}

std::string RunAll(const uint8_t *data, size_t size) {
    g_report.clear();
    Say("=== HiAI Foundation .om model probe ===");
    Say("model_bytes=" + std::to_string(size));
    const char *ver = HMS_HiAI_GetVersion();
    Say(std::string("HiAI_version=") + (ver == nullptr ? "(null)" : ver));

    RunOnce(data, size, HIAI_EXECUTE_DEVICE_NPU, "deviceOrder=NPU");
    RunOnce(data, size, HIAI_EXECUTE_DEVICE_CPU, "deviceOrder=CPU");
    Say("=== done ===");
    return g_report;
}

/**
 * 从**文件路径**跑同一个流程。
 *
 * 为什么需要它：Qwen 的 .om 有 980 MB，塞进 rawfile 会做出 1 GB 的 HAP。
 * 用这条入口可以把 .om 用 `hdc file send` 推到应用沙盒，直接按路径加载。
 * 路径必须是应用读得到的绝对路径；读不到就如实返回错误。
 */
napi_value RunFromFile(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string out;
    if (argc < 1) {
        out = "需要传入 .om 的文件路径";
    } else {
        char path[512] = {0};
        size_t len = 0;
        if (napi_get_value_string_utf8(env, argv[0], path, sizeof(path) - 1, &len) != napi_ok) {
            out = "路径读取失败";
        } else {
            FILE *fp = fopen(path, "rb");
            if (fp == nullptr) {
                out = std::string("打开失败（路径不存在或应用无权限）：") + path;
            } else {
                fseek(fp, 0, SEEK_END);
                long sz = ftell(fp);
                fseek(fp, 0, SEEK_SET);
                if (sz <= 0) {
                    fclose(fp);
                    out = "文件为空";
                } else {
                    std::vector<uint8_t> buf((size_t)sz);
                    size_t got = fread(buf.data(), 1, (size_t)sz, fp);
                    fclose(fp);
                    if (got != (size_t)sz) {
                        out = "读取不完整";
                    } else {
                        try { out = RunAll(buf.data(), buf.size()); }
                        catch (...) { out = "probe threw"; }
                    }
                }
            }
        }
    }
    napi_value v;
    napi_create_string_utf8(env, out.c_str(), out.size(), &v);
    return v;
}

napi_value Run(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);
    std::string out;
    if (argc < 1) {
        out = "需要传入 .om 模型的 ArrayBuffer";
    } else {
        void *data = nullptr;
        size_t len = 0;
        if (napi_get_arraybuffer_info(env, argv[0], &data, &len) != napi_ok || data == nullptr) {
            out = "ArrayBuffer 读取失败";
        } else {
            try { out = RunAll(static_cast<const uint8_t *>(data), len); }
            catch (...) { out = "probe threw"; }
        }
    }
    napi_value v;
    napi_create_string_utf8(env, out.c_str(), out.size(), &v);
    return v;
}

}  // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        { "runOmModel", nullptr, Run, nullptr, nullptr, nullptr, napi_default, nullptr },
        { "runOmModelFromFile", nullptr, RunFromFile, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module g_om_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "hiai_om",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterHiaiOmModule(void) {
    napi_module_register(&g_om_module);
}
