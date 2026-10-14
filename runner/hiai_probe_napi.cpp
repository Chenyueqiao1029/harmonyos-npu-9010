/*
 * LingShu - HiAI Foundation reachability probe (in-app N-API module).
 *
 * WHY THIS EXISTS
 * ---------------
 * The device answers canIUse("SystemCapability.AI.HiAIFoundation") = true while
 * canIUse("SystemCapability.AI.CANN.LLMEngine") = false. So HiAI Foundation IS
 * an app-facing capability on this Kirin 9010 phone, even though CANN LM Engine
 * (the Kirin X90-only product) is not.
 *
 * The earlier failure used the WRONG model format: MindSpore Lite `.ms` with
 * Context{target:['nnrt']}. The official HiAI path is different:
 *     select device named "HIAI_F"
 *     -> OH_NNCompilation_ConstructWithOfflineModelBuffer(hiai.om)
 *     -> HMS_HiAIOptions_SetModelDeviceOrder(...)   (NPU first)
 *     -> Build / Executor / RunSync
 * Huawei's own codelab states the permissions involved: none.
 *
 * This probe answers, without needing a real .om yet:
 *   1. does libhiai_foundation.so load, and do its symbols resolve?
 *   2. can we enumerate NNRt devices from native?
 *   3. if we hand ConstructWithOfflineModelBuffer a deliberately bogus buffer,
 *      WHAT ERROR comes back?  (a real "invalid model" error means the call
 *      got through to the vendor library; an SELinux/hdi failure would show up
 *      earlier and differently)
 *
 * Every step is reported as text; nothing throws across the N-API boundary.
 */
#include "napi/native_api.h"

#include <dlfcn.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

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

/** dlsym a symbol out of libhiai_foundation.so; report whether it exists. */
void ProbeSymbol(void *h, const char *name, std::vector<std::string> &found) {
    void *p = dlsym(h, name);
    found.push_back(std::string(name) + "=" + (p == nullptr ? "MISSING" : "OK"));
}

std::string RunProbe() {
    g_report.clear();
    Say("=== HiAI Foundation reachability probe ===");

    // ---- 1. load libhiai_foundation.so -------------------------------------
    void *hiai = dlopen("libhiai_foundation.so", RTLD_NOW | RTLD_LOCAL);
    if (hiai == nullptr) {
        const char *e = dlerror();
        Say(std::string("dlopen(libhiai_foundation.so) FAILED: ") + (e == nullptr ? "(null)" : e));
    } else {
        Say("dlopen(libhiai_foundation.so) OK");
        std::vector<std::string> found;
        ProbeSymbol(hiai, "HMS_HiAIOptions_SetBandMode", found);
        ProbeSymbol(hiai, "HMS_HiAIOptions_GetBandMode", found);
        ProbeSymbol(hiai, "HMS_HiAIOptions_SetModelDeviceOrder", found);
        ProbeSymbol(hiai, "HMS_HiAIOptions_SetFormatMode", found);
        ProbeSymbol(hiai, "HMS_HiAIOptions_SetDynamicShapeStatus", found);
        ProbeSymbol(hiai, "HMS_LLMEngineContext_CreateFromContextJson", found);
        ProbeSymbol(hiai, "HMS_LLMEngineExecutor_CreateFromExecutorJson", found);
        ProbeSymbol(hiai, "HMS_LLMEngineExecutor_Generate", found);
        std::string line = "  symbols:";
        for (size_t i = 0; i < found.size(); ++i) { line += " " + found[i]; }
        Say(line);
    }

    // ---- 2. libcann_llm_engine.so ------------------------------------------
    void *cann = dlopen("libcann_llm_engine.so", RTLD_NOW | RTLD_LOCAL);
    Say(std::string("dlopen(libcann_llm_engine.so) ") + (cann == nullptr ? "FAILED (expected: syscap=false)" : "OK"));

    // ---- 3. native device enumeration --------------------------------------
    const size_t *devices = nullptr;
    uint32_t count = 0;
    OH_NN_ReturnCode r = OH_NNDevice_GetAllDevicesID(&devices, &count);
    if (r != OH_NN_SUCCESS || devices == nullptr) {
        Say(std::string("native getAllDevicesID=") + RetName(r));
    } else {
        Say("native devices=" + std::to_string(count));
        for (uint32_t i = 0; i < count; ++i) {
            const char *name = nullptr;
            OH_NN_DeviceType type = OH_NN_OTHERS;
            OH_NNDevice_GetName(devices[i], &name);
            OH_NNDevice_GetType(devices[i], &type);
            Say(std::string("  #") + std::to_string(i) +
                " id=" + std::to_string(devices[i]) +
                " type=" + std::to_string(static_cast<int>(type)) +
                " name=" + (name == nullptr ? "(null)" : name));
        }
    }

    // ---- 4. ConstructWithOfflineModelBuffer with a bogus buffer ------------
    // A real "bad model" error means the call reached the vendor library.
    std::vector<uint8_t> bogus(4096, 0xAB);
    OH_NNCompilation *comp = OH_NNCompilation_ConstructWithOfflineModelBuffer(bogus.data(), bogus.size());
    if (comp == nullptr) {
        Say("ConstructWithOfflineModelBuffer(bogus) -> nullptr (the library rejected the buffer)");
    } else {
        Say("ConstructWithOfflineModelBuffer(bogus) -> non-null handle  <-- reachable");
        if (devices != nullptr && count > 0) {
            OH_NN_ReturnCode r2 = OH_NNCompilation_SetDevice(comp, devices[0]);
            Say(std::string("  SetDevice(dev0)=") + RetName(r2));
        }
        OH_NN_ReturnCode r3 = OH_NNCompilation_Build(comp);
        Say(std::string("  Build=") + RetName(r3));
        OH_NNCompilation_Destroy(&comp);
    }

    // ---- 5. same via file path (the .om route) -----------------------------
    OH_NNCompilation *cf = OH_NNCompilation_ConstructWithOfflineModelFile("/data/local/tmp/__nonexistent__.om");
    Say(std::string("ConstructWithOfflineModelFile(missing) -> ") + (cf == nullptr ? "nullptr" : "non-null"));
    if (cf != nullptr) { OH_NNCompilation_Destroy(&cf); }

    Say("=== done ===");
    return g_report;
}

napi_value Run(napi_env env, napi_callback_info info) {
    (void)info;
    std::string out;
    try { out = RunProbe(); } catch (...) { out = "probe threw"; }
    napi_value v;
    napi_create_string_utf8(env, out.c_str(), out.size(), &v);
    return v;
}

}  // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        { "runHiaiProbe", nullptr, Run, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module g_hiai_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "hiai_probe",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterHiaiProbeModule(void) {
    napi_module_register(&g_hiai_module);
}
