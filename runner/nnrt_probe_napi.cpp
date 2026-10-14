/*
 * LingShu - NNRt probe as an in-app N-API module.
 *
 * Why in-app instead of a standalone executable: the standalone binary gets
 * "Permission denied" when executed from the hdc shell (uid=2000 shell,
 * SELinux domain u:r:sh:s0) even though the file mode is 0755 and /data is not
 * mounted noexec. Acceptance must happen inside the application process anyway
 * (codex's point), so this module is the real probe.
 *
 * Exported: runProbe() -> string  (the full human-readable report)
 *
 * The logic is identical to nrt_probe.cpp: build a zero-weight Add model in
 * memory, query op support, compile and run it on each NNRt device, check the
 * numbers. Two .size semantics are both tried (element count per the header vs
 * the byte count written in the official guideline).
 */
#include "napi/native_api.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "neural_network_runtime/neural_network_runtime.h"

namespace {

std::string g_report;

void Say(const std::string &line) {
    g_report += line;
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

OH_NNModel *BuildAddModel(bool idxAsCount) {
    OH_NNModel *model = OH_NNModel_Construct();
    if (model == nullptr) { Say("  [model] Construct nullptr"); return nullptr; }
    int32_t dims[4] = {1, 1, 1, 4};

    auto addTensor = [&](OH_NN_DataType dt, const int32_t *shape, size_t rank,
                         OH_NN_TensorType type, uint32_t index) -> bool {
        NN_TensorDesc *desc = OH_NNTensorDesc_Create();
        if (desc == nullptr) { Say("  [model] TensorDesc_Create failed"); return false; }
        OH_NN_ReturnCode r = OH_NNTensorDesc_SetShape(desc, shape, rank);
        if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SetShape: ") + RetName(r)); return false; }
        r = OH_NNTensorDesc_SetDataType(desc, dt);
        if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SetDataType: ") + RetName(r)); return false; }
        r = OH_NNTensorDesc_SetFormat(desc, OH_NN_FORMAT_NONE);
        if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SetFormat: ") + RetName(r)); return false; }
        r = OH_NNModel_AddTensorToModel(model, desc);
        if (r != OH_NN_SUCCESS) { Say(std::string("  [model] AddTensorToModel: ") + RetName(r)); return false; }
        r = OH_NNModel_SetTensorType(model, index, type);
        if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SetTensorType: ") + RetName(r)); return false; }
        return true;
    };

    if (!addTensor(OH_NN_FLOAT32, dims, 4, OH_NN_TENSOR, 0)) return nullptr;
    if (!addTensor(OH_NN_FLOAT32, dims, 4, OH_NN_TENSOR, 1)) return nullptr;
    int32_t actDim = 1;
    if (!addTensor(OH_NN_INT8, &actDim, 1, OH_NN_ADD_ACTIVATIONTYPE, 2)) return nullptr;
    if (!addTensor(OH_NN_FLOAT32, dims, 4, OH_NN_TENSOR, 3)) return nullptr;

    int8_t activation = OH_NN_FUSED_NONE;
    OH_NN_ReturnCode r = OH_NNModel_SetTensorData(model, 2, &activation, sizeof(int8_t));
    if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SetTensorData: ") + RetName(r)); return nullptr; }

    uint32_t inputIdx[2] = {0, 1};
    uint32_t paramIdx = 2;
    uint32_t outputIdx = 3;
    OH_NN_UInt32Array params  = {&paramIdx,  idxAsCount ? 1u : 1u * 4u};
    OH_NN_UInt32Array inputs  = {inputIdx,  idxAsCount ? 2u : 2u * 4u};
    OH_NN_UInt32Array outputs = {&outputIdx, idxAsCount ? 1u : 1u * 4u};

    r = OH_NNModel_AddOperation(model, OH_NN_OPS_ADD, &params, &inputs, &outputs);
    if (r != OH_NN_SUCCESS) { Say(std::string("  [model] AddOperation: ") + RetName(r)); OH_NNModel_Destroy(&model); return nullptr; }
    r = OH_NNModel_SpecifyInputsAndOutputs(model, &inputs, &outputs);
    if (r != OH_NN_SUCCESS) { Say(std::string("  [model] SpecifyIO: ") + RetName(r)); OH_NNModel_Destroy(&model); return nullptr; }
    r = OH_NNModel_Finish(model);
    if (r != OH_NN_SUCCESS) { Say(std::string("  [model] Finish: ") + RetName(r)); OH_NNModel_Destroy(&model); return nullptr; }
    return model;
}

void RunOnDevice(size_t deviceId, const std::string &tag, bool idxAsCount) {
    Say("---- device=" + tag + " idxAs" + (idxAsCount ? "Count" : "Bytes") + " ----");
    OH_NNModel *model = BuildAddModel(idxAsCount);
    if (model == nullptr) { Say("  finish=model_build_failed"); return; }

    uint32_t opCount = 0;
    const bool *supported = nullptr;
    OH_NN_ReturnCode r = OH_NNModel_GetAvailableOperations(model, deviceId, &supported, &opCount);
    if (r != OH_NN_SUCCESS) {
        Say(std::string("  supported=query_failed(") + RetName(r) + ")");
    } else if (opCount == 0 || supported == nullptr) {
        Say("  supported=no_ops_returned");
    } else {
        uint32_t n = 0;
        for (uint32_t i = 0; i < opCount; ++i) { if (supported[i]) n++; }
        Say("  supported=" + std::to_string(n) + "/" + std::to_string(opCount));
    }

    OH_NNCompilation *comp = OH_NNCompilation_Construct(model);
    if (comp == nullptr) { Say("  build=construct_nullptr"); OH_NNModel_Destroy(&model); return; }
    r = OH_NNCompilation_SetDevice(comp, deviceId);
    Say(std::string("  setDevice=") + RetName(r));
    if (r != OH_NN_SUCCESS) { OH_NNCompilation_Destroy(&comp); OH_NNModel_Destroy(&model); return; }
    r = OH_NNCompilation_Build(comp);
    Say(std::string("  build=") + RetName(r));
    if (r != OH_NN_SUCCESS) { OH_NNCompilation_Destroy(&comp); OH_NNModel_Destroy(&model); return; }

    OH_NNExecutor *exec = OH_NNExecutor_Construct(comp);
    if (exec == nullptr) { Say("  executor=nullptr"); OH_NNCompilation_Destroy(&comp); OH_NNModel_Destroy(&model); return; }
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&comp);

    size_t inCount = 0, outCount = 0;
    OH_NNExecutor_GetInputCount(exec, &inCount);
    OH_NNExecutor_GetOutputCount(exec, &outCount);
    Say("  in=" + std::to_string(inCount) + " out=" + std::to_string(outCount));

    std::vector<NN_Tensor *> ins(inCount, nullptr), outs(outCount, nullptr);
    std::vector<NN_TensorDesc *> inDescs(inCount, nullptr), outDescs(outCount, nullptr);
    for (size_t i = 0; i < inCount; ++i) {
        inDescs[i] = OH_NNExecutor_CreateInputTensorDesc(exec, i);
        ins[i] = OH_NNTensor_Create(deviceId, inDescs[i]);
    }
    for (size_t i = 0; i < outCount; ++i) {
        outDescs[i] = OH_NNExecutor_CreateOutputTensorDesc(exec, i);
        outs[i] = OH_NNTensor_Create(deviceId, outDescs[i]);
    }

    const float A[4] = {1.f, 2.f, 3.f, 4.f};
    const float B[4] = {10.f, 20.f, 30.f, 40.f};
    if (inCount >= 2 && ins[0] != nullptr && ins[1] != nullptr) {
        float *p0 = reinterpret_cast<float *>(OH_NNTensor_GetDataBuffer(ins[0]));
        float *p1 = reinterpret_cast<float *>(OH_NNTensor_GetDataBuffer(ins[1]));
        if (p0 != nullptr) { for (int i = 0; i < 4; ++i) p0[i] = A[i]; }
        if (p1 != nullptr) { for (int i = 0; i < 4; ++i) p1[i] = B[i]; }
    }

    r = OH_NNExecutor_RunSync(exec, ins.data(), inCount, outs.data(), outCount);
    Say(std::string("  run=") + RetName(r));
    if (r == OH_NN_SUCCESS && outCount >= 1 && outs[0] != nullptr) {
        float *y = reinterpret_cast<float *>(OH_NNTensor_GetDataBuffer(outs[0]));
        if (y != nullptr) {
            char buf[160];
            snprintf(buf, sizeof(buf), "  output=%.0f,%.0f,%.0f,%.0f", y[0], y[1], y[2], y[3]);
            bool pass = (y[0] == 11.f && y[1] == 22.f && y[2] == 33.f && y[3] == 44.f);
            Say(std::string(buf) + std::string(" check=") + (pass ? "PASS" : "FAIL"));
        } else {
            Say("  output=(null buffer)");
        }
    }

    for (size_t i = 0; i < inCount; ++i) {
        if (ins[i] != nullptr) OH_NNTensor_Destroy(&ins[i]);
        if (inDescs[i] != nullptr) OH_NNTensorDesc_Destroy(&inDescs[i]);
    }
    for (size_t i = 0; i < outCount; ++i) {
        if (outs[i] != nullptr) OH_NNTensor_Destroy(&outs[i]);
        if (outDescs[i] != nullptr) OH_NNTensorDesc_Destroy(&outDescs[i]);
    }
    OH_NNExecutor_Destroy(&exec);
}

std::string RunAllDevices() {
    g_report.clear();
    Say("=== NNRt in-app probe ===");

    const size_t *devices = nullptr;
    uint32_t count = 0;
    OH_NN_ReturnCode r = OH_NNDevice_GetAllDevicesID(&devices, &count);
    if (r != OH_NN_SUCCESS || devices == nullptr || count == 0) {
        Say(std::string("devices=0 get=") + RetName(r));
        Say("RESULT=NO_DEVICE");
        return g_report;
    }
    Say("devices=" + std::to_string(count));
    std::vector<size_t> ids;
    for (uint32_t i = 0; i < count; ++i) {
        const char *name = nullptr;
        OH_NN_DeviceType type = OH_NN_OTHERS;
        OH_NNDevice_GetName(devices[i], &name);
        OH_NNDevice_GetType(devices[i], &type);
        ids.push_back(devices[i]);
        Say(std::string("  #") + std::to_string(i) +
            " id=" + std::to_string(devices[i]) +
            " type=" + std::to_string(static_cast<int>(type)) +
            " name=" + (name == nullptr ? "(null)" : name));
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        RunOnDevice(ids[i], "dev" + std::to_string(i), true);
    }
    RunOnDevice(ids[0], "dev0", false);
    Say("=== done ===");
    return g_report;
}

napi_value RunProbe(napi_env env, napi_callback_info info) {
    (void)info;
    std::string report;
    try {
        report = RunAllDevices();
    } catch (...) {
        report = "probe threw";
    }
    napi_value out;
    napi_create_string_utf8(env, report.c_str(), report.size(), &out);
    return out;
}

}  /* namespace */

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc[] = {
        { "runProbe", nullptr, RunProbe, nullptr, nullptr, nullptr, napi_default, nullptr },
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module g_module = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "nnrt_probe",
    .nm_priv = nullptr,
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterNnrtProbeModule(void) {
    napi_module_register(&g_module);
}
