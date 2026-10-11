/*
Copyright (C) 2026  Holy Wu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSVulkan4.h>

#include "dct_codelets.h"
#include "shaders.h"

namespace {

using namespace std::string_literals;
using namespace std::string_view_literals;

constexpr int errorSize = 1024;
constexpr int maxPlanes = 3;
constexpr int maxRadius = 16;
constexpr int maxBlockSize = 64;
constexpr int maxGroupSize = 256;
constexpr int maxDimension = 65536; // block positions are packed as two 16-bit coordinates
constexpr int imageTile = 16;
constexpr int bmSubgroupsPreferred = 2;
constexpr int bmChunkMax = 2048;
// Candidate rows per lane in the blockMatch kernel when searches take several chunks: NY_FULL of the full search is bmItemRows, NY_PS of predictive searches
// bmPredictiveItemRows for groups larger than bmSmallGroup and bmItemRows otherwise (measured best); both are 1 for search steps of a block or more. In a
// single chunk itemShape chooses them, and the columns of the full search's items: at most bmItemColsMax, and bmItemCandidatesMax candidates per item.
constexpr int bmItemRows = 4;
constexpr int bmPredictiveItemRows = 2;
constexpr int bmSmallGroup = 8;
constexpr int bmItemRowsMax = 6;
constexpr int bmItemColsMax = 4;
constexpr int bmItemCandidatesMax = 12;
// Most rows per item of a predictive search in a single chunk, with the reference block in registers and in shared memory: more rows than that leave too few
// registers, which the driver makes up for with spills (measured).
constexpr int bmPredictiveItemRowsMaxInRegs = 5;
constexpr int bmPredictiveItemRowsMax = 8;
// Least squared distance between the positions of a lane's slots in a spatial search keyed by slot, which pads its rows for it.
constexpr int bmSpatialLaneSpacing = 10;
// Cost of a window load relative to a distance FMA in the full and the predictive searches (measured).
constexpr double bmFullLoadCost = 2.0;
constexpr double bmPredictiveLoadCost = 4.0;
constexpr int bmRefInRegsMax = 8; // largest block size whose reference block the blockMatch kernel holds in registers
constexpr int filterWorkgroupMin = 64;
constexpr int filterWorkgroupMax = 256;
constexpr int filterWorkgroupLimit = 512; // largest workgroup the filter grows to for filterWarpsTarget
constexpr int filterWarpsTarget = 12;     // warps per SM the filter workgroups should keep busy (measured)
constexpr int filterTileRefs = 4;
constexpr double filterTileGain = 10.0;   // least ratio of estimates accumulated in a tile to the tile's area (measured)
constexpr int64_t filterTileWorkgroups = 3; // least workgroups per SM a tile may leave where more would run without it
constexpr int64_t filterTileWorkgroupsBasic = 2; // the same for basic estimates, without a reference group (measured)
constexpr uint32_t preferredSubgroupSize = 32;
constexpr int64_t globalGroupBudget = int64_t{128} << 20;
constexpr int64_t matchBudget = int64_t{256} << 20;
constexpr uint32_t maxBindings = 7;
constexpr int64_t countBytes = sizeof(uint32_t);
constexpr int64_t accBytes = sizeof(float) * 2; // numerator and denominator per pixel

// Descriptor bindings, as declared by the kernels in shaders.h.
constexpr uint32_t bindIn = 0;
constexpr uint32_t bindOut = 1;
constexpr uint32_t bindWindow = 0;
constexpr uint32_t bindMatches = 1;
constexpr uint32_t bindCounts = 2;
constexpr uint32_t bindFilterSrc = 0;
constexpr uint32_t bindFilterRef = 1;
constexpr uint32_t bindFilterMatches = 2;
constexpr uint32_t bindFilterCounts = 3;
constexpr uint32_t bindFilterAcc = 4;
constexpr uint32_t bindFilterScratch = 5;
constexpr uint32_t bindFilterCos = 6;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error{"BM3D: " + message};
}

std::string replaceAll(std::string text, std::string_view from, std::string_view to) {
    for (auto pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size()))
        text.replace(pos, from.size(), to);
    return text;
}

int divCeil(int a, int b) {
    return (a + b - 1) / b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Owners of VapourSynth and Vulkan objects

struct NodeDeleter {
    const VSAPI* vsapi{};
    void operator()(VSNode* node) const noexcept { vsapi->freeNode(node); }
};
using NodePtr = std::unique_ptr<VSNode, NodeDeleter>;

struct FrameDeleter {
    const VSAPI* vsapi{};
    void operator()(const VSFrame* frame) const noexcept { vsapi->freeFrame(frame); }
};
using FramePtr = std::unique_ptr<const VSFrame, FrameDeleter>;
using OutFramePtr = std::unique_ptr<VSFrame, FrameDeleter>;

struct ShaderDeleter {
    const VSVULKANAPI* vkapi{};
    void operator()(VSGPUShader* shader) const noexcept { vkapi->freeGPUShader(shader); }
};
using ShaderPtr = std::unique_ptr<VSGPUShader, ShaderDeleter>;

struct BufferDeleter {
    const VSVULKANAPI* vkapi{};
    void operator()(VSGPUBuffer* buffer) const noexcept { vkapi->destroyGPUBuffer(buffer); }
};
using BufferPtr = std::unique_ptr<VSGPUBuffer, BufferDeleter>;

struct PoolDeleter {
    const VSVULKANAPI* vkapi{};
    void operator()(VSGPUExecPool* pool) const noexcept { vkapi->freeGPUExecPool(pool); }
};
using PoolPtr = std::unique_ptr<VSGPUExecPool, PoolDeleter>;

template <typename Handle, auto destroy>
class VkOwner {
public:
    VkOwner() = default;
    VkOwner(VkDevice device, const VSVulkanFunctions& vk, Handle handle) noexcept : device_{device}, vk_{&vk}, handle_{handle} {}
    VkOwner(const VkOwner&) = delete;
    VkOwner& operator=(const VkOwner&) = delete;
    VkOwner(VkOwner&& other) noexcept : device_{other.device_}, vk_{other.vk_}, handle_{std::exchange(other.handle_, VK_NULL_HANDLE)} {}
    VkOwner& operator=(VkOwner&& other) noexcept {
        if (this != &other) {
            reset();
            device_ = other.device_;
            vk_ = other.vk_;
            handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
        }
        return *this;
    }
    ~VkOwner() { reset(); }

    Handle get() const noexcept { return handle_; }

private:
    void reset() noexcept {
        if (handle_ != VK_NULL_HANDLE)
            (vk_->*destroy)(device_, std::exchange(handle_, VK_NULL_HANDLE), nullptr);
    }

    VkDevice device_{VK_NULL_HANDLE};
    const VSVulkanFunctions* vk_{};
    Handle handle_{VK_NULL_HANDLE};
};

using SetLayoutOwner = VkOwner<VkDescriptorSetLayout, &VSVulkanFunctions::vkDestroyDescriptorSetLayout>;
using PipelineLayoutOwner = VkOwner<VkPipelineLayout, &VSVulkanFunctions::vkDestroyPipelineLayout>;
using PipelineOwner = VkOwner<VkPipeline, &VSVulkanFunctions::vkDestroyPipeline>;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Device

struct Gpu {
    const VSVULKANAPI* vkapi{};
    const VSVulkanFunctions* vk{};
    VSVulkanCoreHandles handles{};
    bool nvidiaDriver{};
    bool fp16Math{};
    bool fp16DenormPreserve{};
    bool floatAtomics{};
    uint32_t subgroupSize{};
    int maxWorkgroup{};
    int64_t maxSharedMemory{};
    int maxGroupCountX{};
    int64_t maxStorageRange{};
    uint32_t maxStorageBuffers{};
};

bool hasDeviceExtension(const Gpu& gpu, std::string_view name) {
    uint32_t count{};
    if (gpu.vk->vkEnumerateDeviceExtensionProperties(gpu.handles.physicalDevice, nullptr, &count, nullptr) != VK_SUCCESS)
        fail("failed to enumerate device extensions");
    std::vector<VkExtensionProperties> extensions(count);
    if (gpu.vk->vkEnumerateDeviceExtensionProperties(gpu.handles.physicalDevice, nullptr, &count, extensions.data()) != VK_SUCCESS)
        fail("failed to enumerate device extensions");
    return std::ranges::any_of(extensions, [name](const VkExtensionProperties& e) { return name == e.extensionName; });
}

Gpu queryGpu(VSCore* core, const VSAPI& vsapi) {
    Gpu gpu{.vkapi = vsapi.getVulkanAPI()};
    std::array<char, errorSize> err{};
    if (gpu.vkapi->getVulkanHandles(core, &gpu.handles, err.data(), errorSize))
        fail(err.data());
    gpu.vk = gpu.vkapi->getVulkanFunctions(core, err.data(), errorSize);
    if (!gpu.vk)
        fail(err.data());

    VkPhysicalDeviceVulkan11Properties props11{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceVulkan12Properties props12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, .pNext = &props11};
    VkPhysicalDeviceVulkan13Properties props13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, .pNext = &props12};
    VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &props13};
    gpu.vk->vkGetPhysicalDeviceProperties2(gpu.handles.physicalDevice, &props);

    const bool atomicFloatExtension = hasDeviceExtension(gpu, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
    VkPhysicalDeviceVulkan12Features features12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                                .pNext = atomicFloatExtension ? &atomicFloat : nullptr};
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &features12};
    gpu.vk->vkGetPhysicalDeviceFeatures2(gpu.handles.physicalDevice, &features);

    if ((props13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
        fail("the device cannot pin the subgroup size of compute shaders");
    constexpr VkSubgroupFeatureFlags subgroupOps = VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    if ((props11.subgroupSupportedOperations & subgroupOps) != subgroupOps)
        fail("the device lacks subgroup arithmetic, ballot or shuffle operations");

    const VkPhysicalDeviceLimits& limits = props.properties.limits;
    gpu.nvidiaDriver = props12.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    gpu.fp16Math = features12.shaderFloat16 == VK_TRUE;
    // Only a 16-bit mode independent of the 64-bit one may be set alone. NVIDIA's default is as close to the CPU implementation, and the mode costs its
    // filter about 3% (RTX 4060 Ti, RGBH final vn).
    gpu.fp16DenormPreserve = !gpu.nvidiaDriver && props12.shaderDenormPreserveFloat16 == VK_TRUE &&
                             props12.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL;
    gpu.floatAtomics = atomicFloat.shaderBufferFloat32AtomicAdd == VK_TRUE;
    gpu.subgroupSize = std::clamp(preferredSubgroupSize, props13.minSubgroupSize, props13.maxSubgroupSize);
    if (limits.maxComputeWorkGroupInvocations < imageTile * imageTile || limits.maxComputeWorkGroupSize[0] < imageTile ||
        limits.maxComputeWorkGroupSize[1] < imageTile)
        fail("the device's compute workgroups are too small");
    gpu.maxWorkgroup = static_cast<int>(std::bit_floor(
        std::min({limits.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupSize[0], props13.maxComputeWorkgroupSubgroups * gpu.subgroupSize})));
    gpu.maxSharedMemory = limits.maxComputeSharedMemorySize;
    gpu.maxGroupCountX = static_cast<int>(std::min<uint32_t>(limits.maxComputeWorkGroupCount[0], std::numeric_limits<int>::max()));
    gpu.maxStorageRange = limits.maxStorageBufferRange;
    gpu.maxStorageBuffers = limits.maxPerStageDescriptorStorageBuffers;
    return gpu;
}

struct GpuBuffer {
    BufferPtr owner;
    VkBuffer buffer{VK_NULL_HANDLE};
};

// A device local storage buffer, also the destination of fills and copies.
GpuBuffer createBuffer(const Gpu& gpu, VSCore* core, int64_t size) {
    if (size > gpu.maxStorageRange)
        fail("a working buffer of " + std::to_string(size) + " bytes exceeds the device's storage buffer range");
    std::array<char, errorSize> err{};
    VSVulkanBufferInfo info{};
    BufferPtr owner{gpu.vkapi->createGPUBuffer(core, static_cast<VkDeviceSize>(size), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &info, err.data(), errorSize),
                    {gpu.vkapi}};
    if (!owner)
        fail(err.data());
    return {std::move(owner), info.buffer};
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameters

enum class Profile { fast, lc, np, high, vn };

struct Params {
    bool wiener{};
    std::array<double, maxPlanes> sigma{};
    int radius{};
    int blockSize{};
    int blockStep{};
    int groupSize{};
    int bmRange{};
    int bmStep{};
    int psNum{};
    int psRange{};
    int psStep{};
    double thMSE{};
    double lambda{};
    int matrix{};
};

std::optional<Profile> parseProfile(std::string_view name) {
    constexpr std::array<std::pair<std::string_view, Profile>, 5> names{
        {{"fast", Profile::fast}, {"lc", Profile::lc}, {"np", Profile::np}, {"high", Profile::high}, {"vn", Profile::vn}}};
    for (const auto& [n, p] : names)
        if (n == name)
            return p;
    return std::nullopt;
}

int defaultRadius(Profile profile) {
    switch (profile) {
    case Profile::fast:
        return 1;
    case Profile::lc:
        return 2;
    case Profile::np:
        return 3;
    case Profile::high:
    case Profile::vn:
        return 4;
    }
    return 1;
}

// The profile defaults of BM3D_Para and, for temporal filtering, VBM3D_Para of the CPU implementation.
Params profileDefaults(Profile profile, bool wiener, bool temporal) {
    Params p{.wiener = wiener, .blockSize = 8, .blockStep = wiener ? 3 : 4, .groupSize = wiener ? 32 : 16, .bmRange = 16,
             .bmStep = 1, .lambda = wiener ? 0.0 : 2.7};
    switch (profile) {
    case Profile::fast:
        p.bmRange = 9;
        p.groupSize = 8;
        p.blockStep = wiener ? 7 : 8;
        break;
    case Profile::lc:
        p.bmRange = 9;
        p.blockStep = wiener ? 5 : 6;
        p.groupSize = wiener ? 16 : p.groupSize;
        break;
    case Profile::np:
        break;
    case Profile::high:
        p.blockStep = wiener ? 2 : 3;
        break;
    case Profile::vn:
        if (wiener) {
            p.blockSize = 11;
            p.blockStep = 6;
        } else {
            p.blockStep = 4;
            p.groupSize = 32;
            p.lambda = 2.8;
        }
        break;
    }
    if (!temporal)
        return p;

    p.groupSize = profile == Profile::vn ? 16 : 8;
    p.psNum = 2;
    p.psStep = 1;
    switch (profile) {
    case Profile::fast:
        p.bmRange = 7;
        p.psRange = wiener ? 5 : 4;
        break;
    case Profile::lc:
        p.bmRange = 9;
        p.psRange = wiener ? 5 : 4;
        break;
    case Profile::np:
    case Profile::vn:
        p.bmRange = 12;
        p.psRange = wiener ? 6 : 5;
        break;
    case Profile::high:
        p.bmRange = 16;
        p.psRange = wiener ? 8 : 7;
        break;
    }
    return p;
}

double defaultThMSE(Profile profile, bool wiener, double sigma) {
    if (wiener)
        return profile == Profile::vn ? 400.0 + sigma * 40.0 : 200.0 + sigma * 10.0;
    return profile == Profile::vn ? 1000.0 + sigma * 150.0 : 400.0 + sigma * 80.0;
}

constexpr int matrixUnspecified = 2;
constexpr int matrixOPP = 100;

// Matrix coefficients Kr, Kg, Kb as ColorMatrix_Parameter of the CPU implementation.
std::optional<std::array<double, 3>> matrixCoefficients(int matrix) {
    switch (matrix) {
    case 0:
        return std::array{0.0, 1.0, 0.0};
    case 1:
        return std::array{0.2126, 0.7152, 0.0722};
    case 4:
        return std::array{0.30, 0.59, 0.11};
    case 5:
    case 6:
        return std::array{0.299, 0.587, 0.114};
    case 7:
        return std::array{0.212, 0.701, 0.087};
    case 9:
    case 10:
        return std::array{0.2627, 0.6780, 0.0593};
    case matrixOPP:
        return std::array{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0};
    default:
        return std::nullopt;
    }
}

// ColorMatrix_Default of the CPU implementation.
int defaultMatrix(int width, int height) {
    if (width > 2048 || height > 1536)
        return 9;
    if (width > 1024 || height > 576)
        return 1;
    return 6;
}

// Norms of the Y, U and V rows of the RGB -> YUV matrix, which scale sigma and th_mse to the matrix.
std::array<double, maxPlanes> matrixNorms(int matrix) {
    if (matrix == 0)
        return {1.0, 1.0, 1.0};
    if (matrix == matrixOPP)
        return {std::sqrt(1.0 / 3.0), std::sqrt(0.5), std::sqrt(6.0) / 4.0};
    const auto [kr, kg, kb] = *matrixCoefficients(matrix);
    const double normY = std::sqrt(kr * kr + kg * kg + kb * kb);
    const double normU = std::hypot(kr * 0.5 / (1.0 - kb), kg * 0.5 / (1.0 - kb), 0.5);
    const double normV = std::hypot(0.5, kg * 0.5 / (1.0 - kr), kb * 0.5 / (1.0 - kr));
    return {normY, normU, normV};
}

class ArgReader {
public:
    ArgReader(const VSMap& in, const VSAPI& vsapi) : in_{&in}, vsapi_{&vsapi} {}

    std::optional<int> integer(const char* key) const {
        int err{};
        const int v = vsapi_->mapGetIntSaturated(in_, key, 0, &err);
        if (err)
            return std::nullopt;
        return v;
    }

    std::optional<double> real(const char* key) const {
        int err{};
        const double v = vsapi_->mapGetFloat(in_, key, 0, &err);
        if (err)
            return std::nullopt;
        if (!std::isfinite(v))
            fail("\""s + key + "\" must be finite");
        return v;
    }

    int count(const char* key) const { return std::max(vsapi_->mapNumElements(in_, key), 0); }

    double realAt(const char* key, int index) const { return vsapi_->mapGetFloat(in_, key, index, nullptr); }

    std::optional<std::string> data(const char* key) const {
        int err{};
        const char* v = vsapi_->mapGetData(in_, key, 0, &err);
        if (err)
            return std::nullopt;
        return std::string{v};
    }

private:
    const VSMap* in_{};
    const VSAPI* vsapi_{};
};

struct IntRange {
    int lo{};
    int hi{};
};

// Only an assigned value is validated: like the CPU implementation, a profile default such as ps_num may exceed a small group_size.
int intArg(const ArgReader& args, const char* key, int fallback, IntRange range) {
    const std::optional<int> v = args.integer(key);
    if (!v)
        return fallback;
    if (*v < range.lo || *v > range.hi)
        fail("\""s + key + "\" must be an integer in [" + std::to_string(range.lo) + ", " + std::to_string(range.hi) + "]");
    return *v;
}

Params parseParams(const ArgReader& args, const VSVideoInfo& vi, bool wiener) {
    const std::string profileName = args.data("profile").value_or("fast");
    const std::optional<Profile> profile = parseProfile(profileName);
    if (!profile)
        fail("unrecognized \"profile\", must be \"fast\", \"lc\", \"np\", \"high\" or \"vn\"");

    std::array<double, maxPlanes> sigma{10.0, 10.0, 10.0};
    if (const int m = std::min(args.count("sigma"), maxPlanes); m > 0) {
        for (int i = 0; i < maxPlanes; ++i) {
            sigma[i] = i < m ? args.realAt("sigma", i) : sigma[i - 1];
            if (!std::isfinite(sigma[i]) || sigma[i] < 0.0)
                fail("\"sigma\" must be non-negative finite numbers");
        }
    }

    constexpr int unbounded = std::numeric_limits<int>::max();
    const int radius = intArg(args, "radius", defaultRadius(*profile), {0, maxRadius});
    Params p = profileDefaults(*profile, wiener, radius > 0);
    p.sigma = sigma;
    p.radius = radius;
    p.blockSize = intArg(args, "block_size", p.blockSize, {1, maxBlockSize});
    if (p.blockSize > vi.width || p.blockSize > vi.height)
        fail("\"block_size\" must not exceed the width or height of the clip");
    p.blockStep = intArg(args, "block_step", std::min(p.blockStep, p.blockSize), {1, p.blockSize});
    p.groupSize = intArg(args, "group_size", p.groupSize, {1, maxGroupSize});
    p.bmRange = intArg(args, "bm_range", p.bmRange, {1, unbounded});
    p.bmStep = intArg(args, "bm_step", p.bmStep, {1, p.bmRange});
    const Params temporalDefaults = profileDefaults(*profile, wiener, true);
    p.psNum = intArg(args, "ps_num", temporalDefaults.psNum, {1, p.groupSize});
    p.psRange = intArg(args, "ps_range", temporalDefaults.psRange, {1, unbounded});
    p.psStep = intArg(args, "ps_step", temporalDefaults.psStep, {1, p.psRange});

    p.thMSE = args.real("th_mse").value_or(defaultThMSE(*profile, wiener, sigma[0]));
    if (p.thMSE < 0.0)
        fail("\"th_mse\" must be a non-negative number");
    p.lambda = args.real("hard_thr").value_or(p.lambda);
    if (!wiener && p.lambda <= 0.0)
        fail("\"hard_thr\" must be a positive number");

    if (vi.format.colorFamily == cfRGB) {
        p.matrix = matrixOPP;
    } else {
        p.matrix = args.integer("matrix").value_or(matrixUnspecified);
        if (p.matrix == matrixUnspecified)
            p.matrix = defaultMatrix(vi.width, vi.height);
        else if (!matrixCoefficients(p.matrix))
            fail("unsupported \"matrix\"");
    }
    return p;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel configuration

struct CosTable {
    std::vector<float> values;
    int zBase{};
};

struct Plan {
    int width{};
    int height{};
    int planes{};
    bool fp16Input{};
    int slots{};
    int blockPadded{};
    int groupSwizzle{};
    int refCols{};
    int refRows{};
    int batchRows{};
    bool noMatching{};
    float thSSE{};
    std::array<float, maxPlanes * 4> coef{};
    int bmSubgroups{};
    int bmWindows{};
    int bmFullRows{};
    int bmFullCols{1};
    int bmPredictiveRows{};
    int fullCols{};
    int fullRows{};
    int fullPitch{};
    bool bmSlotKeys{};
    int psCols{};
    int psRows{};
    bool psUnion{};
    int bmChunk{};
    bool bmSingleChunk{};
    bool bmCorrelation{};
    bool bmRefInRegs{};
    bool fullRowMajor{};
    bool psRowMajor{};
    int filterWorkgroup{};
    bool globalGroup{};
    bool pingpong{};
    int filterGroupsX{};
    int64_t groupBytes{};
    int searchMargin{};
    int tileRefs{};
    int tileSpan{};
    bool accTile{};
    CosTable cosTable;
};

int64_t storeBytes(const Plan& plan) {
    return plan.fp16Input ? 2 : 4;
}

int64_t planeElements(const Plan& plan) {
    return int64_t{plan.width} * plan.height;
}

int64_t batchRefs(const Plan& plan) {
    return int64_t{plan.batchRows} * plan.refCols;
}

// A group member (MATCH_T in shaders.h): packed position, and window frame when filtering temporally.
int64_t matchBytes(const Params& p) {
    return sizeof(uint32_t) * (p.radius > 0 ? 2 : 1);
}

// A window buffer of `frames` planes. Full-search items of several columns read up to (NX_FULL - 1) * BM_STEP samples past the right edge, past the last plane
// at its last row; only the candidates rejected there take them.
int64_t windowBytes(const Params& p, const Plan& plan, int frames) {
    return (planeElements(plan) * frames + int64_t{plan.bmFullCols - 1} * p.bmStep) * storeBytes(plan);
}

// cos(pi * q / (2n)), q in [0, 4n), for each transform size above the codelets: the block size first, then 33..group size.
CosTable buildCosTable(const Params& p) {
    CosTable table;
    const auto append = [&table](int n) {
        for (int q = 0; q < n * 4; ++q)
            table.values.push_back(static_cast<float>(std::cos(std::numbers::pi * q / (2.0 * n))));
    };
    if (p.blockSize > dct::maxCodeletSize)
        append(p.blockSize);
    table.zBase = static_cast<int>(table.values.size());
    for (int n = dct::maxCodeletSize + 1; n <= p.groupSize; ++n)
        append(n);
    return table;
}

// Search positions along one axis of the largest search window, which the frame may clip.
int windowSide(int range, int step, int extent, int blockSize) {
    return static_cast<int>(std::min(int64_t{range / step} * 2, int64_t{(extent - blockSize) / step})) + 1;
}

// Candidate slots of one search window, rows padded to whole items.
int64_t windowCandidates(int cols, int rows, int itemRows) {
    return int64_t{(rows + itemRows - 1) / itemRows * itemRows} * cols;
}

// Rows and columns per item of a search in a single chunk: an item costs the distance operations of its candidates (an FMA per sample in correlation form, else
// a subtraction and an FMA) plus the window rows its lane loads (in correlation form with the mean offset, and an energy FMA per sample of each candidate
// column), and the lanes take items a subgroup at a time. Ties keep fewer columns.
std::pair<int, int> itemShape(int cols, int rows, int windows, int step, int blockSize, int sg, double loadCost, bool correlation, int maxRows, int maxCols) {
    const double sampleCost = correlation ? 1.0 : 2.0;
    const double sampleLoadCost = correlation ? loadCost + 1.0 : loadCost;
    std::pair best{1, 1};
    double bestCost = std::numeric_limits<double>::infinity();
    for (int nx = 1; nx <= std::min(maxCols, cols); ++nx) {
        for (int ny = 1; ny <= std::min(maxRows, rows) && ny * nx <= bmItemCandidatesMax; ++ny) {
            const int64_t items = int64_t{(rows + ny - 1) / ny} * ((cols + nx - 1) / nx) * windows;
            const double rowCost = sampleLoadCost * ((nx - 1) * step + blockSize) + (correlation ? nx * blockSize : 0);
            const double perItem = sampleCost * ny * nx * blockSize * blockSize + rowCost * ((ny - 1) * step + blockSize);
            const double cost = static_cast<double>((items + sg - 1) / sg) * perItem;
            if (cost < bestCost) {
                best = {ny, nx};
                bestCost = cost;
            }
        }
    }
    return best;
}

Plan makePlan(const Params& p, const VSVideoInfo& vi, const Gpu& gpu) {
    Plan plan{.width = vi.width, .height = vi.height, .planes = vi.format.numPlanes, .fp16Input = vi.format.bitsPerSample == 16, .slots = p.radius * 2 + 1};

    plan.refCols = divCeil(vi.width - p.blockSize, p.blockStep) + 1;
    plan.refRows = divCeil(vi.height - p.blockSize, p.blockStep) + 1;
    if (!std::in_range<int>(int64_t{plan.refCols} * plan.refRows))
        fail("too many reference blocks, increase \"block_step\"");

    // Mirrors the shared arrays of the normalize kernel.
    const int64_t normalizeSpan = imageTile + p.blockSize - 1;
    const int64_t normalizeShared = (normalizeSpan + imageTile) * normalizeSpan * sizeof(float);
    if (normalizeShared > gpu.maxSharedMemory)
        fail("\"block_size\" is too large for the device's shared memory");

    const std::array<double, maxPlanes> norms = matrixNorms(p.matrix);
    const double thMSE = p.thMSE * norms[0];
    plan.noMatching = p.groupSize == 1 || thMSE <= 0.0;
    plan.thSSE = static_cast<float>(thMSE * p.blockSize * p.blockSize / (255.0 * 255.0));

    // Coefficients are in the domain prescaled by the inverse square root of the transform gain, where the gain cancels.
    const std::array<double, 4> thresholdFactors{1.0, std::numbers::sqrt2, 2.0, std::sqrt(8.0)};
    for (int plane = 0; plane < plan.planes; ++plane) {
        const double sigma = p.sigma[plane] / 255.0 * norms[plane];
        for (int flag = 0; flag < 4; ++flag)
            plan.coef[plane * 4 + flag] = static_cast<float>(p.wiener ? std::max(sigma * sigma * (1 << flag), double{std::numeric_limits<float>::min()})
                                                                     : sigma * p.lambda * thresholdFactors[flag]);
    }

    // Mirrors the shared arrays of the blockMatch kernel.
    const auto sg = static_cast<int>(gpu.subgroupSize);
    // Candidates a block or more apart share no window rows, so their items hold a single row.
    plan.bmFullRows = p.bmStep < p.blockSize ? bmItemRows : 1;
    plan.bmPredictiveRows = p.psStep >= p.blockSize ? 1 : p.groupSize > bmSmallGroup ? bmPredictiveItemRows : bmItemRows;
    plan.fullCols = windowSide(p.bmRange, p.bmStep, vi.width, p.blockSize);
    plan.fullRows = windowSide(p.bmRange, p.bmStep, vi.height, p.blockSize);
    plan.psCols = windowSide(p.psRange, p.psStep, vi.width, p.blockSize);
    plan.psRows = windowSide(p.psRange, p.psStep, vi.height, p.blockSize);
    // Row-major slots of a lane lie SG apart: SG / pitch rows down and SG % pitch columns across, or a row further and pitch - SG % pitch columns back, the
    // nearer at the squared distance laneSpacing. Diagonal neighbours there would have correlated distances (see blockMatch).
    const auto laneSpacing = [sg](int64_t pitch) {
        const int64_t down = sg / pitch;
        const int64_t across = sg % pitch;
        return std::min(down * down + across * across, (down + 1) * (down + 1) + (pitch - across) * (pitch - across));
    };
    // A spatial search wider than a subgroup is keyed by slot (see selectSlots), its rows padded until a lane's slots lie apart; narrower ones, with few
    // candidates, run faster on the general merge (measured). No pitch puts a lane's slots further apart than SG columns in a row.
    plan.bmSlotKeys = p.radius == 0 && plan.fullCols > sg;
    plan.fullPitch = plan.fullCols;
    while (plan.bmSlotKeys && laneSpacing(plan.fullPitch) < std::min(bmSpatialLaneSpacing, sg * sg))
        ++plan.fullPitch;
    // A single chunk holds the candidates of every search without item padding, and stages the merges of the rest set in its first 4 * GROUP words; shared
    // memory bounds how many workgroups an SM runs, so every word counts.
    const int64_t list = int64_t{p.groupSize} * sizeof(uint32_t);
    const auto exactCandidatesOf = [&](int psCols, int psRows) {
        return std::max(int64_t{plan.fullPitch} * plan.fullRows,
                        p.radius > 0 ? std::max(int64_t{psCols} * psRows * p.psNum, int64_t{p.groupSize} * 4) : int64_t{0});
    };
    // The search centers of a frame mostly lie a step or so apart, their windows nearly the same: where their bounding box fits a window one step larger, it is
    // searched as a single window, unless the larger layout takes more slots.
    const int unionCols = std::min(plan.psCols + 1, (vi.width - p.blockSize) / p.psStep + 1);
    const int unionRows = std::min(plan.psRows + 1, (vi.height - p.blockSize) / p.psStep + 1);
    plan.psUnion = p.radius > 0 && p.psNum > 1 && exactCandidatesOf(unionCols, unionRows) <= exactCandidatesOf(plan.psCols, plan.psRows);
    if (plan.psUnion) {
        plan.psCols = unionCols;
        plan.psRows = unionRows;
    }
    plan.bmWindows = p.radius > 0 ? p.psNum + (plan.psUnion ? 1 : 0) : 1;
    const int64_t fullCandidates = windowCandidates(plan.fullCols, plan.fullRows, plan.bmFullRows);
    const int64_t maxCandidates = p.radius == 0 ? fullCandidates
                                                : std::max(fullCandidates, windowCandidates(plan.psCols, plan.psRows, plan.bmPredictiveRows) * p.psNum);
    if (!std::in_range<int32_t>(maxCandidates))
        fail("too many search candidates, decrease \"bm_range\", \"ps_range\" or \"ps_num\"");
    const int64_t exactCandidates = exactCandidatesOf(plan.psCols, plan.psRows);
    // Spatial windows no wider than a subgroup are register bound: a reference in shared memory frees the registers their selection network needs (measured).
    // Only NVIDIA's compiler keeps the block, indexed by row at run time, in registers; RADV indexes it through branch ladders and AMD's Windows compiler
    // through scratch memory, both far slower than shared memory (RX 7900 XTX on RADV, BM3D(radius=2): blockMatch 4.9 -> 2.9 ms).
    plan.bmRefInRegs = gpu.nvidiaDriver &&p.blockSize <= bmRefInRegsMax && (p.radius > 0 || plan.bmSlotKeys);
    const auto bmShared = [&](int subgroups, int64_t chunk, bool singleChunk) {
        const int64_t perSubgroup = (plan.bmRefInRegs ? 0 : int64_t{p.blockSize} * divCeil(p.blockSize, 4) * 4 * sizeof(float)) +
                                    int64_t{plan.bmWindows} * 4 * sizeof(uint32_t) + chunk * sizeof(uint32_t) +
                                    list * (plan.bmSlotKeys && singleChunk ? 1 : 2) + (singleChunk ? 0 : list * (p.radius > 0 ? 4 : 2)) +
                                    (p.radius > 0 ? list * 4 + int64_t{p.psNum} * 2 * sizeof(uint32_t) : 0);
        return perSubgroup * subgroups;
    };
    plan.fullRowMajor = plan.bmSlotKeys || laneSpacing(plan.fullPitch) > 2;
    plan.psRowMajor = laneSpacing(plan.psCols) > 2;
    // Distances in correlation form take the reference block offset by its mean, which pays off where the block is read from shared memory; one in registers,
    // which the compiler reads again from the window instead, does not (measured).
    plan.bmCorrelation = p.blockSize > bmRefInRegsMax;
    plan.bmSubgroups = std::min(bmSubgroupsPreferred, gpu.maxWorkgroup / sg);
    plan.bmSingleChunk = exactCandidates <= bmChunkMax && bmShared(1, exactCandidates, true) <= gpu.maxSharedMemory;
    if (plan.bmSingleChunk) {
        std::tie(plan.bmFullRows, plan.bmFullCols) =
            itemShape(plan.fullCols, plan.fullRows, 1, p.bmStep, p.blockSize, sg, bmFullLoadCost, plan.bmCorrelation, bmItemRowsMax, bmItemColsMax);
        // Most union windows are a single window, which items of more rows fill in one pass of the subgroup.
        plan.bmPredictiveRows = itemShape(plan.psCols, plan.psRows, plan.psUnion ? 1 : p.psNum, p.psStep, p.blockSize, sg, bmPredictiveLoadCost,
                                          plan.bmCorrelation,
                                          gpu.nvidiaDriver && p.blockSize <= bmRefInRegsMax ? bmPredictiveItemRowsMaxInRegs : bmPredictiveItemRowsMax, 1)
                                    .first;
        plan.bmChunk = static_cast<int>(exactCandidates);
        while (bmShared(plan.bmSubgroups, plan.bmChunk, true) > gpu.maxSharedMemory)
            --plan.bmSubgroups;
    } else {
        // Chunks hold whole items of either search.
        static_assert(bmItemRows % bmPredictiveItemRows == 0 && bmChunkMax % bmItemRows == 0);
        plan.bmChunk = divCeil(static_cast<int>(std::min<int64_t>(maxCandidates, bmChunkMax)), bmItemRows) * bmItemRows;
        while (bmShared(plan.bmSubgroups, plan.bmChunk, false) > gpu.maxSharedMemory) {
            if (plan.bmSubgroups > 1)
                --plan.bmSubgroups;
            else if (plan.bmChunk > bmItemRows)
                plan.bmChunk = std::max(plan.bmChunk / 2 / bmItemRows * bmItemRows, bmItemRows);
            else
                fail("block matching does not fit into the device's shared memory");
        }
    }

    // Mirrors the shared arrays of the filter kernel. A block row holds 2 * halfBlock elements; for 8-byte elements an odd stride puts the rows a pass reads at
    // once into distinct banks, while 4-byte elements fare better with rows aligned for 16-byte loads (measured).
    const int halfBlock = (p.blockSize + 1) / 2;
    plan.filterWorkgroup = std::min(std::clamp(static_cast<int>(std::bit_ceil(static_cast<unsigned>(p.groupSize * halfBlock))),
                                               std::max(filterWorkgroupMin, sg), std::max(filterWorkgroupMax, sg)),
                                    gpu.maxWorkgroup);
    plan.pingpong = p.blockSize > dct::maxCodeletSize || p.groupSize > dct::maxCodeletSize;
    const int elementBytes = gpu.fp16Math && plan.fp16Input ? 4 : 8;
    plan.blockPadded = 2 * halfBlock + (elementBytes == 8 ? 1 : 0);
    plan.groupBytes = int64_t{p.groupSize} * halfBlock * plan.blockPadded * elementBytes * (p.wiener ? 2 : 1) * (plan.pingpong ? 2 : 1);
    // Wiener groups with a codelet z-pass keep the subgroup sums of the shrinkage in their own data unless the group is in memory (see STAT_IN_REF).
    const int64_t partialShared = std::max(filterWorkgroupLimit / sg, 1) * sizeof(float);
    const int64_t filterFixedShared = p.wiener && p.groupSize <= dct::maxCodeletSize ? 0 : partialShared;
    plan.globalGroup = plan.groupBytes + filterFixedShared > gpu.maxSharedMemory;
    const int64_t groupShared = plan.globalGroup ? partialShared : plan.groupBytes + filterFixedShared;

    // Tiles of filterTileRefs x filterTileRefs reference blocks share an accumulation tile covering their search windows. It pays off only when far more
    // estimates land in the tile than its flush writes back, and not when it leaves an SM few workgroups where more would run without it.
    plan.searchMargin = std::min(p.bmRange / p.bmStep * p.bmStep, std::max(vi.width, vi.height));
    const int span = (filterTileRefs - 1) * p.blockStep + p.blockSize + plan.searchMargin * 2;
    const double tileSamples = static_cast<double>(p.groupSize) * p.blockSize * p.blockSize * filterTileRefs * filterTileRefs;
    const int64_t tileShared = int64_t{span} * span * sizeof(float);
    const int64_t tileWorkgroups = gpu.maxSharedMemory / (groupShared + tileShared);
    plan.accTile = tileWorkgroups > 0 && tileSamples >= filterTileGain * span * span &&
                   (tileWorkgroups >= (p.wiener ? filterTileWorkgroups : filterTileWorkgroupsBasic) || tileWorkgroups == gpu.maxSharedMemory / groupShared);
    plan.tileRefs = plan.accTile ? filterTileRefs : 1;
    plan.tileSpan = plan.accTile ? span : 1;

    // Shared memory bounds the workgroups an SM runs, an SM granting each workgroup the device's limit per workgroup at most (measured): where few fit, larger
    // workgroups keep more warps busy.
    const int64_t workgroupShared = groupShared + (plan.accTile ? int64_t{span} * span * sizeof(float) : 0);
    const int64_t workgroupsPerSm = std::max<int64_t>(gpu.maxSharedMemory / workgroupShared, 1);
    while (plan.filterWorkgroup < std::min(filterWorkgroupLimit, gpu.maxWorkgroup) && workgroupsPerSm * plan.filterWorkgroup / sg < filterWarpsTarget)
        plan.filterWorkgroup *= 2;

    // Without the odd stride, blocks of four row pairs of 8-byte elements are permuted to stay free of bank conflicts (see swizzled in shaders.h), where that
    // fits more workgroups into an SM. The workgroup size and the tile stay those chosen for the padded layout.
    const int64_t paddingBytes = plan.groupBytes / plan.blockPadded;
    if (elementBytes == 8 && halfBlock == 4 && !plan.pingpong && !plan.globalGroup &&
        gpu.maxSharedMemory / (workgroupShared - paddingBytes) > workgroupsPerSm) {
        plan.groupSwizzle = 2;
        --plan.blockPadded;
        plan.groupBytes -= paddingBytes;
    }

    // Matching and filtering run in batches of whole tile rows of reference blocks, bounded by the size of the matches.
    const int64_t rowsBudget = std::min(matchBudget, gpu.maxStorageRange) / (int64_t{plan.refCols} * p.groupSize * matchBytes(p));
    plan.batchRows = static_cast<int>(std::min<int64_t>(std::max<int64_t>(rowsBudget / plan.tileRefs * plan.tileRefs, plan.tileRefs), plan.refRows));
    plan.filterGroupsX = static_cast<int>(std::min<int64_t>(int64_t{divCeil(plan.batchRows, plan.tileRefs)} * divCeil(plan.refCols, plan.tileRefs),
                                                            gpu.maxGroupCountX));
    if (plan.globalGroup) {
        const int64_t budget = std::min(globalGroupBudget, gpu.maxStorageRange);
        plan.filterGroupsX = std::clamp(static_cast<int>(budget / (plan.groupBytes * plan.planes)), 1, plan.filterGroupsX);
    }

    if (plan.pingpong)
        plan.cosTable = buildCosTable(p);
    return plan;
}

std::string floatBits(float v) {
    return "uintBitsToFloat(" + std::to_string(std::bit_cast<uint32_t>(v)) + "u)";
}

std::string buildPreamble(const Params& p, const Plan& plan, const Gpu& gpu) {
    const bool fp16Math = gpu.fp16Math && plan.fp16Input;
    std::string s = "#version 460\n"
                    "#extension GL_EXT_shader_16bit_storage : require\n"
                    "#extension GL_KHR_shader_subgroup_basic : require\n"
                    "#extension GL_KHR_shader_subgroup_arithmetic : require\n"
                    "#extension GL_KHR_shader_subgroup_ballot : require\n"
                    "#extension GL_KHR_shader_subgroup_shuffle : require\n"
                    "#extension GL_EXT_control_flow_attributes : require\n";
    if (fp16Math)
        s += "#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require\n";
    if (gpu.floatAtomics)
        s += "#extension GL_EXT_shader_atomic_float : require\n";
    // Half precision groups are prescaled far down (see filterGroup), which takes dark and near-neutral samples into the subnormal range: keep them where a
    // driver would otherwise flush them (capability and execution mode DenormPreserve for 16-bit floats).
    const bool denormPreserve = fp16Math && gpu.fp16DenormPreserve;
    if (denormPreserve)
        s += "#extension GL_EXT_spirv_intrinsics : require\n";

    std::string coef;
    for (int i = 0; i < plan.planes * 4; ++i)
        coef += (i ? ", " : "") + floatBits(plan.coef[i]);

    const std::vector<std::pair<std::string_view, std::string>> defines{
        {"WIDTH", std::to_string(plan.width)},
        {"HEIGHT", std::to_string(plan.height)},
        {"PLANES", std::to_string(plan.planes)},
        {"STORE_T", plan.fp16Input ? "float16_t" : "float"},
        {"T", fp16Math ? "f16vec2" : "vec2"},
        {"S", fp16Math ? "float16_t" : "float"},
        {"FLOAT_ATOMICS", gpu.floatAtomics ? "1" : "0"},
        {"IMAGE_TILE", std::to_string(imageTile)},
        {"SLOTS", std::to_string(plan.slots)},
        {"RADIUS", std::to_string(p.radius)},
        {"BLOCK", std::to_string(p.blockSize)},
        {"BLOCK_P", std::to_string(plan.blockPadded)},
        {"GROUP_SWIZZLE", std::to_string(plan.groupSwizzle)},
        {"BLOCK_STEP", std::to_string(p.blockStep)},
        {"GROUP", std::to_string(p.groupSize)},
        {"BM_RANGE", std::to_string(p.bmRange)},
        {"BM_STEP", std::to_string(p.bmStep)},
        {"PS_NUM", std::to_string(p.psNum)},
        {"PS_RANGE", std::to_string(p.psRange)},
        {"PS_STEP", std::to_string(p.psStep)},
        {"REF_COLS", std::to_string(plan.refCols)},
        {"TH_SSE", floatBits(plan.thSSE)},
        {"NO_MATCHING", plan.noMatching ? "1" : "0"},
        {"WIENER", p.wiener ? "1" : "0"},
        {"SG", std::to_string(gpu.subgroupSize)},
        {"NSG", std::to_string(plan.bmSubgroups)},
        {"MAX_WINDOWS", std::to_string(plan.bmWindows)},
        {"NY_FULL", std::to_string(plan.bmFullRows)},
        {"NX_FULL", std::to_string(plan.bmFullCols)},
        {"NY_PS", std::to_string(plan.bmPredictiveRows)},
        {"FULL_COLS", std::to_string(plan.fullCols)},
        {"FULL_ROWS", std::to_string(plan.fullRows)},
        {"FULL_PITCH", std::to_string(plan.fullPitch)},
        {"SLOT_KEYS", plan.bmSlotKeys && plan.bmSingleChunk ? "1" : "0"},
        {"PS_COLS", std::to_string(plan.psCols)},
        {"PS_ROWS", std::to_string(plan.psRows)},
        {"PS_UNION", plan.psUnion ? "1" : "0"},
        {"REF_IN_REGS", plan.bmRefInRegs ? "1" : "0"},
        {"WHOLE_ROW_LOADS", gpu.nvidiaDriver ? "0" : "1"},
        {"SSD_CORRELATION", plan.bmCorrelation ? "1" : "0"},
        {"CHUNK", std::to_string(plan.bmChunk)},
        {"SINGLE_CHUNK", plan.bmSingleChunk ? "1" : "0"},
        {"FULL_ROW_MAJOR", plan.fullRowMajor ? "1" : "0"},
        {"PS_ROW_MAJOR", plan.psRowMajor ? "1" : "0"},
        {"WG", std::to_string(plan.filterWorkgroup)},
        {"GLOBAL_GROUP", plan.globalGroup ? "1" : "0"},
        {"ACC_TILE", plan.accTile ? "1" : "0"},
        {"TILE_REFS", std::to_string(plan.tileRefs)},
        {"TILE_SPAN", std::to_string(plan.tileSpan)},
        {"SEARCH_MARGIN", std::to_string(plan.searchMargin)},
        {"PINGPONG", plan.pingpong ? "1" : "0"},
        {"Z_TABLE_BASE", std::to_string(plan.cosTable.zBase)},
        {"MAX_CODELET", std::to_string(dct::maxCodeletSize)},
        {"COEF_TABLE", coef},
        {"FLT_EPSILON", floatBits(std::numeric_limits<float>::epsilon())},
        {"DCT2_BLOCK", "dct2_" + std::to_string(p.blockSize)},
        {"DCT3_BLOCK", "dct3_" + std::to_string(p.blockSize)},
        {"FMA(a, b, c)", "fma(T(a), b, c)"},
        {"FMS(a, b, c)", "fma(T(a), b, -(c))"},
        {"FNMS(a, b, c)", "fma(-T(a), b, c)"},
        {"SFMA(a, b, c)", "fma(S(a), b, c)"},
        {"SFMS(a, b, c)", "fma(S(a), b, -(c))"},
        {"SFNMS(a, b, c)", "fma(-S(a), b, c)"},
        {"Z_SCALAR", fp16Math ? "0" : "1"},
    };
    for (const auto& [name, value] : defines)
        s += "#define "s.append(name) + " " + value + "\n";
    if (denormPreserve)
        s += "spirv_execution_mode(capabilities = [4464], 4459, 16);\n";
    return s;
}

// The z-pass of one line pair (T) or one line (S, Z_SCALAR) for every group size with a codelet: forward transform, shrinkage, inverse transform. The Wiener
// weights are derived from the reference spectrum first and stored in its place, so only one spectrum is live at a time.
std::string zLineFunction(int groupSize, bool scalar) {
    constexpr std::string_view caseTemplate = R"glsl(
    case @: {
        ELEM a[@], o[@];
#if WIENER
        [[unroll]] for (uint z = 0; z < @; ++z)
            a[z] = G(REF_GRP + swizzled(z, off) + z * BLK_T)[C];
        dct2_@(a, o);
        [[unroll]] for (uint z = 0; z < @; ++z)
            G(REF_GRP + swizzled(z, off) + z * BLK_T)[C] = wienerWeight(o[z], z, row, x, stat);
#endif
        [[unroll]] for (uint z = 0; z < @; ++z)
            a[z] = G(swizzled(z, off) + z * BLK_T)[C];
        dct2_@(a, o);
        [[unroll]] for (uint z = 0; z < @; ++z)
#if WIENER
            o[z] *= G(REF_GRP + swizzled(z, off) + z * BLK_T)[C];
#else
            o[z] = threshold(o[z], z, row, x, stat);
#endif
        dct3_@(o, a);
        [[unroll]] for (uint z = 0; z < @; ++z)
            G(swizzled(z, off) + z * BLK_T)[C] = a[z];
        break;
    }
)glsl";
    std::string s = "void zLine(uint k, uint off, uint row, uint x, inout float stat) {\n    switch (k) {\n";
    for (int n = 1; n <= std::min(groupSize, dct::maxCodeletSize); ++n)
        s += replaceAll(std::string{caseTemplate}, "@", std::to_string(n));
    s += "    default:\n        break;\n    }\n}\n";
    s = replaceAll(std::move(s), "ELEM", scalar ? "S" : "T");
    s = replaceAll(std::move(s), "[C]", scalar ? "[row % 2]" : "");
    if (scalar) {
        s = replaceAll(std::move(s), "dct2_", "dct2s_");
        s = replaceAll(std::move(s), "dct3_", "dct3s_");
    }
    return s;
}

// A codelet on S instead of T: the type T as a whole word becomes S, and the names and FMA macros get scalar spellings.
std::string scalarCodelet(std::string_view codelet) {
    std::string s;
    const auto identifier = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_'; };
    for (size_t i = 0; i < codelet.size(); ++i) {
        const bool word = codelet[i] == 'T' && (i == 0 || !identifier(codelet[i - 1])) && (i + 1 == codelet.size() || !identifier(codelet[i + 1]));
        s += word ? 'S' : codelet[i];
    }
    for (const auto& [from, to] :
         {std::pair{"dct2_"sv, "dct2s_"sv}, {"dct3_"sv, "dct3s_"sv}, {"FMA("sv, "SFMA("sv}, {"FMS("sv, "SFMS("sv}, {"FNMS("sv, "SFNMS("sv}})
        s = replaceAll(std::move(s), from, to);
    return s;
}

std::string filterSource(const Params& p, const std::string& preamble, bool scalarZ) {
    std::string s = preamble;
    s += shader::common;
    const int maxCodelet = std::min(p.groupSize, dct::maxCodeletSize);
    for (int n = 1; n <= std::max(maxCodelet, p.blockSize <= dct::maxCodeletSize ? p.blockSize : 0); ++n) {
        if (n == p.blockSize || (n <= maxCodelet && !scalarZ)) {
            s += dct::forwardCodelets[n];
            s += dct::inverseCodelets[n];
        }
        if (n <= maxCodelet && scalarZ) {
            s += scalarCodelet(dct::forwardCodelets[n]);
            s += scalarCodelet(dct::inverseCodelets[n]);
        }
    }
    s += shader::filterHead;
    s += zLineFunction(p.groupSize, scalarZ);
    s += shader::filterTail;
    return s;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pipelines

// Push constant blocks, laid out as declared by the kernels in shaders.h.
struct ConvertPush {
    uint32_t srcStride{};
    uint32_t dstOffset{};
};

struct BlockMatchPush {
    uint32_t firstFrame{};
    uint32_t numFrames{};
    uint32_t cur{};
    uint32_t refBase{};
    uint32_t refCount{};
};

struct FilterPush {
    uint32_t firstFrame{};
    uint32_t activeLo{};
    uint32_t activeHi{};
    uint32_t cur{};
    uint32_t refRow0{};
    uint32_t refRows{};
};

struct NormalizePush {
    uint32_t dstStride{};
    uint32_t accOffset{};
};

template <typename T>
concept PushConstants = std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> && sizeof(T) <= 128;

static_assert(PushConstants<ConvertPush> && sizeof(ConvertPush) == 8);
static_assert(PushConstants<BlockMatchPush> && sizeof(BlockMatchPush) == 20);
static_assert(PushConstants<FilterPush> && sizeof(FilterPush) == 24);
static_assert(PushConstants<NormalizePush> && sizeof(NormalizePush) == 8);

struct BindingLayout {
    uint32_t binding{};
    uint32_t count{};
};

struct Kernel {
    SetLayoutOwner setLayout;
    PipelineLayoutOwner pipelineLayout;
    PipelineOwner pipeline;
    const char* name{};
};

struct KernelSpec {
    const char* name{};
    std::string source;
    std::span<const BindingLayout> layout;
    uint32_t pushSize{};
    bool pinSubgroup{};
};

Kernel createKernel(const Gpu& gpu, VSCore* core, const KernelSpec& spec) {
    const VkDevice device = gpu.handles.device;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t descriptors{};
    for (const BindingLayout& b : spec.layout) {
        bindings.push_back({.binding = b.binding, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = b.count,
                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT});
        descriptors += b.count;
    }
    if (descriptors > gpu.maxStorageBuffers)
        fail("the device supports too few storage buffers per shader stage");
    const VkDescriptorSetLayoutCreateInfo setInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                  .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                  .bindingCount = static_cast<uint32_t>(bindings.size()),
                                                  .pBindings = bindings.data()};
    VkDescriptorSetLayout setLayout{VK_NULL_HANDLE};
    if (gpu.vk->vkCreateDescriptorSetLayout(device, &setInfo, nullptr, &setLayout) != VK_SUCCESS)
        fail("failed to create a descriptor set layout");
    Kernel kernel{.setLayout = {device, *gpu.vk, setLayout}, .name = spec.name};

    const VkPushConstantRange range{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = spec.pushSize};
    const VkPipelineLayoutCreateInfo layoutInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                                .setLayoutCount = 1,
                                                .pSetLayouts = &setLayout,
                                                .pushConstantRangeCount = 1,
                                                .pPushConstantRanges = &range};
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    if (gpu.vk->vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        fail("failed to create a pipeline layout");
    kernel.pipelineLayout = {device, *gpu.vk, pipelineLayout};

    std::array<char, errorSize> err{};
    const ShaderPtr shader{gpu.vkapi->compileGPUShader(core, slGLSL, spec.source.c_str(), err.data(), errorSize), {gpu.vkapi}};
    if (!shader)
        fail("shader compilation failed: "s + err.data());
    size_t codeBytes{};
    const uint32_t* code = gpu.vkapi->getGPUShaderCode(shader.get(), &codeBytes);

    const VkShaderModuleCreateInfo moduleInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = codeBytes, .pCode = code};
    const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .pNext = &moduleInfo, .requiredSubgroupSize = gpu.subgroupSize};
    const VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .pNext = spec.pinSubgroup ? static_cast<const void*>(&subgroupInfo) : &moduleInfo,
                  .flags = spec.pinSubgroup ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : VkPipelineShaderStageCreateFlags{},
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .pName = "main"},
        .layout = pipelineLayout};
    VkPipeline pipeline{VK_NULL_HANDLE};
    if (gpu.vk->vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        fail("failed to create a compute pipeline");
    kernel.pipeline = {device, *gpu.vk, pipeline};
    return kernel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Recording

// One recording of the filter's exec pool; abandoned unless submitted.
class ExecContext {
public:
    ExecContext(const VSVULKANAPI& vkapi, VSGPUExecPool* pool) : vkapi_{&vkapi} {
        std::array<char, errorSize> err{};
        context_ = vkapi.gpuExecAcquire(pool, err.data(), errorSize);
        if (!context_)
            fail(err.data());
    }
    ExecContext(const ExecContext&) = delete;
    ExecContext& operator=(const ExecContext&) = delete;
    ExecContext(ExecContext&&) = delete;
    ExecContext& operator=(ExecContext&&) = delete;
    ~ExecContext() {
        if (context_)
            vkapi_->gpuExecAbandon(context_);
    }

    VkCommandBuffer commandBuffer() const noexcept { return vkapi_->gpuExecCommandBuffer(context_); }
    void reads(const VSFrame* frame) const noexcept { vkapi_->gpuExecReadsFrame(context_, frame); }
    void writes(VSFrame* frame, int planes) const noexcept {
        for (int plane = 0; plane < planes; ++plane)
            vkapi_->gpuExecWritesPlane(context_, frame, plane);
    }

    // Hands a per-submission buffer to the context, which destroys it once the submission completes.
    VkBuffer retain(GpuBuffer&& buffer) const noexcept {
        vkapi_->gpuExecUsesBuffer(context_, buffer.owner.release());
        return buffer.buffer;
    }

    void submit() {
        std::array<char, errorSize> err{};
        if (vkapi_->gpuExecSubmit(std::exchange(context_, nullptr), nullptr, err.data(), errorSize))
            fail(err.data());
    }

private:
    const VSVULKANAPI* vkapi_{};
    VSGPUExecContext* context_{};
};

using PlaneBuffers = std::array<VkBuffer, maxPlanes>;

struct DescriptorWrite {
    uint32_t binding{};
    std::array<VkDescriptorBufferInfo, maxPlanes> buffers{};
    uint32_t count{};
};

DescriptorWrite bindBuffers(uint32_t binding, VkBuffer buffer) {
    return {.binding = binding, .buffers = {{{.buffer = buffer, .range = VK_WHOLE_SIZE}}}, .count = 1};
}

DescriptorWrite bindBuffers(uint32_t binding, const PlaneBuffers& buffers, int planes) {
    DescriptorWrite b{.binding = binding, .count = static_cast<uint32_t>(planes)};
    for (int plane = 0; plane < planes; ++plane)
        b.buffers[plane] = {.buffer = buffers[plane], .range = VK_WHOLE_SIZE};
    return b;
}

struct Groups {
    int x{};
    int y{1};
};

// GPU time of every dispatch from timestamp queries, for profiling: enabled by the environment variable BM3DVK_PROFILE set to anything but empty or 0, and
// logged when the filter is freed. Dispatches past the capacity go untimed.
class KernelTimer {
public:
    explicit KernelTimer(const Gpu& gpu) : vk_{gpu.vk}, device_{gpu.handles.device}, names_(capacity / 2) {
        VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vk_->vkGetPhysicalDeviceProperties2(gpu.handles.physicalDevice, &props);
        period_ = props.properties.limits.timestampPeriod;
        uint32_t families{};
        vk_->vkGetPhysicalDeviceQueueFamilyProperties2(gpu.handles.physicalDevice, &families, nullptr);
        std::vector<VkQueueFamilyProperties2> familyProps(families, {.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2});
        vk_->vkGetPhysicalDeviceQueueFamilyProperties2(gpu.handles.physicalDevice, &families, familyProps.data());
        const uint32_t validBits = familyProps[gpu.handles.computeQueueFamily].queueFamilyProperties.timestampValidBits;
        if (validBits == 0)
            fail("the compute queue does not support timestamps, unset BM3DVK_PROFILE");
        timestampMask_ = validBits >= 64 ? ~uint64_t{} : (uint64_t{1} << validBits) - 1;
        const VkQueryPoolCreateInfo info{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = capacity};
        if (vk_->vkCreateQueryPool(device_, &info, nullptr, &pool_) != VK_SUCCESS)
            fail("failed to create a timestamp query pool");
        vk_->vkResetQueryPool(device_, pool_, 0, capacity);
    }
    KernelTimer(const KernelTimer&) = delete;
    KernelTimer& operator=(const KernelTimer&) = delete;
    KernelTimer(KernelTimer&&) = delete;
    KernelTimer& operator=(KernelTimer&&) = delete;
    ~KernelTimer() { vk_->vkDestroyQueryPool(device_, pool_, nullptr); }

    // The query pair of a dispatch about to be recorded, its first timestamp written.
    std::optional<uint32_t> begin(VkCommandBuffer cmd, const char* name) {
        const uint32_t pair = next_.fetch_add(1, std::memory_order_relaxed);
        if (pair >= capacity / 2)
            return std::nullopt;
        names_[pair] = name;
        vk_->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, pair * 2);
        return pair;
    }

    void end(VkCommandBuffer cmd, uint32_t pair) const { vk_->vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool_, pair * 2 + 1); }

    // Average and total time per kernel; call once every timed submission has completed.
    std::string report() const {
        const uint32_t pairs = std::min(next_.load(), capacity / 2);
        if (pairs == 0)
            return "BM3D profile: no dispatch was timed";
        struct Result {
            uint64_t value;
            uint64_t available;
        };
        std::vector<Result> results(size_t{pairs} * 2);
        vk_->vkGetQueryPoolResults(device_, pool_, 0, pairs * 2, results.size() * sizeof(Result), results.data(), sizeof(Result),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        std::map<std::string_view, std::pair<double, int>> totals;
        for (uint32_t pair = 0; pair < pairs; ++pair) {
            const Result& first = results[size_t{pair} * 2];
            const Result& second = results[size_t{pair} * 2 + 1];
            if (first.available && second.available) {
                auto& [ms, count] = totals[names_[pair]];
                ms += static_cast<double>((second.value - first.value) & timestampMask_) * period_ * 1e-6;
                ++count;
            }
        }
        std::string s = "BM3D profile:";
        for (const auto& [name, total] : totals)
            s += "\n  "s.append(name) + ": " + std::to_string(total.first / total.second) + " ms x " + std::to_string(total.second) + " = " +
                 std::to_string(total.first) + " ms";
        return s;
    }

private:
    static constexpr uint32_t capacity = 1 << 17;

    const VSVulkanFunctions* vk_{};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueryPool pool_{VK_NULL_HANDLE};
    float period_{};
    uint64_t timestampMask_{};
    std::atomic<uint32_t> next_{};
    std::vector<const char*> names_;
};

bool profilingRequested() {
    const char* value = std::getenv("BM3DVK_PROFILE");
    return value && *value && std::string_view{value} != "0";
}

class Recorder {
public:
    Recorder(const Gpu& gpu, VkCommandBuffer cmd, KernelTimer* timer) : vk_{gpu.vk}, cmd_{cmd}, timer_{timer} {}

    template <PushConstants Push>
    void dispatch(const Kernel& kernel, std::span<const DescriptorWrite> bindings, const Push& push, Groups groups) const {
        // Labels name the kernels in GPU captures; the entry points exist only alongside debug utils.
        if (vk_->vkCmdBeginDebugUtilsLabelEXT) {
            const VkDebugUtilsLabelEXT label{.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, .pLabelName = kernel.name};
            vk_->vkCmdBeginDebugUtilsLabelEXT(cmd_, &label);
        }
        const std::optional<uint32_t> timed = timer_ ? timer_->begin(cmd_, kernel.name) : std::nullopt;
        vk_->vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline.get());
        std::array<VkWriteDescriptorSet, maxBindings> writes{};
        for (size_t i = 0; i < bindings.size(); ++i)
            writes[i] = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                         .dstBinding = bindings[i].binding,
                         .descriptorCount = bindings[i].count,
                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                         .pBufferInfo = bindings[i].buffers.data()};
        vk_->vkCmdPushDescriptorSet(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipelineLayout.get(), 0, static_cast<uint32_t>(bindings.size()),
                                    writes.data());
        const VkPushConstantsInfo pushInfo{.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
                                           .layout = kernel.pipelineLayout.get(),
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .size = sizeof(Push),
                                           .pValues = &push};
        vk_->vkCmdPushConstants2(cmd_, &pushInfo);
        vk_->vkCmdDispatch(cmd_, static_cast<uint32_t>(groups.x), static_cast<uint32_t>(groups.y), 1);
        if (timed)
            timer_->end(cmd_, *timed);
        if (vk_->vkCmdEndDebugUtilsLabelEXT)
            vk_->vkCmdEndDebugUtilsLabelEXT(cmd_);
    }

    // Orders every compute and transfer access before it against every one after it, including earlier submissions.
    void barrier() const {
        const VkMemoryBarrier2 memoryBarrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                             .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                             .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                             .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                             .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                                              VK_ACCESS_2_TRANSFER_WRITE_BIT};
        const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memoryBarrier};
        vk_->vkCmdPipelineBarrier2(cmd_, &dependency);
    }

    void zero(VkBuffer buffer, int64_t offset, int64_t size) const {
        vk_->vkCmdFillBuffer(cmd_, buffer, static_cast<VkDeviceSize>(offset), static_cast<VkDeviceSize>(size), 0);
    }

private:
    const VSVulkanFunctions* vk_{};
    VkCommandBuffer cmd_{};
    KernelTimer* timer_{};
};

PlaneBuffers framePlanes(const VSVULKANAPI& vkapi, const VSFrame* frame, int planes) {
    PlaneBuffers buffers{};
    for (int plane = 0; plane < planes; ++plane) {
        VSVulkanPlaneInfo info{};
        if (vkapi.getGPUPlane(frame, plane, &info))
            fail("the frame is not GPU resident");
        buffers[plane] = info.buffer;
    }
    return buffers;
}

uint32_t elementStride(const VSAPI& vsapi, const VSFrame* frame) {
    return static_cast<uint32_t>(vsapi.getStride(frame, 0) / vsapi.getVideoFrameFormat(frame)->bytesPerSample);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Filter

std::array<BindingLayout, 2> imageLayout(int planes) {
    return {{{bindIn, static_cast<uint32_t>(planes)}, {bindOut, static_cast<uint32_t>(planes)}}};
}

constexpr std::array<BindingLayout, 3> blockMatchLayout{{{bindWindow, 1}, {bindMatches, 1}, {bindCounts, 1}}};

std::vector<BindingLayout> filterLayout(const Params& p, const Plan& plan) {
    const auto planes = static_cast<uint32_t>(plan.planes);
    std::vector<BindingLayout> layout{{bindFilterSrc, planes}};
    if (p.wiener)
        layout.push_back({bindFilterRef, planes});
    layout.insert(layout.end(), {{bindFilterMatches, 1}, {bindFilterCounts, 1}, {bindFilterAcc, planes}});
    if (plan.globalGroup)
        layout.push_back({bindFilterScratch, 1});
    if (plan.pingpong)
        layout.push_back({bindFilterCos, 1});
    return layout;
}

// Working buffers of one window of converted frames, one accumulation of estimates and one batch of matches.
struct WorkBuffers {
    PlaneBuffers srcWindow{};
    PlaneBuffers refWindow{};
    PlaneBuffers acc{};
    VkBuffer matches{VK_NULL_HANDLE};
    VkBuffer counts{VK_NULL_HANDLE};
    VkBuffer scratch{VK_NULL_HANDLE};
};

// One reference frame's block matching and filtering, over window frames [firstFrame, firstFrame + numFrames) of which the reference frame is the cur-th;
// estimates land in frames [activeLo, activeHi] only.
struct Step {
    int firstFrame{};
    int numFrames{1};
    int cur{};
    int activeLo{};
    int activeHi{};
};

// Temporal filtering scatters the estimates of reference frame m into frames m - radius .. m + radius. Accumulators of those frames live in a ring, so each
// source frame is converted once and each frame is emitted once all of its reference frames have been processed. getFrame runs fmUnordered, which serializes
// every access to this state.
struct TemporalCursor {
    bool valid{};
    int nextRef{};
    int emittedUpTo{};
    int firstComplete{};
    int convertedUpTo{};
};

struct EmittedFrame {
    OutFramePtr frame;
    uint64_t age{};
};

class Filter {
public:
    Filter(NodePtr node, NodePtr refNode, const Params& params, const Gpu& gpu, VSCore* core, const VSAPI& vsapi)
        : vsapi_{&vsapi}, node_{std::move(node)}, refNode_{std::move(refNode)}, vi_{*vsapi.getVideoInfo(node_.get())}, params_{params}, gpu_{gpu},
          plan_{makePlan(params_, vi_, gpu_)}, preamble_{buildPreamble(params_, plan_, gpu_)},
          convert_{createKernel(gpu_, core, {"BM3D convert", kernelSource(shader::convert), imageLayout(plan_.planes), sizeof(ConvertPush), false})},
          blockMatch_{createKernel(gpu_, core, {"BM3D blockMatch", kernelSource(shader::blockMatch), blockMatchLayout, sizeof(BlockMatchPush), true})},
          filter_{createKernel(gpu_, core, {"BM3D filter", filterSource(params_, preamble_, !(gpu_.fp16Math && plan_.fp16Input)),
                                            filterLayout(params_, plan_), sizeof(FilterPush), true})},
          normalize_{createKernel(gpu_, core, {"BM3D normalize", kernelSource(shader::normalize), imageLayout(plan_.planes), sizeof(NormalizePush), false})},
          maxEmitted_{emittedCapacity(core)}, returned_(static_cast<size_t>(vi_.numFrames)),
          timer_{profilingRequested() ? std::make_unique<KernelTimer>(gpu_) : nullptr}, pool_{createPool(core)} {
        if (plan_.pingpong)
            uploadCosTable(core);
        if (params_.radius > 0)
            persistent_ = createWorkBuffers(core, [this](GpuBuffer&& buffer) { return persistentBuffers_.emplace_back(std::move(buffer)).buffer; });
    }

    std::vector<VSFilterDependency> dependencies() const {
        std::vector<VSFilterDependency> deps{{node_.get(), params_.radius > 0 ? rpGeneral : rpStrictSpatial}};
        if (refNode_)
            deps.push_back({refNode_.get(), params_.radius > 0 ? rpGeneral : rpStrictSpatial});
        return deps;
    }

    const VSVideoInfo& videoInfo() const noexcept { return vi_; }
    bool temporal() const noexcept { return params_.radius > 0; }

    void request(int n, VSFrameContext* frameCtx) const {
        const int lo = std::max(n - params_.radius * 2, 0);
        const int hi = n + std::min(params_.radius * 2, vi_.numFrames - 1 - n);
        for (int f = lo; f <= hi; ++f) {
            vsapi_->requestFrameFilter(f, node_.get(), frameCtx);
            if (refNode_)
                vsapi_->requestFrameFilter(f, refNode_.get(), frameCtx);
        }
    }

    // Sigma and th_mse are normalized by the matrix, so opponent color space input must be filtered with matrix=100.
    void checkMatrix(int n, VSFrameContext* frameCtx) const {
        if (params_.matrix == matrixOPP)
            return;
        const FramePtr src{vsapi_->getFrameFilter(n, node_.get(), frameCtx), {vsapi_}};
        int err{};
        if (vsapi_->mapGetInt(vsapi_->getFramePropertiesRO(src.get()), "BM3D_OPP", 0, &err) == 1)
            fail("the input frame property \"BM3D_OPP=1\" indicates opponent color space input, \"matrix=100\" must be specified");
    }

    OutFramePtr spatialFrame(int n, VSFrameContext* frameCtx, VSCore* core) const;
    OutFramePtr temporalFrame(int n, VSFrameContext* frameCtx, VSCore* core);

    // Waits out every submission; false when they may still be running, so nothing they use may be destroyed.
    bool drain(VSCore* core) const {
        std::array<char, errorSize> err{};
        if (vsGPUDrainSafeToDestroy(gpu_.vkapi->gpuExecPoolWaitIdle(pool_.get(), err.data(), errorSize)))
            return true;
        vsapi_->logMessage(mtWarning, err.data(), core);
        return false;
    }

    // Logs the kernel times once every submission has completed, when profiling.
    void reportProfile(VSCore* core) const {
        if (timer_)
            vsapi_->logMessage(mtInformation, timer_->report().c_str(), core);
    }

    void freeNodes() noexcept {
        node_.reset();
        refNode_.reset();
    }

private:
    struct Sources {
        int lo{};
        std::vector<FramePtr> src;
        std::vector<FramePtr> ref;
    };

    std::string kernelSource(std::string_view kernel) const { return preamble_ + std::string{shader::common} + std::string{kernel}; }

    size_t emittedCapacity(VSCore* core) const {
        VSCoreInfo2 info{};
        vsapi_->getCoreInfo2(core, &info);
        return static_cast<size_t>(info.numThreads + params_.radius + 1);
    }

    PoolPtr createPool(VSCore* core) const {
        std::array<char, errorSize> err{};
        PoolPtr pool{gpu_.vkapi->createGPUExecPool(core, vqCompute, err.data(), errorSize), {gpu_.vkapi}};
        if (!pool)
            fail(err.data());
        return pool;
    }

    Sources getSources(int lo, int hi, VSFrameContext* frameCtx) const;
    void uploadCosTable(VSCore* core);
    template <typename Keep>
    WorkBuffers createWorkBuffers(VSCore* core, Keep keep) const;
    void recordConvert(const Recorder& rec, const ExecContext& ctx, const WorkBuffers& work, const Sources& sources, int frame) const;
    void recordStep(const Recorder& rec, const WorkBuffers& work, const Step& step) const;
    void recordNormalize(const Recorder& rec, const WorkBuffers& work, VSFrame* dst, int frame) const;
    OutFramePtr newFrame(const VSFrame* propSrc, VSCore* core) const;
    bool canContinue(int n) const;
    TemporalCursor restart(const Recorder& rec, int n) const;

    const VSAPI* vsapi_{};
    NodePtr node_;
    NodePtr refNode_;
    VSVideoInfo vi_{};
    Params params_;
    Gpu gpu_;
    Plan plan_;
    std::string preamble_;
    Kernel convert_;
    Kernel blockMatch_;
    Kernel filter_;
    Kernel normalize_;
    GpuBuffer cosTable_;
    std::vector<GpuBuffer> persistentBuffers_;
    WorkBuffers persistent_;
    size_t maxEmitted_{};
    TemporalCursor cursor_;
    std::map<int, EmittedFrame> emitted_;
    uint64_t emitAge_{};
    // Frames returned at least once; a restart's cursor passing one again does not emit it again.
    std::vector<bool> returned_;
    std::unique_ptr<KernelTimer> timer_;
    // Declared last so that it is destroyed first: freeing the pool drains every submission still using the members above.
    PoolPtr pool_;
};

Filter::Sources Filter::getSources(int lo, int hi, VSFrameContext* frameCtx) const {
    Sources sources{.lo = lo};
    sources.src.reserve(static_cast<size_t>(hi - lo + 1));
    sources.ref.reserve(refNode_ ? static_cast<size_t>(hi - lo + 1) : 0);
    for (int f = lo; f <= hi; ++f) {
        sources.src.emplace_back(vsapi_->getFrameFilter(f, node_.get(), frameCtx), FrameDeleter{vsapi_});
        if (refNode_)
            sources.ref.emplace_back(vsapi_->getFrameFilter(f, refNode_.get(), frameCtx), FrameDeleter{vsapi_});
    }
    return sources;
}

void Filter::uploadCosTable(VSCore* core) {
    const std::vector<float>& values = plan_.cosTable.values;
    const auto bytes = static_cast<int64_t>(values.size() * sizeof(float));
    cosTable_ = createBuffer(gpu_, core, bytes);

    std::array<char, errorSize> err{};
    VSVulkanBufferInfo stagingInfo{};
    BufferPtr staging{gpu_.vkapi->createGPUBuffer(core, static_cast<VkDeviceSize>(bytes), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &stagingInfo,
                                                  err.data(), errorSize),
                      {gpu_.vkapi}};
    if (!staging)
        fail(err.data());
    std::ranges::copy(values, std::span{static_cast<float*>(stagingInfo.mapped), values.size()}.begin());

    ExecContext ctx{*gpu_.vkapi, pool_.get()};
    const VkBuffer src = ctx.retain({std::move(staging), stagingInfo.buffer});
    const VkBufferCopy2 region{.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .size = static_cast<VkDeviceSize>(bytes)};
    const VkCopyBufferInfo2 copy{
        .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2, .srcBuffer = src, .dstBuffer = cosTable_.buffer, .regionCount = 1, .pRegions = &region};
    gpu_.vk->vkCmdCopyBuffer2(ctx.commandBuffer(), &copy);
    const VkMemoryBarrier2 memoryBarrier{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                                         .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                                         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                         .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT};
    const VkDependencyInfo dependency{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &memoryBarrier};
    gpu_.vk->vkCmdPipelineBarrier2(ctx.commandBuffer(), &dependency);
    ctx.submit();
    if (const int drained = gpu_.vkapi->gpuExecPoolWaitIdle(pool_.get(), err.data(), errorSize); drained != gdDrained) {
        // A copy that may still be running keeps its destination: the table is leaked rather than destroyed.
        if (!vsGPUDrainSafeToDestroy(drained))
            cosTable_.owner.release();
        fail(err.data());
    }
}

// The working buffers for plan_.slots frames, each handed to keep, which takes ownership and returns its handle.
template <typename Keep>
WorkBuffers Filter::createWorkBuffers(VSCore* core, Keep keep) const {
    const auto buffer = [&](int64_t bytes) { return keep(createBuffer(gpu_, core, bytes)); };
    WorkBuffers work;
    for (int plane = 0; plane < plan_.planes; ++plane) {
        work.srcWindow[plane] = buffer(windowBytes(params_, plan_, plan_.slots));
        if (params_.wiener)
            work.refWindow[plane] = buffer(windowBytes(params_, plan_, plan_.slots));
        work.acc[plane] = buffer(planeElements(plan_) * accBytes * plan_.slots);
    }
    work.matches = buffer(batchRefs(plan_) * params_.groupSize * matchBytes(params_));
    work.counts = buffer(batchRefs(plan_) * countBytes);
    if (plan_.globalGroup)
        work.scratch = buffer(plan_.groupBytes * plan_.filterGroupsX * plan_.planes);
    return work;
}

// Converts source frame `frame` into its window slot.
void Filter::recordConvert(const Recorder& rec, const ExecContext& ctx, const WorkBuffers& work, const Sources& sources, int frame) const {
    const auto offset = static_cast<uint32_t>(planeElements(plan_) * (frame % plan_.slots));
    const auto convertOne = [&](const VSFrame* src, const PlaneBuffers& window) {
        ctx.reads(src);
        const std::array bindings{bindBuffers(bindIn, framePlanes(*gpu_.vkapi, src, plan_.planes), plan_.planes), bindBuffers(bindOut, window, plan_.planes)};
        rec.dispatch(convert_, bindings, ConvertPush{.srcStride = elementStride(*vsapi_, src), .dstOffset = offset},
                     {divCeil(plan_.width, imageTile), divCeil(plan_.height, imageTile)});
    };
    convertOne(sources.src[frame - sources.lo].get(), work.srcWindow);
    if (params_.wiener)
        convertOne(sources.ref[frame - sources.lo].get(), work.refWindow);
}

void Filter::recordStep(const Recorder& rec, const WorkBuffers& work, const Step& step) const {
    const PlaneBuffers& matchWindow = params_.wiener ? work.refWindow : work.srcWindow;
    const std::array bmBindings{bindBuffers(bindWindow, matchWindow[0]), bindBuffers(bindMatches, work.matches), bindBuffers(bindCounts, work.counts)};

    std::array<DescriptorWrite, maxBindings> bindings{};
    size_t count{};
    bindings[count++] = bindBuffers(bindFilterSrc, work.srcWindow, plan_.planes);
    if (params_.wiener)
        bindings[count++] = bindBuffers(bindFilterRef, work.refWindow, plan_.planes);
    bindings[count++] = bindBuffers(bindFilterMatches, work.matches);
    bindings[count++] = bindBuffers(bindFilterCounts, work.counts);
    bindings[count++] = bindBuffers(bindFilterAcc, work.acc, plan_.planes);
    if (plan_.globalGroup)
        bindings[count++] = bindBuffers(bindFilterScratch, work.scratch);
    if (plan_.pingpong)
        bindings[count++] = bindBuffers(bindFilterCos, cosTable_.buffer);

    for (int row0 = 0; row0 < plan_.refRows; row0 += plan_.batchRows) {
        const int rows = std::min(plan_.batchRows, plan_.refRows - row0);
        const int refCount = rows * plan_.refCols;
        const BlockMatchPush bmPush{.firstFrame = static_cast<uint32_t>(step.firstFrame),
                                    .numFrames = static_cast<uint32_t>(step.numFrames),
                                    .cur = static_cast<uint32_t>(step.cur),
                                    .refBase = static_cast<uint32_t>(row0 * plan_.refCols),
                                    .refCount = static_cast<uint32_t>(refCount)};
        rec.dispatch(blockMatch_, bmBindings, bmPush, {std::min(divCeil(refCount, plan_.bmSubgroups), gpu_.maxGroupCountX)});
        rec.barrier();

        const FilterPush push{.firstFrame = static_cast<uint32_t>(step.firstFrame),
                              .activeLo = static_cast<uint32_t>(step.activeLo),
                              .activeHi = static_cast<uint32_t>(step.activeHi),
                              .cur = static_cast<uint32_t>(step.cur),
                              .refRow0 = static_cast<uint32_t>(row0),
                              .refRows = static_cast<uint32_t>(rows)};
        const int tiles = divCeil(rows, plan_.tileRefs) * divCeil(plan_.refCols, plan_.tileRefs);
        rec.dispatch(filter_, std::span{bindings.data(), count}, push, {std::min(plan_.filterGroupsX, tiles), plan_.planes});
        rec.barrier();
    }
}

// Writes the estimate of `frame`, held in its accumulator slot, to dst.
void Filter::recordNormalize(const Recorder& rec, const WorkBuffers& work, VSFrame* dst, int frame) const {
    const std::array bindings{bindBuffers(bindIn, work.acc, plan_.planes), bindBuffers(bindOut, framePlanes(*gpu_.vkapi, dst, plan_.planes), plan_.planes)};
    const NormalizePush push{.dstStride = elementStride(*vsapi_, dst), .accOffset = static_cast<uint32_t>(planeElements(plan_) * 2 * (frame % plan_.slots))};
    rec.dispatch(normalize_, bindings, push, {divCeil(plan_.width, imageTile), divCeil(plan_.height, imageTile)});
}

OutFramePtr Filter::newFrame(const VSFrame* propSrc, VSCore* core) const {
    OutFramePtr dst{gpu_.vkapi->newGPUVideoFrame(&vi_.format, vi_.width, vi_.height, propSrc, core), {vsapi_}};
    if (!dst)
        fail("failed to allocate an output frame");
    return dst;
}

OutFramePtr Filter::spatialFrame(int n, VSFrameContext* frameCtx, VSCore* core) const {
    const Sources sources = getSources(n, n, frameCtx);
    OutFramePtr dst = newFrame(sources.src[0].get(), core);

    ExecContext ctx{*gpu_.vkapi, pool_.get()};
    const WorkBuffers work = createWorkBuffers(core, [&ctx](GpuBuffer&& buffer) { return ctx.retain(std::move(buffer)); });
    const Recorder rec{gpu_, ctx.commandBuffer(), timer_.get()};
    for (int plane = 0; plane < plan_.planes; ++plane)
        rec.zero(work.acc[plane], 0, planeElements(plan_) * accBytes);
    recordConvert(rec, ctx, work, sources, n);
    rec.barrier();
    recordStep(rec, work, {});
    ctx.writes(dst.get(), plan_.planes);
    recordNormalize(rec, work, dst.get(), n);
    ctx.submit();
    return dst;
}

bool Filter::canContinue(int n) const {
    return cursor_.valid && n >= cursor_.emittedUpTo && cursor_.nextRef >= n - params_.radius;
}

// Restarts the ring so that frame n is the first complete one; only frame 0 needs no reference frames before it.
TemporalCursor Filter::restart(const Recorder& rec, int n) const {
    const int first = std::max(n - params_.radius, 0);
    const int firstComplete = first == 0 ? 0 : first + params_.radius;
    for (int plane = 0; plane < plan_.planes; ++plane)
        rec.zero(persistent_.acc[plane], 0, planeElements(plan_) * accBytes * plan_.slots);
    return {.valid = true, .nextRef = first, .emittedUpTo = firstComplete, .firstComplete = firstComplete,
            .convertedUpTo = std::max(first - params_.radius, 0)};
}

OutFramePtr Filter::temporalFrame(int n, VSFrameContext* frameCtx, VSCore* core) {
    returned_[n] = true;
    if (const auto it = emitted_.find(n); it != emitted_.end()) {
        OutFramePtr frame = std::move(it->second.frame);
        emitted_.erase(it);
        return frame;
    }

    const int radius = params_.radius;
    const int last = vi_.numFrames - 1;
    const Sources sources = getSources(std::max(n - radius * 2, 0), n + std::min(radius * 2, last - n), frameCtx);

    ExecContext ctx{*gpu_.vkapi, pool_.get()};
    const Recorder rec{gpu_, ctx.commandBuffer(), timer_.get()};
    rec.barrier();
    // The cursor is committed only once the submission is made, so a failure leaves a restart for the next call.
    TemporalCursor cursor = canContinue(n) ? cursor_ : restart(rec, n);
    cursor_.valid = false;

    std::vector<std::pair<int, OutFramePtr>> frames;
    while (cursor.emittedUpTo <= n) {
        const int m = cursor.nextRef++;
        for (; cursor.convertedUpTo <= m + std::min(radius, last - m); ++cursor.convertedUpTo)
            recordConvert(rec, ctx, persistent_, sources, cursor.convertedUpTo);
        rec.barrier();

        const int firstFrame = std::max(m - radius, 0);
        const int lastFrame = m + std::min(radius, last - m);
        recordStep(rec, persistent_,
                   {.firstFrame = firstFrame, .numFrames = lastFrame - firstFrame + 1, .cur = m - firstFrame,
                    .activeLo = std::max(firstFrame, cursor.firstComplete), .activeHi = lastFrame});

        // A frame already returned, passed again after a restart, is only cleared from the ring.
        for (const int retireHi = m == last ? last : m - radius; cursor.emittedUpTo <= retireHi; ++cursor.emittedUpTo) {
            const int f = cursor.emittedUpTo;
            if (f == n || !returned_[f]) {
                OutFramePtr dst = newFrame(sources.src[f - sources.lo].get(), core);
                ctx.writes(dst.get(), plan_.planes);
                recordNormalize(rec, persistent_, dst.get(), f);
                rec.barrier();
                frames.emplace_back(f, std::move(dst));
            }
            for (int plane = 0; plane < plan_.planes; ++plane)
                rec.zero(persistent_.acc[plane], planeElements(plan_) * accBytes * (f % plan_.slots), planeElements(plan_) * accBytes);
            rec.barrier();
        }
    }
    ctx.submit();
    cursor_ = cursor;

    OutFramePtr result;
    for (auto& [f, frame] : frames) {
        if (f == n)
            result = std::move(frame);
        else
            emitted_.insert_or_assign(f, EmittedFrame{std::move(frame), ++emitAge_});
    }
    // Frames emitted ahead are kept for requests already on their way; the oldest go first.
    while (emitted_.size() > maxEmitted_)
        emitted_.erase(std::ranges::min_element(emitted_, {}, [](const auto& entry) { return entry.second.age; }));
    return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// VapourSynth callbacks

const VSFrame* VS_CC filterGetFrame(int n, int activationReason, void* instanceData, void**, VSFrameContext* frameCtx, VSCore* core,
                                    const VSAPI* vsapi) noexcept {
    auto& d = *static_cast<Filter*>(instanceData);
    try {
        if (activationReason == arInitial) {
            d.request(n, frameCtx);
            return nullptr;
        }
        if (activationReason != arAllFramesReady)
            return nullptr;
        d.checkMatrix(n, frameCtx);
        return (d.temporal() ? d.temporalFrame(n, frameCtx, core) : d.spatialFrame(n, frameCtx, core)).release();
    } catch (const std::exception& e) {
        vsapi->setFilterError(e.what(), frameCtx);
    } catch (...) {
        vsapi->setFilterError("BM3D: unknown error", frameCtx);
    }
    return nullptr;
}

void VS_CC filterFree(void* instanceData, VSCore* core, const VSAPI* vsapi) noexcept {
    std::unique_ptr<Filter> d{static_cast<Filter*>(instanceData)};
    // Submissions that may still be running keep the GPU objects they use: the filter is leaked rather than destroyed.
    if (!d->drain(core)) {
        d.release()->freeNodes();
        return;
    }
    try {
        d->reportProfile(core);
    } catch (const std::exception& e) {
        vsapi->logMessage(mtWarning, e.what(), core);
    }
}

void checkClip(const VSVideoInfo& vi, std::string_view name) {
    if (!vsh::isConstantVideoFormat(&vi))
        fail(std::string{name} + " must have a constant format and dimensions");
    if ((vi.format.colorFamily != cfGray && vi.format.colorFamily != cfRGB) || vi.format.sampleType != stFloat)
        fail(std::string{name} + " must be Gray or RGB of 16/32-bit float");
    if (vi.width > maxDimension || vi.height > maxDimension)
        fail(std::string{name} + " must not exceed " + std::to_string(maxDimension) + " in width or height");
}

void VS_CC filterCreate(const VSMap* in, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) noexcept {
    try {
        NodePtr node{vsapi->mapGetNode(in, "clip", 0, nullptr), {vsapi}};
        const VSVideoInfo& vi = *vsapi->getVideoInfo(node.get());
        checkClip(vi, "clip");

        int err{};
        NodePtr refNode{vsapi->mapGetNode(in, "ref", 0, &err), {vsapi}};
        if (refNode) {
            const VSVideoInfo& rvi = *vsapi->getVideoInfo(refNode.get());
            checkClip(rvi, "ref");
            if (!vsh::isSameVideoInfo(&rvi, &vi) || rvi.numFrames != vi.numFrames)
                fail("ref must have the same format, dimensions and number of frames as clip");
        }

        const Params params = parseParams(ArgReader{*in, *vsapi}, vi, refNode != nullptr);
        if (std::ranges::all_of(std::span{params.sigma}.first(vi.format.numPlanes), [](double s) { return s == 0.0; })) {
            vsapi->mapConsumeNode(out, "clip", node.release(), maAppend);
            return;
        }

        auto d = std::make_unique<Filter>(std::move(node), std::move(refNode), params, queryGpu(core, *vsapi), core, *vsapi);
        const std::vector<VSFilterDependency> deps = d->dependencies();
        VSNode* filterNode = vsapi->createVideoFilterEx2("BM3D", &d->videoInfo(), filterGetFrame, filterFree, d->temporal() ? fmUnordered : fmParallel,
                                                         ffGPUOutput, deps.data(), static_cast<int>(deps.size()), d.get(), core);
        if (!filterNode)
            fail("failed to create the filter");
        d.release();
        vsapi->mapConsumeNode(out, "clip", filterNode, maAppend);
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, e.what());
    } catch (...) {
        vsapi->mapSetError(out, "BM3D: unknown error");
    }
}

} // namespace

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) noexcept {
    vspapi->configPlugin("com.holywu.bm3dvk2",
                         "bm3dvk2",
                         "Block-Matching and 3D filtering, using Vulkan compute",
                         VS_MAKE_VERSION(1, 0),
                         VAPOURSYNTH_API_VERSION,
                         0,
                         plugin);

    vspapi->registerFunction("BM3D",
                             "clip:vnode:gpu;"
                             "ref:vnode:gpu:opt;"
                             "profile:data:opt;"
                             "sigma:float[]:opt;"
                             "radius:int:opt;"
                             "block_size:int:opt;"
                             "block_step:int:opt;"
                             "group_size:int:opt;"
                             "bm_range:int:opt;"
                             "bm_step:int:opt;"
                             "ps_num:int:opt;"
                             "ps_range:int:opt;"
                             "ps_step:int:opt;"
                             "th_mse:float:opt;"
                             "hard_thr:float:opt;"
                             "matrix:int:opt;",
                             "clip:vnode:gpu;",
                             filterCreate,
                             nullptr,
                             plugin);
}
