#ifndef BM3DVK_SHADERS_H
#define BM3DVK_SHADERS_H

#include <string_view>

// GLSL kernels. Each is completed by a preamble of #defines built per filter instance; see buildPreamble in bm3dvk.cpp.
namespace shader {

// Shared by every kernel: element types, float atomics, the window/accumulator addressing and the reference block grid.
inline constexpr std::string_view common = R"glsl(
#define PLANE_SIZE (WIDTH * HEIGHT)
// Elements of an accumulator slot: the numerator plane, then the denominator plane.
#define ACC_SLOT (PLANE_SIZE * ACC_PLANES)
#define INF uintBitsToFloat(0x7F800000u)

#if FLOAT_ATOMICS
#define ACC_T float
#else
#define ACC_T uint
#endif

// A group member in the matches buffer: its packed position, and its window frame when filtering temporally.
#if RADIUS > 0
#define MATCH_T uvec2
#else
#define MATCH_T uint
#endif

uint packPos(ivec2 p) { return (uint(p.y) << 16) | uint(p.x); }
ivec2 unpackPos(uint p) { return ivec2(p & 0xFFFFu, p >> 16); }
uint slotOf(uint frame) { return frame % SLOTS; }
uint accNum(uint frame, ivec2 p) { return slotOf(frame) * ACC_SLOT + uint(p.y) * WIDTH + uint(p.x); }
uint accDen(uint frame, ivec2 p) { return accNum(frame, p) + PLANE_SIZE; }
ivec2 refPosition(uint row, uint col) { return min(ivec2(col, row) * BLOCK_STEP, ivec2(WIDTH, HEIGHT) - BLOCK); }
)glsl";

// RGB -> OPP (or a plain copy for Gray) from frame planes into one slot of a window, of the planes in pc.planes (always plane 0).
inline constexpr std::string_view convert = R"glsl(
layout(local_size_x = IMAGE_TILE, local_size_y = IMAGE_TILE) in;

layout(set = 0, binding = 0, std430) readonly buffer Src { STORE_T data[]; } src[PLANES];
layout(set = 0, binding = 1, std430) writeonly buffer Dst { STORE_T data[]; } dst[PLANES];

layout(push_constant) uniform Push {
    uint srcStride;
    uint frame;
    uint planes;
} pc;

void main() {
    const uvec2 p = gl_GlobalInvocationID.xy;
    if (p.x >= WIDTH || p.y >= HEIGHT)
        return;

    const uint s = p.y * pc.srcStride + p.x;
    const uint d = slotOf(pc.frame) * PLANE_SIZE + p.y * WIDTH + p.x;
#if PLANES == 3
    const float r = float(src[0].data[s]);
    const float g = float(src[1].data[s]);
    const float b = float(src[2].data[s]);
    dst[0].data[d] = STORE_T((r + g + b) * (1.0 / 3.0));
    if ((pc.planes & 2u) != 0u)
        dst[1].data[d] = STORE_T((r - b) * 0.5);
    if ((pc.planes & 4u) != 0u)
        dst[2].data[d] = STORE_T((r - g * 2.0 + b) * 0.25);
#else
    dst[0].data[d] = src[0].data[s];
#endif
}
)glsl";

// Accumulated numerator / denominator -> output frame planes (OPP -> RGB for RGB). The denominator is accumulated per block at its top-left position; the
// denominator of a pixel is the sum over the BLOCK x BLOCK positions whose blocks cover it, computed separably from a shared tile. Planes left unfiltered
// (outside ACTIVE_MASK) are taken from the source window.
inline constexpr std::string_view normalize = R"glsl(
layout(local_size_x = IMAGE_TILE, local_size_y = IMAGE_TILE) in;

layout(set = 0, binding = 0, std430) readonly buffer Acc { ACC_T data[]; } acc[PLANES];
layout(set = 0, binding = 1, std430) writeonly buffer Dst { STORE_T data[]; } dst[PLANES];
layout(set = 0, binding = 2, std430) readonly buffer SrcWin { STORE_T data[]; } srcWin[PLANES];

layout(push_constant) uniform Push {
    uint dstStride;
    uint frame;
} pc;

#define SPAN (IMAGE_TILE + BLOCK - 1)

shared float blockWeights[SPAN][SPAN];
shared float rowSums[SPAN][IMAGE_TILE];

float accValue(uint plane, uint i) {
#if FLOAT_ATOMICS
    return acc[plane].data[i];
#else
    return uintBitsToFloat(acc[plane].data[i]);
#endif
}

float estimate(uint plane, ivec2 p) {
    if ((uint(ACTIVE_MASK) >> plane & 1u) == 0u)
        return p.x < WIDTH && p.y < HEIGHT ? float(srcWin[plane].data[slotOf(pc.frame) * PLANE_SIZE + uint(p.y) * WIDTH + uint(p.x)]) : 0.0;
    const ivec2 origin = ivec2(gl_WorkGroupID.xy) * IMAGE_TILE - (BLOCK - 1);
    for (uint i = gl_LocalInvocationIndex; i < SPAN * SPAN; i += IMAGE_TILE * IMAGE_TILE) {
        const ivec2 q = origin + ivec2(i % SPAN, i / SPAN);
        const bool inside = all(greaterThanEqual(q, ivec2(0))) && q.x < WIDTH && q.y < HEIGHT;
        blockWeights[i / SPAN][i % SPAN] = inside ? accValue(plane, accDen(pc.frame, q)) : 0.0;
    }
    barrier();
    for (uint r = gl_LocalInvocationID.y; r < SPAN; r += IMAGE_TILE) {
        float sum = 0.0;
        for (uint c = 0; c < BLOCK; ++c)
            sum += blockWeights[r][gl_LocalInvocationID.x + c];
        rowSums[r][gl_LocalInvocationID.x] = sum;
    }
    barrier();
    float den = 0.0;
    for (uint r = 0; r < BLOCK; ++r)
        den += rowSums[gl_LocalInvocationID.y + r][gl_LocalInvocationID.x];
    barrier();
    const bool inside = p.x < WIDTH && p.y < HEIGHT;
    return inside ? accValue(plane, accNum(pc.frame, p)) / den : 0.0;
}

void main() {
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    const uint d = uint(p.y) * pc.dstStride + uint(p.x);
#if PLANES == 3
    const float y = estimate(0, p);
    const float u = estimate(1, p);
    const float v = estimate(2, p);
    if (p.x < WIDTH && p.y < HEIGHT) {
        dst[0].data[d] = STORE_T(y + u + v * (2.0 / 3.0));
        dst[1].data[d] = STORE_T(y - v * (4.0 / 3.0));
        dst[2].data[d] = STORE_T(y - u + v * (2.0 / 3.0));
    }
#else
    const float y = estimate(0, p);
    if (p.x < WIDTH && p.y < HEIGHT)
        dst[0].data[d] = STORE_T(y);
#endif
}
)glsl";

// Block matching: one subgroup per reference block of the batch [refBase, refBase + refCount). Writes the group of each reference block as members (see
// MATCH_T), the reference block first, followed by the matches in the order the CPU implementation stacks them.
inline constexpr std::string_view blockMatch = R"glsl(
layout(local_size_x = SG * NSG) in;

layout(set = 0, binding = 0, std430) readonly buffer Win { STORE_T data[]; } win;
layout(set = 0, binding = 1, std430) writeonly buffer Matches { MATCH_T entries[]; } matches;
layout(set = 0, binding = 2, std430) writeonly buffer Counts { uint counts[]; } counts;

layout(push_constant) uniform Push {
    uint firstFrame;
    uint numFrames;
    uint cur;
    uint refBase;
    uint refCount;
} pc;

uint sg;
uint lane;

// The search, which NO_MATCHING leaves out with its shared memory.
#if !NO_MATCHING
// Candidates are laid out in items of ny vertically adjacent search positions of nx adjacent columns (NY_FULL x NX_FULL for the full search, NY_PS x 1 for
// predictive searches), so that one lane computes the distances of an item from shared loads of the window rows. Candidate s is row s % ny of item s / ny (nx
// is 1 unless SINGLE_CHUNK). Every window of a search takes the items of the largest window the search can have (FULL_COLS x FULL_ROWS or PS_COLS x PS_ROWS),
// so that positions follow from divisions by constants; the columns and rows a window clipped at the frame border lacks are rejected.
#define FULL_ITEMS ((FULL_ROWS + NY_FULL - 1) / NY_FULL * ((FULL_COLS + NX_FULL - 1) / NX_FULL))
#define PS_ITEMS ((PS_ROWS + NY_PS - 1) / NY_PS * PS_COLS)
#define FULL_SLOTS (FULL_ROWS * FULL_PITCH)
#define PS_SLOTS (PS_ROWS * PS_COLS)

// When every search fits one chunk (SINGLE_CHUNK), the padding rows and columns of the last items are spared. A window stores candidate (w, row, col) row-major
// at slot (w * rows + row) * cols + col, unless its width puts the slots of a lane (lane, lane + SG, ...) at diagonal neighbours (FULL_ROW_MAJOR,
// PS_ROW_MAJOR), whose correlated distances would keep winning the selection rounds. Such a window keeps the item order, the items of its last item row holding
// just the rows left: candidate (w, row0 + k, col) takes slot (w * rows + row0) * cols + col * n + k, n being the rows of its item. A spatial search keyed by
// slot (SLOT_KEYS) instead pads its rows to FULL_PITCH slots, the padding INF, so that its slots are row-major and lie apart for a lane: they then ascend with
// positions and serve as keys (see selectSlots).
#if SINGLE_CHUNK
#define SEARCH_SLOTS(full) ((full) ? FULL_SLOTS : PS_SLOTS)
#define ITEM_END(base, count, ny, slots, items) ((slots) == (items) ? (count) : (count) / (slots) * (items))
#define CAND_SLOT(rowMajor, item, k, ny, w, row0, col, maxRows, pitch, base) \
    ((rowMajor) != 0 ? ((w) * (maxRows) + (row0) + (k)) * (pitch) + (col) \
                : ((w) * (maxRows) + (row0)) * (pitch) + (col) * min(uint(ny), (maxRows) - (row0)) + (k))
#define CAND_STORED(row0, k, maxRows, col, wholeCols, cols) ((row0) + (k) < (maxRows) && ((wholeCols) || (col) < (cols)))
#else
#define SEARCH_SLOTS(full) ((full) ? FULL_ITEMS * NY_FULL : PS_ITEMS * NY_PS)
#define ITEM_END(base, count, ny, slots, items) (((base) + (count)) / (ny))
#define CAND_SLOT(rowMajor, item, k, ny, w, row0, col, maxRows, pitch, base) ((item) * (ny) + (k) - (base))
#define CAND_STORED(row0, k, maxRows, col, wholeCols, cols) true
#endif

// The reference block is read in quads of columns, REF4(y, q) holding columns 4q to 4q + 3 of row y (zero past BLOCK): from registers, or from shared memory
// where one load serves four columns. In correlation form (SSD_CORRELATION) it holds -2r', r' = r - refMean, so that |r - f|^2 = refEnergy + sum(f'^2 - 2r'f'),
// f' = f - refMean: one FMA per sample and candidate. The mean offset keeps the terms small where blocks are flat. Distances are single precision even for half
// precision input: f16vec2 distances were slower and lost 1-7 dB against the CPU (measured).
#define REF_QUADS ((BLOCK + 3) / 4)
float refMean;
float refEnergy;
#if SSD_CORRELATION
#define REF_SAMPLE(r) fma(r, -2.0, refMean * 2.0)
#define WIN_SAMPLE(f) ((f) - refMean)
#define SSD_TERM(r, f, d) fma(r, f, d)
#else
#define REF_SAMPLE(r) (r)
#define WIN_SAMPLE(f) (f)
#define SSD_TERM(r, f, d) fma((r) - (f), (r) - (f), d)
#endif
#if REF_IN_REGS
float refReg[BLOCK * BLOCK];
#define REF(y, x) ((x) < BLOCK ? refReg[(y) * BLOCK + min(x, BLOCK - 1)] : 0.0)
#define REF4(y, q) vec4(REF(y, (q) * 4), REF(y, (q) * 4 + 1), REF(y, (q) * 4 + 2), REF(y, (q) * 4 + 3))
#else
shared vec4 refBlock[NSG][BLOCK * REF_QUADS];
#define REF4(y, q) refBlock[sg][(y) * REF_QUADS + (q)]
#endif
// Larger blocks would take too long to compile unrolled.
#if BLOCK <= 16
#define ROW_UNROLL [[unroll]]
#else
#define ROW_UNROLL
#endif
// The window samples of row `row` an item reads: all at once in a loop of constant bounds (WHOLE_ROW_LOADS), or quad by quad as the distance terms take them,
// which keeps NVIDIA's registers low. RADV unrolls neither a loop whose bounds vary with the quad nor the loops around it, leaving v[] indexed at run time. Its
// sums then start from a zero it cannot fold: from a literal 0.0 it starts a sum with a multiply, which ACO fuses into the next term instead, out of source
// order.
#if WHOLE_ROW_LOADS
#define LOAD_ROW(NX, STEP) \
    [[unroll]] for (int i = 0; i < ((NX) - 1) * (STEP) + BLOCK; ++i) \
        v[i] = WIN_SAMPLE(float(win.data[row + i]));
#define LOAD_QUAD(q, NX, STEP)
#define SUM_ZERO uintBitsToFloat(item >> 31)
#else
#define LOAD_ROW(NX, STEP)
#define LOAD_QUAD(q, NX, STEP) \
    [[unroll]] for (int i = q == 0 ? 0 : q * 4 + ((NX) - 1) * (STEP); i < min((q + 1) * 4 + ((NX) - 1) * (STEP), ((NX) - 1) * (STEP) + BLOCK); ++i) \
        v[i] = WIN_SAMPLE(float(win.data[row + i]));
#define SUM_ZERO 0.0
#endif

// The search windows of the current search as left, top, columns and rows.
shared ivec4 searchWin[NSG][MAX_WINDOWS];
// Candidate distances as bits.
shared uint candD[NSG][CHUNK];
#if !SLOT_KEYS
shared float listD[NSG][GROUP];
#endif
shared uint listK[NSG][GROUP];
#if RADIUS > 0
shared float restD[NSG][GROUP];
shared uint restC[NSG][GROUP];
shared uint restP[NSG][GROUP];
shared uint restF[NSG][GROUP];
shared uint centers[NSG][PS_NUM];
shared uint curCenters[NSG][PS_NUM];
#endif
// Staging of merge results, (RADIUS > 0 ? 4 : 2) arrays of GROUP words. In a single chunk it takes the candidate slots, which are dead by then, and the
// selection writes its list directly.
#if SINGLE_CHUNK
#define STAGE(i) candD[sg][i]
#else
shared uint stage[NSG][(RADIUS > 0 ? 4 : 2) * GROUP];
#define STAGE(i) stage[sg][i]
#endif

uint frameOffset(uint windowFrame) {
    return slotOf(pc.firstFrame + windowFrame) * PLANE_SIZE;
}

void syncSubgroup() {
    subgroupMemoryBarrierShared();
    subgroupBarrier();
}

// _SearchBoundary of the CPU implementation, range already rounded down to a multiple of step.
int searchBoundary(int pos, int boundary, int range, int step) {
    const int r = min(range, abs(pos - boundary)) / step * step;
    return pos > boundary ? pos - r : pos + r;
}

// The search window around center.
ivec4 windowAround(ivec2 center, int range, int step) {
    range = range / step * step;
    const int l = searchBoundary(center.x, 0, range, step);
    const int r = searchBoundary(center.x, WIDTH - BLOCK, range, step);
    const int t = searchBoundary(center.y, 0, range, step);
    const int b = searchBoundary(center.y, HEIGHT - BLOCK, range, step);
    return ivec4(l, t, (r - l) / step + 1, (b - t) / step + 1);
}

// Window w, position of the first candidate and its row and column within the window of an item of a search laid out in items of ny rows and nx columns, cols
// columns and `items` items per window; false for the columns and rows a clipped window lacks.
bool itemPos(uint item, uint ny, uint nx, int step, uint cols, uint items, out uint w, out ivec2 pos, out uint row0, out uint col) {
    w = item / items;
    const uint local = item - w * items;
    const uint itemCols = (cols + nx - 1) / nx;
    col = local % itemCols * nx;
    row0 = local / itemCols * ny;
    const ivec4 window = searchWin[sg][w];
    pos = window.xy + ivec2(col, row0) * step;
    return col < uint(window.z) && row0 < uint(window.w);
}

bool inWindow(uint w, ivec2 pos, int step) {
    const ivec4 window = searchWin[sg][w];
    const ivec2 d = pos - window.xy;
    return all(greaterThanEqual(d, ivec2(0))) && d.x % step == 0 && d.y % step == 0 && uint(d.x / step) < uint(window.z) && uint(d.y / step) < uint(window.w);
}

// A predictive search over the bounding box of its windows (see predictiveSearch) holds the windows in records 1..psCenters; where they do not fill the box
// (psNotRect), positions outside all of them are rejected. Bit k tells whether row k of the item of ny rows at pos lies in one of them, the box lying on the
// position grid of every window.
#if PS_UNION
uint psCenters;
bool psNotRect;

uint centerRows(ivec2 pos, int ny) {
    if (!psNotRect)
        return ~0u;
    uint rows = 0;
    [[unroll]] for (uint c = 1; c <= PS_NUM && rows != (1u << ny) - 1u; ++c) {
        const ivec4 window = searchWin[sg][c];
        const ivec2 d = (pos - window.xy) / PS_STEP;
        if (c <= psCenters && uint(d.x) < uint(window.z))
            rows |= (1u << clamp(window.w - d.y, 0, ny)) - (1u << clamp(-d.y, 0, ny));
    }
    return rows;
}
#else
#define centerRows(pos, ny) ~0u
#endif

// Adds window row `row`, r rows below the item's first candidate, to the distance terms d of the item's ROWS x NX candidates. Each candidate sums its terms by
// rows and then columns, and in correlation form adds the energy of its own columns, whatever the item's shape.
#define ADD_ROW(ROWS, NX, STEP) \
    float v[((NX) - 1) * (STEP) + BLOCK]; \
    float energy[NX]; \
    [[unroll]] for (int x = 0; x < (NX); ++x) \
        energy[x] = SUM_ZERO; \
    LOAD_ROW(NX, STEP) \
    [[unroll]] for (int q = 0; q < REF_QUADS; ++q) { \
        LOAD_QUAD(q, NX, STEP) \
        [[unroll]] for (int x = 0; x < (NX); ++x) \
            [[unroll]] for (int c = 0; c < 4; ++c) \
                if (q * 4 + c < BLOCK) \
                    energy[x] = fma(v[x * (STEP) + q * 4 + c], v[x * (STEP) + q * 4 + c], energy[x]); \
        [[unroll]] for (int k = 0; k < ROWS; ++k) { \
            const int y = r - k * (STEP); \
            if (y >= 0 && y < BLOCK) { \
                const vec4 e = REF4(y, q); \
                [[unroll]] for (int x = 0; x < (NX); ++x) \
                    [[unroll]] for (int c = 0; c < 4; ++c) \
                        if (q * 4 + c < BLOCK) \
                            d[k][x] = SSD_TERM(e[c], v[x * (STEP) + q * 4 + c], d[k][x]); \
            } \
        } \
    } \
    if (SSD_CORRELATION != 0) \
        [[unroll]] for (int k = 0; k < ROWS; ++k) \
            if (r - k * (STEP) >= 0 && r - k * (STEP) < BLOCK) \
                [[unroll]] for (int x = 0; x < (NX); ++x) \
                    d[k][x] += energy[x];

// Distances of candidates [base, base + count) into candD, INF for those rejected: outside a clipped window, past the threshold, at the excluded reference
// position of a full search, or already covered by an earlier window of a predictive search. A lane handles whole items of ROWS x NX candidates, loading each
// window row once for every candidate it overlaps. At a clipped right edge, an item's columns past the window read up to (NX - 1) * STEP samples past the row
// (the window buffers are padded for the last one), which only the rejected candidates there take. The threshold tests the sum before the clamp of the
// correlation form, so that NaN sums of non-finite samples fail it rather than clamp to 0, as do -INF sums of finite samples whose products overflow. Kept
// distances are capped at FLT_MAX, below the INF of rejected ones: a TH_SSE overflowing to INF keeps infinite distances, as the CPU implementation does.
#define DEFINE_DISTANCES(NAME, ROWS, NX, STEP, COLS, PITCH, ITEMS, MAX_ROWS, ROW_MAJOR, FULL) \
void NAME(uint base, uint count, uint offset, ivec2 refPos) { \
    for (uint item = base / ROWS + lane; item < ITEM_END(base, count, ROWS, (MAX_ROWS) * (PITCH), ITEMS); item += SG) { \
        uint w, row0, col; \
        ivec2 pos; \
        const bool inside = itemPos(item, ROWS, NX, STEP, COLS, ITEMS, w, pos, row0, col); \
        float d[ROWS][NX]; \
        [[unroll]] for (int k = 0; k < ROWS; ++k) \
            [[unroll]] for (int x = 0; x < (NX); ++x) \
                d[k][x] = SUM_ZERO; \
        if (inside) { \
            ROW_UNROLL for (int r = 0; r < (ROWS - 1) * (STEP) + BLOCK; ++r) { \
                const uint row = offset + uint(min(pos.y + r, HEIGHT - 1)) * WIDTH + uint(pos.x); \
                ADD_ROW(ROWS, NX, STEP) \
            } \
        } \
        const uint rows = uint(searchWin[sg][w].w); \
        const uint cols = uint(searchWin[sg][w].z); \
        const uint kept = FULL ? ~0u : centerRows(pos, ROWS); \
        [[unroll]] for (int k = 0; k < ROWS; ++k) \
            [[unroll]] for (int x = 0; x < (NX); ++x) { \
                const ivec2 p = pos + ivec2(x, k) * (STEP); \
                const float sum = SSD_CORRELATION != 0 ? refEnergy + d[k][x] : d[k][x]; \
                const float dist = min(SSD_CORRELATION != 0 ? max(sum, 0.0) : sum, FLT_MAX); \
                bool keep = inside && row0 + k < rows && (x == 0 || col + x < cols) && ((kept >> k) & 1u) != 0u && sum <= TH_SSE && !(FULL && p == refPos) && \
                            (SSD_CORRELATION == 0 || sum > -INF); \
                for (uint e = 0; e < w && keep; ++e) \
                    keep = !inWindow(e, p, STEP); \
                if (CAND_STORED(row0, k, MAX_ROWS, col + x, (COLS) % (NX) == 0, COLS)) \
                    candD[sg][CAND_SLOT(ROW_MAJOR, item, k, ROWS, w, row0, col + x, MAX_ROWS, PITCH, base)] = floatBitsToUint(keep ? dist : INF); \
            } \
    } \
}

DEFINE_DISTANCES(fullDistances, NY_FULL, NX_FULL, BM_STEP, FULL_COLS, FULL_PITCH, FULL_ITEMS, FULL_ROWS, FULL_ROW_MAJOR, true)
#if RADIUS > 0
DEFINE_DISTANCES(predictiveDistances, NY_PS, 1, PS_STEP, PS_COLS, PS_COLS, PS_ITEMS, PS_ROWS, PS_ROW_MAJOR, false)
#endif

bool keyLess(float da, uint ka, float db, uint kb) {
    return da < db || (da == db && ka < kb);
}

uint candidateKey(uint i, uint na, uint base, bool full) {
    if (i < na)
        return listK[sg][i];
    const uint s = base + i - na;
    const int step = full ? BM_STEP : PS_STEP;
#if SINGLE_CHUNK
    const uint slots = full ? FULL_SLOTS : PS_SLOTS;
    const uint pitch = full ? FULL_PITCH : PS_COLS;
    const uint rows = full ? FULL_ROWS : PS_ROWS;
    const uint ny = full ? NY_FULL : NY_PS;
    const uint w = full ? 0 : s / slots;
    const uint r = s - w * slots;
    uint row, col;
    if (full ? FULL_ROW_MAJOR != 0 : PS_ROW_MAJOR != 0) {
        row = r / pitch;
        col = r % pitch;
    } else {
        const uint row0 = r / (pitch * ny) * ny;
        const uint itemRows = min(ny, rows - row0);
        const uint rem = r - row0 * pitch;
        col = itemRows == ny ? rem / ny : rem / max(rows % ny, 1);
        row = row0 + rem - col * itemRows;
    }
    return packPos(searchWin[sg][w].xy + ivec2(col, row) * step);
#else
    const uint ny = full ? NY_FULL : NY_PS;
    uint w, row0, col;
    ivec2 pos;
    itemPos(s / ny, ny, 1, step, full ? FULL_COLS : PS_COLS, full ? FULL_ITEMS : PS_ITEMS, w, pos, row0, col);
    return packPos(pos + ivec2(0, int(s % ny) * step));
#endif
}

#if !SLOT_KEYS
// A list entry or candidate distance of mergeCandidates.
float elementD(uint i, uint na) {
    return i < na ? listD[sg][i] : uintBitsToFloat(candD[sg][i - na]);
}
#endif

// The lane holding the smallest (distance, key) of the subgroup, its distance in minD. Keys are compared only when distances tie, which is rare enough that the
// common case costs one reduction.
uint argMin(float d, uint key, out float minD) {
    minD = subgroupMin(d);
    uvec4 ballot = subgroupBallot(d == minD);
    if (subgroupBallotBitCount(ballot) > 1) {
        const uint minK = subgroupMin(d == minD ? key : 0xFFFFFFFFu);
        ballot = subgroupBallot(d == minD && key == minK);
    }
    return subgroupBallotFindLSB(ballot);
}

// Bitonic compare-exchange of (d, i) with the pair of lane ^ mask: the smaller stays where keepMin, else the larger.
void exchange(inout float d, inout uint i, uint mask, bool keepMin) {
    const float otherD = subgroupShuffleXor(d, mask);
    const uint otherI = subgroupShuffleXor(i, mask);
    if (keyLess(otherD, otherI, d, i) == keepMin) {
        d = otherD;
        i = otherI;
    }
}

// Sorts the two cached pairs of every lane, (d0, i0) not after (d1, i1), by a bitonic network: lane l ends up with ranks 2l (lo) and 2l + 1 (hi).
void sortPairs(float d0, uint i0, float d1, uint i1, out float lo, out uint loI, out float hi, out uint hiI) {
    // Odd lanes start descending, so that the network starts at sequences of four.
    const bool odd = (lane & 1u) != 0u;
    lo = odd ? d1 : d0;
    hi = odd ? d0 : d1;
    loI = odd ? i1 : i0;
    hiI = odd ? i0 : i1;
    [[unroll]] for (uint s = 4; s <= SG * 2; s *= 2) {
        const bool up = (lane & (s / 2)) == 0u;
        [[unroll]] for (uint j = s / 4; j > 0; j /= 2) {
            const bool keepMin = ((lane & j) == 0u) == up;
            exchange(lo, loI, j, keepMin);
            exchange(hi, hiI, j, keepMin);
        }
        if (keyLess(hi, hiI, lo, loI) == up) {
            const float d = lo;
            const uint i = loI;
            lo = hi;
            loI = hiI;
            hi = d;
            hiI = i;
        }
    }
}

// Inserts (eD, eI) into the ranks sortPairs left, the largest dropping out.
void insertPair(float eD, uint eI, inout float lo, inout uint loI, inout float hi, inout uint hiI) {
    const uint prevLane = max(lane, 1u) - 1u;
    const float prevD = subgroupShuffle(hi, prevLane);
    const uint prevI = subgroupShuffle(hiI, prevLane);
    const bool loLess = keyLess(lo, loI, eD, eI);
    const bool prevLess = lane == 0u || keyLess(prevD, prevI, eD, eI);
    if (!keyLess(hi, hiI, eD, eI)) {
        hi = loLess ? eD : lo;
        hiI = loLess ? eI : loI;
    }
    if (!loLess) {
        lo = prevLess ? eD : prevD;
        loI = prevLess ? eI : prevI;
    }
}

// The pair of rank k - 1 of the ranks sortPairs left.
void rankedPair(uint k, float lo, uint loI, float hi, uint hiI, out float d, out uint i) {
    const bool high = (k & 1u) == 0u;
    d = subgroupShuffle(high ? hi : lo, (k - 1) / 2);
    i = subgroupShuffle(high ? hiI : loI, (k - 1) / 2);
}

// The two smallest (distance, slot) pairs of the lane's slots of candD[0, count) (lane, lane + SG, ...), the earlier slot first among equal distances, and the
// third smallest distance. Skipping INF distances changes no result but the registers the compiler allocates: callers pass a constant skipInf, true for
// selectCached (an unconditional min costs it up to 40 registers), false for selectSlots (measured faster).
void lowestTwo(uint count, bool skipInf, out float d0, out uint i0, out float d1, out uint i1, out float d2) {
    d0 = INF;
    d1 = INF;
    d2 = INF;
    i0 = 0xFFFFFFFFu;
    i1 = 0xFFFFFFFFu;
    for (uint i = lane; i < count; i += SG) {
        const float d = uintBitsToFloat(candD[sg][i]);
        if (skipInf && d == INF)
            continue;
        if (d < d0) {
            d2 = d1;
            d1 = d0;
            i1 = i0;
            d0 = d;
            i0 = i;
        } else if (d < d1) {
            d2 = d1;
            d1 = d;
            i1 = i;
        } else if (!skipInf) {
            d2 = min(d2, d);
        } else if (d < d2) {
            d2 = d;
        }
    }
}

#if RADIUS > 0
// Whether rest entries precede a predictive match of equal distance: ties go to the earlier frame (see mergeRest), so they do when searching forward.
bool restFirst;

// Whether a predictive search stops before its match of rank n, of distance minD: the match ranks n + #{rest entries preceding it} in the rest set, both
// growing with n; the first PS_NUM matches are still needed as the next frame's search centers.
bool predictiveCut(bool full, uint n, uint nr, float minD) {
    if (full || n < uint(PS_NUM))
        return false;
    if (n >= GROUP - 1)
        return true;
    const uint bound = GROUP - 2 - n;
    return bound < nr && (restFirst ? restD[sg][bound] <= minD : restD[sg][bound] < minD);
}
#endif

#if SLOT_KEYS
// mergeCandidates of a spatial search, whose single chunk is keyed by slot; the matches go to listK as positions. Each lane scans its elements (lane, lane +
// SG, ...) for its two smallest (distance, slot) pairs and its third smallest distance. Where k <= 2 * SG, a bitonic network sorts the cached pairs, lane l
// holding ranks 2l and 2l + 1. No match exceeds the k-th of them, T, so beyond those only a lane whose third distance is at most T's can hold matches: its
// elements past its cached ones up to T are inserted one by one, a few per search, the largest pair dropping out. Otherwise rounds select ascending (distance,
// slot): as the elements a lane has left are those past the last one selected, the rounds write no slot and need no synchronization.
uint selectSlots(uint k) {
    float myD, nextD, thirdD;
    uint myI, nextI;
    lowestTwo(FULL_SLOTS, false, myD, myI, nextD, nextI, thirdD);

#if GROUP - 1 <= SG * 2
    float lo, hi;
    uint loI, hiI;
    sortPairs(myD, myI, nextD, nextI, lo, loI, hi, hiI);
    float boundD;
    uint boundI;
    rankedPair(k, lo, loI, hi, hiI, boundD, boundI);
    // A third element tied with T follows it if the second element ties too and does not precede T.
    bool pending = thirdD < boundD || (thirdD == boundD && thirdD != INF && (nextD < thirdD || nextI < boundI));
    for (uvec4 owners = subgroupBallot(pending); owners != uvec4(0); owners = subgroupBallot(pending)) {
        const uint owner = subgroupBallotFindLSB(owners);
        pending = pending && lane != owner;
        const float ownerD = subgroupShuffle(nextD, owner);
        const uint ownerI = subgroupShuffle(nextI, owner);
        for (uint base = owner; base < FULL_SLOTS; base += SG * SG) {
            const uint i = base + lane * SG;
            const float d = i < FULL_SLOTS ? uintBitsToFloat(candD[sg][i]) : INF;
            bool found = d != INF && keyLess(ownerD, ownerI, d, i) && !keyLess(boundD, boundI, d, i);
            for (uvec4 finds = subgroupBallot(found); finds != uvec4(0); finds = subgroupBallot(found)) {
                const uint from = subgroupBallotFindLSB(finds);
                found = found && lane != from;
                insertPair(subgroupShuffle(d, from), base + from * SG, lo, loI, hi, hiI);
            }
        }
    }

    const uint n = min(k, subgroupBallotBitCount(subgroupBallot(lo != INF)) + subgroupBallotBitCount(subgroupBallot(hi != INF)));
    if (lane * 2 < n)
        listK[sg][lane * 2] = candidateKey(loI, 0, 0, true);
    if (lane * 2 + 1 < n)
        listK[sg][lane * 2 + 1] = candidateKey(hiI, 0, 0, true);
#else
    bool nextKnown = true;
    uint n = 0;
    for (; n < k; ++n) {
        float minD;
        const uint winner = argMin(myD, myI, minD);
        if (minD == INF)
            break;
        const bool promote = subgroupBallotBitExtract(subgroupBallot(nextKnown), winner);
        if (lane == winner) {
            listK[sg][n] = candidateKey(myI, 0, 0, true);
            if (promote) {
                myD = nextD;
                myI = nextI;
                nextKnown = false;
            }
        }
        if (promote)
            continue;

        const uint lastI = subgroupShuffle(myI, winner);
        float d = INF;
        uint index = 0xFFFFFFFFu;
        for (uint i = winner + lane * SG; i < FULL_SLOTS; i += SG * SG) {
            const float e = uintBitsToFloat(candD[sg][i]);
            if (e < d && (e > minD || (e == minD && i > lastI))) {
                d = e;
                index = i;
            }
        }
        const float refillD = subgroupMin(d);
        const uint refillI = subgroupMin(d == refillD ? index : 0xFFFFFFFFu);
        if (lane == winner) {
            myD = refillD;
            myI = refillI;
        }
    }
#endif
    syncSubgroup();
    return n;
}
#else
#if SINGLE_CHUNK && GROUP * 2 <= SG
// The rounds of mergeCandidates for a single chunk (na == 0) after a scan without keys: each lane caches its two smallest distances, keyed after the scan, and
// its third smallest distance, so that the rounds need no refills and leave candD as it is. A cached element tied with the third distance may follow an
// uncached one of that distance: such a head takes key 0, as does the third, and once a round selects one of them, false is returned for mergeCandidates to run
// as usual.
bool selectCached(uint nc, uint k, bool full, uint nr, out uint n) {
    float d0, d1, d2;
    uint i0, i1;
    lowestTwo(nc, true, d0, i0, d1, i1, d2);
    uint key0 = d0 != INF ? candidateKey(i0, 0, 0, full) : 0u;
    uint key1 = d1 != INF ? candidateKey(i1, 0, 0, full) : 0u;
    if (d0 == d1 && key1 < key0) {
        const uint smaller = key1;
        key1 = key0;
        key0 = smaller;
    }

#if GROUP - 1 >= 7
    // A full search sorts the cached pairs as selectSlots does, which pays from k = 7 on (measured), the network costing about as much as 5-6 rounds. Only a
    // lane whose third distance is at most T's can hold further matches: its elements other than the cached ones up to T, keyed only for such lanes, are
    // inserted.
    if (full) {
        float lo, hi;
        uint loK, hiK;
        sortPairs(d0, key0, d1, key1, lo, loK, hi, hiK);
        float boundD;
        uint boundK;
        rankedPair(k, lo, loK, hi, hiK, boundD, boundK);
        bool pending = d2 <= boundD && d2 != INF;
        for (uvec4 owners = subgroupBallot(pending); owners != uvec4(0); owners = subgroupBallot(pending)) {
            const uint owner = subgroupBallotFindLSB(owners);
            pending = pending && lane != owner;
            const uint cached0 = subgroupShuffle(i0, owner);
            const uint cached1 = subgroupShuffle(i1, owner);
            for (uint base = owner; base < nc; base += SG * SG) {
                const uint i = base + lane * SG;
                const float d = i < nc ? uintBitsToFloat(candD[sg][i]) : INF;
                const uint key = candidateKey(i, 0, 0, true);
                bool found = d <= boundD && d != INF && i != cached0 && i != cached1 && (d < boundD || key < boundK);
                for (uvec4 finds = subgroupBallot(found); finds != uvec4(0); finds = subgroupBallot(found)) {
                    const uint from = subgroupBallotFindLSB(finds);
                    found = found && lane != from;
                    insertPair(subgroupShuffle(d, from), subgroupShuffle(key, from), lo, loK, hi, hiK);
                }
            }
        }

        n = min(k, subgroupBallotBitCount(subgroupBallot(lo != INF)) + subgroupBallotBitCount(subgroupBallot(hi != INF)));
        if (lane * 2 < n) {
            listD[sg][lane * 2] = lo;
            listK[sg][lane * 2] = loK;
        }
        if (lane * 2 + 1 < n) {
            listD[sg][lane * 2 + 1] = hi;
            listK[sg][lane * 2 + 1] = hiK;
        }
        syncSubgroup();
        return true;
    }
#endif

    float head = d0;
    uint headKey = d0 == d2 ? 0u : key0;
    uint pops = 0;
    for (n = 0; n < k; ++n) {
        float minD;
        const uint winner = argMin(head, headKey, minD);
        if (minD == INF)
            break;
#if RADIUS > 0
        if (predictiveCut(full, n, nr, minD))
            break;
#endif
        if (lane == winner) {
            listD[sg][n] = minD;
            listK[sg][n] = headKey;
            ++pops;
            head = pops == 1 ? d1 : pops == 2 ? d2 : INF;
            headKey = pops == 1 && d1 != d2 ? key1 : 0u;
        }
    }
    syncSubgroup();
    if (subgroupBallotBitCount(subgroupBallot(pops > 2 || (pops > 0 && (pops == 1 ? d0 : d1) == d2))) != 0)
        return false;
    return true;
}
#endif

// Selects the k smallest (distance, position) pairs of listD/K[0, na) and the candidates [base, base + nc) held in candD[0, nc) into listD/K, ascending.
// Distances of INF are never selected. Each lane caches the two smallest elements it owns (lane, lane + SG, ...); a lane that wins a round promotes its
// runner-up, and only once that is used up does the whole subgroup rescan the lane's elements to refill its minimum. A predictive search stops early once its
// next match could no longer enter the nr entries of the rest set (see mergeRest). A single chunk tries selectCached first.
uint mergeCandidates(uint na, uint nc, uint k, uint base, bool full, uint nr) {
#if SINGLE_CHUNK && GROUP * 2 <= SG
    uint selected;
    if (selectCached(nc, k, full, nr, selected))
        return selected;
#endif
    const uint total = na + nc;
    float myD = INF, nextD = INF;
    uint myK = 0xFFFFFFFFu, nextK = 0xFFFFFFFFu;
    uint myI = 0xFFFFFFFFu, nextI = 0xFFFFFFFFu;
    bool nextKnown = true;
    for (uint i = lane; i < total; i += SG) {
        const float d = elementD(i, na);
        if (d == INF)
            continue;
        const uint key = candidateKey(i, na, base, full);
        if (keyLess(d, key, myD, myK)) {
            nextD = myD;
            nextK = myK;
            nextI = myI;
            myD = d;
            myK = key;
            myI = i;
        } else if (keyLess(d, key, nextD, nextK)) {
            nextD = d;
            nextK = key;
            nextI = i;
        }
    }

    uint n = 0;
    for (; n < k; ++n) {
        float minD;
        const uint winner = argMin(myD, myK, minD);
        if (minD == INF)
            break;
#if RADIUS > 0
        if (predictiveCut(full, n, nr, minD))
            break;
#endif
        const bool promote = subgroupBallotBitExtract(subgroupBallot(nextKnown), winner);
        if (lane == winner) {
#if SINGLE_CHUNK
            listD[sg][n] = minD;
            listK[sg][n] = myK;
#else
            STAGE(n) = floatBitsToUint(minD);
            STAGE(GROUP + n) = myK;
#endif
            if (myI < na)
                listD[sg][myI] = INF;
            else
                candD[sg][myI - na] = floatBitsToUint(INF);
            if (promote) {
                myD = nextD;
                myK = nextK;
                myI = nextI;
                nextKnown = false;
            }
        }
        syncSubgroup();
        if (promote)
            continue;

        float d = INF;
        uint key = 0xFFFFFFFFu;
        uint index = 0xFFFFFFFFu;
        for (uint i = winner + lane * SG; i < total; i += SG * SG) {
            const float e = elementD(i, na);
            if (e == INF)
                continue;
            const uint eKey = candidateKey(i, na, base, full);
            if (keyLess(e, eKey, d, key)) {
                d = e;
                key = eKey;
                index = i;
            }
        }
        const float refillD = subgroupMin(d);
        const uint refillK = subgroupMin(d == refillD ? key : 0xFFFFFFFFu);
        const uint refillI = subgroupMin(d == refillD && key == refillK ? index : 0xFFFFFFFFu);
        if (lane == winner) {
            myD = refillD;
            myK = refillK;
            myI = refillI;
        }
        syncSubgroup();
    }
    syncSubgroup();
#if !SINGLE_CHUNK
    for (uint i = lane; i < n; i += SG) {
        listD[sg][i] = uintBitsToFloat(STAGE(i));
        listK[sg][i] = STAGE(GROUP + i);
    }
    syncSubgroup();
#endif
    return n;
}
#endif

// Block matching over the first `windows` windows: the k best matches end up in listD/K (see mergeCandidates for nr).
uint search(uint offset, bool full, ivec2 refPos, uint k, uint windows, uint nr) {
    const uint total = windows * SEARCH_SLOTS(full);
    uint n = 0;
    for (uint base = 0; base < total; base += CHUNK) {
        const uint nc = min(CHUNK, total - base);
#if RADIUS > 0
        if (!full)
            predictiveDistances(base, nc, offset, refPos);
        else
#endif
            fullDistances(base, nc, offset, refPos);
        syncSubgroup();
#if SLOT_KEYS
        n = selectSlots(k);
#else
        n = mergeCandidates(n, nc, k, base, full, nr);
#endif
    }
    return n;
}

#if RADIUS > 0
// Predictive search in another frame over the union of the windows around the centers. Returns the number of matches.
uint predictiveSearch(uint numCenters, uint offset, uint nr) {
#if PS_UNION
    // The windows go to records 1..numCenters. Where their bounding box fits the layout, it is searched as the single window 0; otherwise they move down to
    // records 0.. and are searched one after another.
    uint windows = numCenters;
    bool notRect = false;
    if (lane == 0) {
        ivec2 lo = ivec2(0x7FFFFFFF);
        ivec2 hi = ivec2(-1);
        for (uint c = 1; c <= numCenters; ++c) {
            const ivec4 window = windowAround(unpackPos(centers[sg][c - 1]), PS_RANGE, PS_STEP);
            searchWin[sg][c] = window;
            lo = min(lo, window.xy);
            hi = max(hi, window.xy + (window.zw - 1) * PS_STEP);
        }
        bool aligned = true;
        for (uint c = 1; c <= numCenters; ++c)
            aligned = aligned && all(equal((searchWin[sg][c].xy - lo) % PS_STEP, ivec2(0)));
        const uvec2 box = uvec2((hi - lo) / PS_STEP + 1);
        if (aligned && box.x <= PS_COLS && box.y <= PS_ROWS) {
            // Two windows fill their box when the box holds no more positions than they do together.
            uint covered = box.x * box.y;
            if (numCenters == 2) {
                const ivec4 a = searchWin[sg][1];
                const ivec4 b = searchWin[sg][2];
                const ivec2 hiOverlap = min(a.xy + (a.zw - 1) * PS_STEP, b.xy + (b.zw - 1) * PS_STEP);
                const ivec2 overlap = max((hiOverlap - max(a.xy, b.xy)) / PS_STEP + 1, ivec2(0));
                covered = uint(a.z * a.w + b.z * b.w - overlap.x * overlap.y);
            }
            notRect = numCenters > 2 || covered != box.x * box.y;
            searchWin[sg][0] = ivec4(lo, box);
            windows = 1;
        } else {
            for (uint c = 0; c < numCenters; ++c)
                searchWin[sg][c] = searchWin[sg][c + 1];
        }
    }
    windows = subgroupBroadcast(windows, 0);
    psNotRect = subgroupBroadcast(notRect, 0);
    psCenters = numCenters;
#else
    const uint windows = numCenters;
    for (uint c = lane; c < numCenters; c += SG)
        searchWin[sg][c] = windowAround(unpackPos(centers[sg][c]), PS_RANGE, PS_STEP);
#endif
    syncSubgroup();
    return search(offset, false, ivec2(-1), GROUP, windows, nr);
}

// Keeps the GROUP - 1 smallest (distance, key) entries of the rest set and the frame list in listD/K, the key of a match being frame * GROUP + its rank in its
// frame's list: as a frame lists its matches by (distance, position), equal distances order by frame and position like the CPU implementation. Both sets are
// small and hold unique keys, so every element finds its rank directly instead of in selection rounds.
uint mergeRest(uint nr, uint nl, uint frame) {
    const uint total = nr + nl;
    for (uint i = lane; i < total; i += SG) {
        const float d = i < nr ? restD[sg][i] : listD[sg][i - nr];
        const uint c = i < nr ? restC[sg][i] : frame * GROUP + i - nr;
        uint rank = 0;
        for (uint j = 0; j < total; ++j)
            rank += keyLess(j < nr ? restD[sg][j] : listD[sg][j - nr], j < nr ? restC[sg][j] : frame * GROUP + j - nr, d, c) ? 1 : 0;
        if (rank < GROUP - 1) {
            STAGE(rank) = floatBitsToUint(d);
            STAGE(GROUP + rank) = c;
            STAGE(GROUP * 2 + rank) = i < nr ? restP[sg][i] : listK[sg][i - nr];
            STAGE(GROUP * 3 + rank) = i < nr ? restF[sg][i] : frame;
        }
    }
    const uint n = min(total, uint(GROUP - 1));
    syncSubgroup();
    for (uint i = lane; i < n; i += SG) {
        restD[sg][i] = uintBitsToFloat(STAGE(i));
        restC[sg][i] = STAGE(GROUP + i);
        restP[sg][i] = STAGE(GROUP * 2 + i);
        restF[sg][i] = STAGE(GROUP * 3 + i);
    }
    syncSubgroup();
    return n;
}

// Keeps the best positions of a frame as the search centers of the next frame.
void saveCenters(uint n) {
    for (uint i = lane; i < min(n, uint(PS_NUM)); i += SG)
        centers[sg][i] = listK[sg][i];
    syncSubgroup();
}
#endif
#endif

void main() {
    sg = gl_SubgroupID;
    lane = gl_SubgroupInvocationID;
#if SLOT_KEYS
    // Nothing else writes the padding slots.
    for (uint row = lane; row < FULL_ROWS; row += SG)
        for (uint col = FULL_COLS; col < FULL_PITCH; ++col)
            candD[sg][row * FULL_PITCH + col] = floatBitsToUint(INF);
#endif

    for (uint slot = gl_WorkGroupID.x * NSG + sg; slot < pc.refCount; slot += gl_NumWorkGroups.x * NSG) {
        const uint ref = pc.refBase + slot;
        const ivec2 refPos = refPosition(ref / REF_COLS, ref % REF_COLS);
        const uint out0 = slot * GROUP;
        if (lane == 0)
#if RADIUS > 0
            matches.entries[out0] = uvec2(packPos(refPos), pc.cur);
#else
            matches.entries[out0] = packPos(refPos);
#endif
#if NO_MATCHING
        if (lane == 0)
            counts.counts[slot] = 1;
#else
        const uint curOffset = frameOffset(pc.cur);
        const uint refOffset = curOffset + uint(refPos.y) * WIDTH + uint(refPos.x);
#if SSD_CORRELATION
        float sum = 0.0;
        for (uint i = lane; i < BLOCK * BLOCK; i += SG)
            sum += float(win.data[refOffset + i / BLOCK * WIDTH + i % BLOCK]);
        refMean = subgroupAdd(sum) / (BLOCK * BLOCK);
        float energy = 0.0;
        for (uint i = lane; i < BLOCK * BLOCK; i += SG) {
            const float v = float(win.data[refOffset + i / BLOCK * WIDTH + i % BLOCK]) - refMean;
            energy = fma(v, v, energy);
        }
        refEnergy = subgroupAdd(energy);
#endif
#if REF_IN_REGS
        [[unroll]] for (uint i = 0; i < BLOCK * BLOCK; ++i)
            refReg[i] = REF_SAMPLE(float(win.data[refOffset + i / BLOCK * WIDTH + i % BLOCK]));
#else
        for (uint i = lane; i < BLOCK * REF_QUADS * 4; i += SG) {
            const uint y = i / (REF_QUADS * 4);
            const uint x = i % (REF_QUADS * 4);
            refBlock[sg][y * REF_QUADS + x / 4][x % 4] = x < BLOCK ? REF_SAMPLE(float(win.data[refOffset + y * WIDTH + x])) : 0.0;
        }
#endif
        if (lane == 0)
            searchWin[sg][0] = windowAround(refPos, BM_RANGE, BM_STEP);
        syncSubgroup();

        const uint n = search(curOffset, true, refPos, GROUP - 1, 1, 0);
#if RADIUS == 0
        for (uint i = lane; i < n; i += SG)
            matches.entries[out0 + 1 + i] = listK[sg][i];
        if (lane == 0)
            counts.counts[slot] = n + 1;
#else
        // The rest set starts with the current frame's matches, keyed as in mergeRest.
        for (uint i = lane; i < n; i += SG) {
            restD[sg][i] = listD[sg][i];
            restC[sg][i] = pc.cur * GROUP + i;
            restP[sg][i] = listK[sg][i];
            restF[sg][i] = pc.cur;
        }
        if (lane == 0)
            curCenters[sg][0] = packPos(refPos);
        for (uint i = lane; i < min(n, uint(PS_NUM - 1)); i += SG)
            curCenters[sg][i + 1] = listK[sg][i];
        syncSubgroup();
        const uint numCurCenters = min(n + 1, uint(PS_NUM));
        uint nr = n;

        for (int dir = -1; dir <= 1; dir += 2) {
            for (uint i = lane; i < numCurCenters; i += SG)
                centers[sg][i] = curCenters[sg][i];
            syncSubgroup();
            restFirst = dir > 0;
            uint numCenters = numCurCenters;
            for (int f = int(pc.cur) + dir; f >= 0 && f < int(pc.numFrames); f += dir) {
                const uint nl = predictiveSearch(numCenters, frameOffset(uint(f)), nr);
                // A frame without matches keeps the centers of the last one that had some, as the CPU implementation does.
                saveCenters(nl);
                if (nl > 0)
                    numCenters = min(nl, uint(PS_NUM));
                nr = mergeRest(nr, nl, uint(f));
            }
        }

        // Like the CPU implementation, the members follow the reference block sorted by distance, as the rest set holds them.
        for (uint i = lane; i < nr; i += SG)
            matches.entries[out0 + 1 + i] = uvec2(restP[sg][i], restF[sg][i]);
        if (lane == 0)
            counts.counts[slot] = nr + 1;
#endif
        syncSubgroup();
#endif
    }
}
)glsl";

// Collaborative filtering: workgroups stride over the tiles of TILE_REFS x TILE_REFS reference blocks of the batch, filtering the group of each block in one
// plane. The group is held as T pairs of vertically adjacent samples, element (z, p, x) covering rows 2p and 2p + 1 of column x of block z, so the row and
// column transforms run on two lines at once (the codelet z-pass too, unless Z_SCALAR). The z-pass function zLine, specialized for the group sizes in use, is
// spliced in between the two halves.
inline constexpr std::string_view filterHead = R"glsl(
layout(local_size_x = WG) in;

layout(set = 0, binding = 0, std430) readonly buffer SrcWin { STORE_T data[]; } srcWin[PLANES];
#if WIENER
layout(set = 0, binding = 1, std430) readonly buffer RefWin { STORE_T data[]; } refWin[PLANES];
#endif
layout(set = 0, binding = 2, std430) readonly buffer Matches { MATCH_T entries[]; } matches;
layout(set = 0, binding = 3, std430) readonly buffer Counts { uint counts[]; } counts;
layout(set = 0, binding = 4, std430) buffer Acc { ACC_T data[]; } acc[PLANES];
#if GLOBAL_GROUP
layout(set = 0, binding = 5, std430) coherent buffer Scratch { T data[]; } scratch;
#endif
#if PINGPONG
layout(set = 0, binding = 6, std430) readonly buffer CosTable { float data[]; } cosTable;
#endif

layout(push_constant) uniform Push {
    uint firstFrame;
    uint activeLo;
    uint cur;
    uint refRow0;
    uint refRows;
} pc;

// A block takes HB rows of BLOCK_P elements T, each holding a row pair of the block (see makePlan for the padding).
#define HB ((BLOCK + 1) / 2)
#define BLK_T (HB * BLOCK_P)
#define GRP_T (GROUP * BLK_T)
#define REF_GRP (WIENER * GRP_T)
#define BUF_T (GRP_T * (WIENER + 1))
#define NUM_SG (WG / SG)

// Index of element i of block z, i counted in the unpermuted layout. Unpadded blocks of four row pairs of 8-byte elements (GROUP_SWIZZLE) stay free of bank
// conflicts by a permutation instead: block z swaps its row pairs by z & 1 and XORs the column with z & 1, the high bit of the row pair and z >> 1 & 1. Blocks
// start at multiples of 32 elements there.
uint swizzled(uint z, uint i) {
#if GROUP_SWIZZLE
    return i ^ ((z & 1u) * 12u | (i >> 3 & 2u) | (z >> 1 & 1u));
#else
    return i;
#endif
}

#if GLOBAL_GROUP
uint groupBase;
#define G(i) scratch.data[groupBase + (i)]
// The fences around barrier() release this invocation's scratch writes and acquire those of the others.
#define SYNC() memoryBarrierBuffer(); barrier(); memoryBarrierBuffer()
#else
shared T groupData[BUF_T * (PINGPONG + 1)];
#define G(i) groupData[i]
#define SYNC() barrier()
#endif

// A Wiener group in shared memory with a codelet z-pass keeps the sum of each subgroup in a reference element its first z-line no longer needs (see statElem),
// so that shared memory holds the group data alone.
#define STAT_IN_REF (WIENER != 0 && GROUP <= MAX_CODELET && GLOBAL_GROUP == 0)
#if !STAT_IN_REF
shared float partial[NUM_SG];
#endif

uint plane;
uint groupSlot;

// Position and window frame of member z of the current group.
uvec2 member(uint z) {
#if RADIUS > 0
    return matches.entries[groupSlot * GROUP + z];
#else
    return uvec2(matches.entries[groupSlot * GROUP + z], 0u);
#endif
}

// Thresholds (basic) or noise variances (Wiener) of the plane, indexed by the number of zero frequency axes.
float coef[4];

void accumulate(uint i, float v) {
#if FLOAT_ATOMICS
    atomicAdd(acc[plane].data[i], v);
#else
    uint old = acc[plane].data[i];
    for (;;) {
        const uint prev = atomicCompSwap(acc[plane].data[i], old, floatBitsToUint(uintBitsToFloat(old) + v));
        if (prev == old)
            break;
        old = prev;
    }
#endif
}

// The numerator of the reference frame around the tile of reference blocks the workgroup processes, which holds every block the full search can match:
// estimates accumulate here and reach memory in one pass per tile. Without ACC_TILE, where the tile would not fit into shared memory or would not pay off (see
// makePlan), every estimate goes to memory directly.
// The tile holds fixed point sums of 26 fraction bits: integer atomics are native in shared memory, where float ones may be emulated. Values out of range go to
// memory directly, and so does the 2^32 a wrapping addition loses. So do groups weighted below 2^16 units, which keeps the error rounding adds to an estimate
// within 2^-17.
#define TILE_SCALE 67108864.0
#if ACC_TILE
shared int tileNum[TILE_SPAN * TILE_SPAN];
ivec2 tileOrigin;
ivec2 tileSize;
bool groupInTile;

void accumulateTile(uint t, uint i, float v) {
    const float s = v * TILE_SCALE;
    if (!(abs(s) < 1073741824.0)) {
        accumulate(i, v);
        return;
    }
    const int q = int(round(s));
    const int old = atomicAdd(tileNum[t], q);
    if (((old ^ (old + q)) & (q ^ (old + q))) < 0)
        accumulate(i, q < 0 ? -4294967296.0 / TILE_SCALE : 4294967296.0 / TILE_SCALE);
}

float tileValue(uint t) {
    return float(tileNum[t]) / TILE_SCALE;
}
#endif

uint flagCount(uint z, uint y, uint x) {
    return uint(x == 0) + uint(y == 0) + uint(z == 0);
}

// Hard thresholding of the coefficient at (z, y, x), counting the retained coefficients.
S threshold(S v, uint z, uint y, uint x, inout float stat) {
    const bool keep = abs(float(v)) > coef[flagCount(z, y, x)];
    stat += keep ? 1.0 : 0.0;
    return keep ? v : S(0);
}

// The empirical Wiener weight of the reference coefficient at (z, y, x), summing the squared weights. It is computed in single precision: squares of half
// precision coefficients and small noise variances are not representable in half precision.
S wienerWeight(S r, uint z, uint y, uint x, inout float stat) {
    const float r2 = float(r) * float(r);
    const float w = r2 / (r2 + coef[flagCount(z, y, x)]);
    stat += w * w;
    return S(w);
}

// The same on T elements, rows 2p and 2p + 1; the padding row of odd blocks is zeroed and not counted.
T threshold(T v, uint z, uint p, uint x, inout float stat) {
    [[unroll]] for (uint c = 0; c < 2; ++c)
        v[c] = p * 2 + c < BLOCK ? threshold(v[c], z, p * 2 + c, x, stat) : S(0);
    return v;
}

T wienerWeight(T r, uint z, uint p, uint x, inout float stat) {
    [[unroll]] for (uint c = 0; c < 2; ++c)
        r[c] = p * 2 + c < BLOCK ? wienerWeight(r[c], z, p * 2 + c, x, stat) : S(0);
    return r;
}

// The codelet z-pass runs on single lines of S (Z_SCALAR), which halves the registers a line pair of single precision spectra takes, or on line pairs of T.
// Line l covers column x of row (pair) `row` at offset `off` within a block; false for the padding row of odd blocks. Consecutive scalar lines are the two
// halves of one T element.
#if Z_SCALAR
#define Z_LINES (HB * BLOCK * 2)
#else
#define Z_LINES (HB * BLOCK)
#endif

bool zLineOf(uint l, out uint off, out uint row, out uint x) {
#if Z_SCALAR
    x = l / 2 % BLOCK;
    row = l / (BLOCK * 2) * 2 + l % 2;
    off = row / 2 * BLOCK_P + x;
    return row < BLOCK;
#else
    x = l % BLOCK;
    row = l / BLOCK;
    off = row * BLOCK_P + x;
    return true;
#endif
}

#if STAT_IN_REF
#define Z_SUBGROUPS min(NUM_SG, (Z_LINES + SG - 1) / SG)
// The reference element of block 0 on line s * SG, the first z-line of subgroup s: once filtered, nothing reads it until the next group. A scalar line there
// (an even one) takes the first component.
uint statElem(uint s) {
    uint off, row, x;
    zLineOf(s * SG, off, row, x);
    return REF_GRP + swizzled(0u, off);
}
#if Z_SCALAR
#define STORE_STAT(i, v) G(i).x = (v)
#define LOAD_STAT(i) G(i).x
#else
#define STORE_STAT(i, v) G(i) = unpackFloat2x16(floatBitsToUint(v))
#define LOAD_STAT(i) uintBitsToFloat(packFloat2x16(G(i)))
#endif
#endif

#if PINGPONG
// Unnormalized DCT-II (forward) and DCT-III (inverse) matrix entries of blocks without a codelet, read from the table of cos(pi * q / (2 * BLOCK)), q in [0, 4
// * BLOCK), that leads the cos table.
float dctCoef(uint m, uint j, bool inverse) {
    if (inverse)
        return j == 0 ? 1.0 : 2.0 * cosTable.data[j * (m * 2 + 1) % (BLOCK * 4)];
    return 2.0 * cosTable.data[(j * 2 + 1) * m % (BLOCK * 4)];
}

uint zTableOffset(uint n) {
    return Z_TABLE_BASE + n * (n - 1) * 2 - MAX_CODELET * (MAX_CODELET + 1) * 2;
}
#endif
)glsl";

inline constexpr std::string_view filterTail = R"glsl(
void gather(uint k, float scale) {
    for (uint i = gl_LocalInvocationIndex; i < k * BLK_T; i += WG) {
        const uint z = i / BLK_T;
        const uint p = i % BLK_T / BLOCK_P;
        const uint x = i % BLOCK_P;
        const uvec2 m = member(z);
        const ivec2 pos = unpackPos(m.x);
        const uint offset = slotOf(pc.firstFrame + m.y) * PLANE_SIZE + (uint(pos.y) + p * 2) * WIDTH + uint(pos.x) + x;
        const bool inside = x < BLOCK;
        const bool second = p * 2 + 1 < BLOCK;
        const uint e = swizzled(z, i);
        G(e) = T(vec2(inside ? float(srcWin[plane].data[offset]) : 0.0, inside && second ? float(srcWin[plane].data[offset + WIDTH]) : 0.0) * scale);
#if WIENER
        G(REF_GRP + e) = T(vec2(inside ? float(refWin[plane].data[offset]) : 0.0, inside && second ? float(refWin[plane].data[offset + WIDTH]) : 0.0) * scale);
#endif
    }
}

#if BLOCK <= MAX_CODELET
void rowPass(uint k, uint numGroups, bool inverse) {
    for (uint l = gl_LocalInvocationIndex; l < numGroups * k * HB; l += WG) {
        const uint base = l / (k * HB) * GRP_T + l % (k * HB) * BLOCK_P;
        const uint z = l % (k * HB) / HB;
        T a[BLOCK], o[BLOCK];
        [[unroll]] for (uint x = 0; x < BLOCK; ++x)
            a[x] = G(swizzled(z, base + x));
        if (inverse)
            DCT3_BLOCK(a, o);
        else
            DCT2_BLOCK(a, o);
        [[unroll]] for (uint x = 0; x < BLOCK; ++x)
            G(swizzled(z, base + x)) = o[x];
    }
}

void colPass(uint k, uint numGroups, bool inverse) {
    for (uint l = gl_LocalInvocationIndex; l < numGroups * k * HB; l += WG) {
        const uint block = l % (k * HB) / HB;
        const uint base = l / (k * HB) * GRP_T + block * BLK_T + l % HB * 2;
        // With an even block stride, every column pair of a half-warp starts on an even one of its 16 bank pairs; odd blocks swap which column each component
        // holds to take the odd ones (the transforms are component-wise).
#if BLOCK_P % 2 == 1 && HB % 2 == 0
        const uint flip = block % 2;
#else
        const uint flip = 0;
#endif
        const uint first = base + flip;
        const uint second = base + 1 - flip;
        T a[BLOCK], o[BLOCK];
        [[unroll]] for (uint q = 0; q < HB; ++q) {
            const T u = G(swizzled(block, first + q * BLOCK_P));
            const T v = G(swizzled(block, second + q * BLOCK_P));
            a[q * 2] = T(u.x, v.x);
            if (q * 2 + 1 < BLOCK)
                a[q * 2 + 1] = T(u.y, v.y);
        }
        if (inverse)
            DCT3_BLOCK(a, o);
        else
            DCT2_BLOCK(a, o);
        [[unroll]] for (uint q = 0; q < HB; ++q) {
            const T odd = q * 2 + 1 < BLOCK ? o[min(q * 2 + 1, BLOCK - 1)] : T(0);
            G(swizzled(block, first + q * BLOCK_P)) = T(o[q * 2].x, odd.x);
            G(swizzled(block, second + q * BLOCK_P)) = T(o[q * 2].y, odd.y);
        }
    }
}
#else
// Out of place matrix forms of the passes for block sizes without a codelet: the forward row pass into the second buffer (aggregate inlines the inverse one),
// the column pass src -> dst. Like the matrix z-pass and aggregate, they sum in single precision even for half precision groups: half precision sums of so many
// terms lose up to 11 dB against the CPU (block_size=64) for at most 5% speed.
void rowPass(uint k, uint numGroups) {
    for (uint e = gl_LocalInvocationIndex; e < numGroups * k * HB * BLOCK; e += WG) {
        const uint m = e % BLOCK;
        const uint line = e / BLOCK;
        const uint base = line / (k * HB) * GRP_T + line % (k * HB) * BLOCK_P;
        vec2 sum = vec2(0.0);
        for (uint j = 0; j < BLOCK; ++j)
            sum += dctCoef(m, j, false) * vec2(G(base + j));
        G(BUF_T + base + m) = T(sum);
    }
}

void colPass(uint k, uint numGroups, bool inverse, uint src, uint dst) {
    for (uint e = gl_LocalInvocationIndex; e < numGroups * k * HB * BLOCK; e += WG) {
        const uint x = e % BLOCK;
        const uint q = e / BLOCK % HB;
        const uint block = e / (BLOCK * HB);
        const uint base = block / k * GRP_T + block % k * BLK_T + x;
        vec2 sum = vec2(0.0);
        for (uint j = 0; j < HB; ++j) {
            const vec2 v = vec2(G(src + base + j * BLOCK_P));
            sum.x += dctCoef(q * 2, j * 2, inverse) * v.x;
            sum.y += q * 2 + 1 < BLOCK ? dctCoef(q * 2 + 1, j * 2, inverse) * v.x : 0.0;
            if (j * 2 + 1 < BLOCK) {
                sum.x += dctCoef(q * 2, j * 2 + 1, inverse) * v.y;
                sum.y += q * 2 + 1 < BLOCK ? dctCoef(q * 2 + 1, j * 2 + 1, inverse) * v.y : 0.0;
            }
        }
        G(dst + base + q * BLOCK_P) = T(sum);
    }
}
#endif

#if GROUP > MAX_CODELET
// The z-pass of groups larger than any codelet: forward into the second buffer, shrink there, inverse back.
float zPassMatrix(uint k) {
    const uint table = zTableOffset(k);
    // The terms take the coefficients of dctCoef halved, their table indices (2j + 1) * m and j * (2m + 1) modulo 4k stepped rather than divided, and the sums
    // are doubled once.
    for (uint e = gl_LocalInvocationIndex; e < (WIENER + 1) * k * HB * BLOCK; e += WG) {
        const uint g = e / (k * HB * BLOCK);
        const uint m = e / (HB * BLOCK) % k;
        const uint off = g * GRP_T + e % (HB * BLOCK) / BLOCK * BLOCK_P + e % BLOCK;
        vec2 sum = vec2(0.0);
        uint q = m;
        for (uint j = 0; j < k; ++j) {
            sum += cosTable.data[table + q] * vec2(G(off + j * BLK_T));
            q += m * 2;
            q -= q >= k * 4 ? k * 4 : 0u;
        }
        G(BUF_T + off + m * BLK_T) = T(sum * 2.0);
    }
    SYNC();
    float stat = 0.0;
    for (uint e = gl_LocalInvocationIndex; e < k * HB * BLOCK; e += WG) {
        const uint z = e / (HB * BLOCK);
        const uint p = e / BLOCK % HB;
        const uint x = e % BLOCK;
        const uint i = BUF_T + z * BLK_T + p * BLOCK_P + x;
#if WIENER
        G(i) *= wienerWeight(G(i + GRP_T), z, p, x, stat);
#else
        G(i) = threshold(G(i), z, p, x, stat);
#endif
    }
    SYNC();
    for (uint e = gl_LocalInvocationIndex; e < k * HB * BLOCK; e += WG) {
        const uint m = e / (HB * BLOCK);
        const uint off = e % (HB * BLOCK) / BLOCK * BLOCK_P + e % BLOCK;
        vec2 sum = vec2(G(BUF_T + off)) * 0.5;
        uint q = 0;
        for (uint j = 1; j < k; ++j) {
            q += m * 2 + 1;
            q -= q >= k * 4 ? k * 4 : 0u;
            sum += cosTable.data[table + q] * vec2(G(BUF_T + off + j * BLK_T));
        }
        G(off + m * BLK_T) = T(sum * 2.0);
    }
    return stat;
}
#endif

float zPass(uint k) {
#if GROUP > MAX_CODELET
    if (k > MAX_CODELET)
        return zPassMatrix(k);
#endif
    float stat = 0.0;
    uint off, row, x;
#if STAT_IN_REF
    // Subgroup s starts at line s * SG, whose reference element then holds the subgroup's sum (see statElem).
    const uint first = gl_SubgroupID * SG + gl_SubgroupInvocationID;
#else
    const uint first = gl_LocalInvocationIndex;
#endif
    for (uint l = first; l < Z_LINES; l += WG)
        if (zLineOf(l, off, row, x))
            zLine(k, off, row, x, stat);
    return stat;
}

bool accumulates(uint frame) {
    return frame >= pc.activeLo;
}

// Adds a filtered sample pair of block z, rows 2p and 2p + 1 of column x, to the numerator of its frame.
void accumulatePair(uint z, uint p, uint x, vec2 v) {
    const uvec2 m = member(z);
    const uint frame = pc.firstFrame + m.y;
    if (!accumulates(frame))
        return;
    const ivec2 pos = unpackPos(m.x) + ivec2(x, p * 2);
    const uint i = accNum(frame, pos);
#if ACC_TILE
    if (frame == pc.firstFrame + pc.cur && groupInTile) {
        const uint t = uint(pos.y - tileOrigin.y) * TILE_SPAN + uint(pos.x - tileOrigin.x);
        accumulateTile(t, i, v.x);
        if (p * 2 + 1 < BLOCK)
            accumulateTile(t + TILE_SPAN, i + WIDTH, v.y);
        return;
    }
#endif
    accumulate(i, v.x);
    if (p * 2 + 1 < BLOCK)
        accumulate(i + WIDTH, v.y);
}

// Inverse row transform of the source group, aggregating the result with the given weights. The denominator receives the weight once per block, at the block's
// position; normalization spreads it over the block. Samples are accumulated element by element, so consecutive lanes add to consecutive addresses of one
// block.
void aggregate(uint k, float numWeight, float denWeight) {
#if ACC_TILE
    groupInTile = denWeight * TILE_SCALE >= 65536.0;
#endif
    for (uint z = gl_LocalInvocationIndex; z < k; z += WG) {
        const uvec2 m = member(z);
        const uint frame = pc.firstFrame + m.y;
        const ivec2 pos = unpackPos(m.x);
        if (accumulates(frame))
            accumulate(accDen(frame, pos), denWeight);
    }
#if BLOCK <= MAX_CODELET
    rowPass(k, 1, true);
    SYNC();
    for (uint i = gl_LocalInvocationIndex; i < k * BLK_T; i += WG) {
        const uint x = i % BLOCK_P;
        if (x < BLOCK)
            accumulatePair(i / BLK_T, i % BLK_T / BLOCK_P, x, vec2(G(swizzled(i / BLK_T, i))) * numWeight);
    }
#else
    for (uint e = gl_LocalInvocationIndex; e < k * HB * BLOCK; e += WG) {
        const uint z = e / (HB * BLOCK);
        const uint p = e / BLOCK % HB;
        const uint m = e % BLOCK;
        vec2 sum = vec2(0.0);
        for (uint j = 0; j < BLOCK; ++j)
            sum += dctCoef(m, j, true) * vec2(G(BUF_T + z * BLK_T + p * BLOCK_P + j));
        accumulatePair(z, p, m, sum * numWeight);
    }
#endif
}

void filterGroup(uint slot) {
    const uint k = counts.counts[slot];
    groupSlot = slot;

    // Prescaling by the inverse square root of the transform gain keeps half precision spectra in range.
    const float scale = inversesqrt(float(k * BLOCK * BLOCK * 8));
    gather(k, scale);
    SYNC();
#if BLOCK <= MAX_CODELET
    rowPass(k, WIENER + 1, false);
    SYNC();
    colPass(k, WIENER + 1, false);
#else
    rowPass(k, WIENER + 1);
    SYNC();
    colPass(k, WIENER + 1, false, BUF_T, 0);
#endif
    SYNC();

    const float stat = subgroupAdd(zPass(k));
#if STAT_IN_REF
    // Lane 0 filtered line s * SG itself; subgroups past Z_SUBGROUPS have no z-line and a sum of 0.
    if (gl_SubgroupInvocationID == 0 && gl_SubgroupID < Z_SUBGROUPS)
        STORE_STAT(statElem(gl_SubgroupID), stat);
    SYNC();
    float total = 0.0;
    [[unroll]] for (uint i = 0; i < Z_SUBGROUPS; ++i)
        total += LOAD_STAT(statElem(i));
#else
    if (subgroupElect())
        partial[gl_SubgroupID] = stat;
    SYNC();
    float total = 0.0;
    [[unroll]] for (uint i = 0; i < NUM_SG; ++i)
        total += partial[i];
#endif
#if WIENER
    const float denWeight = 1.0 / max(total, FLT_EPSILON);
#else
    const float denWeight = total < 1.0 ? 1.0 : 1.0 / total;
#endif

#if BLOCK <= MAX_CODELET
    colPass(k, 1, true);
#else
    colPass(k, 1, true, 0, BUF_T);
#endif
    SYNC();
    aggregate(k, denWeight * scale, denWeight);
    SYNC();
}

void main() {
    const uint activePlanes[] = uint[](ACTIVE_PLANES);
    plane = activePlanes[gl_WorkGroupID.y];
    float coefTable[PLANES * 4] = float[](COEF_TABLE);
    [[unroll]] for (uint i = 0; i < 4; ++i)
        coef[i] = coefTable[plane * 4 + i];
#if GLOBAL_GROUP
    groupBase = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * BUF_T * (PINGPONG + 1);
#endif

    const uint tilesX = (REF_COLS + TILE_REFS - 1) / TILE_REFS;
    const uint numTiles = (pc.refRows + TILE_REFS - 1) / TILE_REFS * tilesX;
#if ACC_TILE
    const uint numFrame = pc.firstFrame + pc.cur;
    const bool tileAccumulates = accumulates(numFrame);
#endif
    for (uint tile = gl_WorkGroupID.x; tile < numTiles; tile += gl_NumWorkGroups.x) {
        const uint row0 = pc.refRow0 + tile / tilesX * TILE_REFS;
        const uint col0 = tile % tilesX * TILE_REFS;
        const uint row1 = min(row0 + TILE_REFS, pc.refRow0 + pc.refRows);
        const uint col1 = min(col0 + TILE_REFS, REF_COLS);
#if ACC_TILE
        tileOrigin = max(refPosition(row0, col0) - SEARCH_MARGIN, ivec2(0));
        tileSize = min(refPosition(row1 - 1, col1 - 1) + BLOCK + SEARCH_MARGIN, ivec2(WIDTH, HEIGHT)) - tileOrigin;
        for (uint i = gl_LocalInvocationIndex; i < TILE_SPAN * TILE_SPAN; i += WG)
            tileNum[i] = 0;
        SYNC();
#endif

        for (uint row = row0; row < row1; ++row)
            for (uint col = col0; col < col1; ++col)
                filterGroup((row - pc.refRow0) * REF_COLS + col);

#if ACC_TILE
        if (tileAccumulates) {
            const uint base = accNum(numFrame, tileOrigin);
            for (uint i = gl_LocalInvocationIndex; i < uint(tileSize.x * tileSize.y); i += WG) {
                const ivec2 q = ivec2(i % uint(tileSize.x), i / uint(tileSize.x));
                const float v = tileValue(uint(q.y) * TILE_SPAN + uint(q.x));
                if (v != 0.0)
                    accumulate(base + uint(q.y) * WIDTH + uint(q.x), v);
            }
        }
        SYNC();
#endif
    }
}
)glsl";

} // namespace shader

#endif // BM3DVK_SHADERS_H
