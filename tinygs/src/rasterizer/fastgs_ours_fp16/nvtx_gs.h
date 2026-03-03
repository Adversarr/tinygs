#pragma once
#include <nvtx3/nvtx3.hpp>
#include <cuda_runtime.h>

namespace gs_nvtx {

// Domain
struct domain { static constexpr char const* name{"fast_gs"}; };
using range  = nvtx3::scoped_range_in<domain>;
using attr   = nvtx3::event_attributes;
using regstr = nvtx3::registered_string_in<domain>;
using ncat   = nvtx3::named_category_in<domain>;

// Categories
struct cat_kernel { static constexpr char const* name{"KERNEL"}; static constexpr uint32_t id{1}; };
struct cat_cub    { static constexpr char const* name{"CUB"};    static constexpr uint32_t id{2}; };
struct cat_mem    { static constexpr char const* name{"MEM"};    static constexpr uint32_t id{3}; };
struct cat_copy   { static constexpr char const* name{"COPY"};   static constexpr uint32_t id{4}; };

// Colors
static constexpr nvtx3::rgb C_BLUE   {  0,153,255};
static constexpr nvtx3::rgb C_ORANGE {255,153,  0};
static constexpr nvtx3::rgb C_PURPLE {153,102,255};
static constexpr nvtx3::rgb C_GREEN  { 51,204, 51};
static constexpr nvtx3::rgb C_PINK   {255,102,153};
static constexpr nvtx3::rgb C_CYAN   {102,204,255};
static constexpr nvtx3::rgb C_RED    {255, 51, 51};
static constexpr nvtx3::rgb C_GRAY   {102,102,102};

// Construct-on-first-use category getters
inline ncat const& catK()  { return ncat::get<cat_kernel>(); }
inline ncat const& catC()  { return ncat::get<cat_cub>(); }
inline ncat const& catM()  { return ncat::get<cat_mem>(); }
inline ncat const& catCp() { return ncat::get<cat_copy>(); }

// Common registered message tags (can be extended as needed)
struct m_preprocess          { static constexpr char const* message{"preprocess"}; };
struct m_sort_depth          { static constexpr char const* message{"sort_depth"}; };
struct m_apply_depth_order   { static constexpr char const* message{"apply_depth_ordering"}; };
struct m_scan_primitive      { static constexpr char const* message{"scan_primitive_offsets"}; };
struct m_create_instances    { static constexpr char const* message{"create_instances"}; };
struct m_sort_tiles          { static constexpr char const* message{"sort_tiles"}; };
struct m_extract_ranges      { static constexpr char const* message{"extract_instance_ranges"}; };
struct m_bucket_counts       { static constexpr char const* message{"extract_bucket_counts"}; };
struct m_scan_buckets        { static constexpr char const* message{"scan_bucket_counts"}; };
struct m_blend               { static constexpr char const* message{"blend"}; };
struct m_blend_backward      { static constexpr char const* message{"blend_backward"}; };
struct m_preprocess_backward { static constexpr char const* message{"preprocess_backward"}; };
struct m_reduce_w2c_grad     { static constexpr char const* message{"reduce_w2c_grad"}; };
struct m_memset_per_tile     { static constexpr char const* message{"memset_per_tile"}; };
struct m_memset_per_prim     { static constexpr char const* message{"memset_per_primitive"}; };
struct m_copy_counts_d2h     { static constexpr char const* message{"copy_counts_d2h"}; };

// Helpers: unique name maker
#define GS_CONCAT_(a,b) a##b
#define GS_CONCAT(a,b) GS_CONCAT_(a,b)

// Domain-scoped function range
#define GS_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(::gs_nvtx::domain)

// Range with registered message tag (TAG is a struct with `message`)
#define GS_RANGE_SCOPE(TAG, COLOR, CAT, PAYLOAD) \
  auto& GS_CONCAT(_gs_msg_, __LINE__) = ::gs_nvtx::regstr::get<TAG>(); \
  nvtx3::event_attributes GS_CONCAT(_gs_attr_, __LINE__){GS_CONCAT(_gs_msg_, __LINE__), COLOR, CAT, nvtx3::payload{static_cast<int64_t>(PAYLOAD)}}; \
  ::gs_nvtx::range GS_CONCAT(_gs_range_, __LINE__){GS_CONCAT(_gs_attr_, __LINE__)};

} // namespace gs_nvtx