#pragma once
#include <nvtx3/nvtx3.hpp>

namespace dl_nvtx {

// Domain for dataloader-related NVTX ranges
struct domain { static constexpr char const* name{"dataloader"}; };
using range  = nvtx3::scoped_range_in<domain>;
using attr   = nvtx3::event_attributes;
using regstr = nvtx3::registered_string_in<domain>;
using ncat   = nvtx3::named_category_in<domain>;

// Categories
struct cat_main     { static constexpr char const* name{"MAIN"};    static constexpr uint32_t id{1}; };
struct cat_prefetch { static constexpr char const* name{"PREFETCH"}; static constexpr uint32_t id{2}; };

// Construct-on-first-use category getters
inline ncat const& catMain()     { return ncat::get<cat_main>(); }
inline ncat const& catPrefetch() { return ncat::get<cat_prefetch>(); }

// Colors
static constexpr nvtx3::rgb C_BLUE   {  0,153,255};
static constexpr nvtx3::rgb C_ORANGE {255,153,  0};
static constexpr nvtx3::rgb C_GREEN  { 51,204, 51};
static constexpr nvtx3::rgb C_RED    {255, 51, 51};
static constexpr nvtx3::rgb C_GRAY   {102,102,102};

// Helpers: unique name maker
#define DL_CONCAT_(a,b) a##b
#define DL_CONCAT(a,b) DL_CONCAT_(a,b)

// Domain-scoped function range
#define DL_FUNC_RANGE() NVTX3_FUNC_RANGE_IN(::dl_nvtx::domain)

// Range with registered message tag (TAG is a struct with `message`)
#define DL_RANGE_SCOPE(TAG, COLOR, CAT, PAYLOAD) \
  auto& DL_CONCAT(_dl_msg_, __LINE__) = ::dl_nvtx::regstr::get<TAG>(); \
  nvtx3::event_attributes DL_CONCAT(_dl_attr_, __LINE__){DL_CONCAT(_dl_msg_, __LINE__), COLOR, CAT, nvtx3::payload{static_cast<int64_t>(PAYLOAD)}}; \
  ::dl_nvtx::range DL_CONCAT(_dl_range_, __LINE__){DL_CONCAT(_dl_attr_, __LINE__)};

// Range with literal message
#define DL_RANGE_SCOPE_LIT(MSG_LIT, COLOR, CAT, PAYLOAD) \
  nvtx3::event_attributes DL_CONCAT(_dl_attr_, __LINE__){MSG_LIT, COLOR, CAT, nvtx3::payload{static_cast<int64_t>(PAYLOAD)}}; \
  ::dl_nvtx::range DL_CONCAT(_dl_range_, __LINE__){DL_CONCAT(_dl_attr_, __LINE__)};

// Common registered message tags
struct m_next         { static constexpr char const* message{"next"}; };
struct m_reset        { static constexpr char const* message{"reset"}; };
struct m_prefetch     { static constexpr char const* message{"prefetch"}; };
struct m_transfer_gpu { static constexpr char const* message{"transfer_gpu"}; };

} // namespace dl_nvtx