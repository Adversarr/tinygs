/*
 * Tiny self-contained version of the PCG Random Number Generation for C++
 * put together from pieces of the much larger C/C++ codebase.
 * Wenzel Jakob, February 2015
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 *
 * Note: This code was modified to work with CUDA by the tiny-cuda-nn authors.
 */

#pragma once

#include "tinygs/common.hpp"

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

namespace tinygs {

/// @brief PCG32 Pseudorandom number generator
struct pcg32 {
	/// @brief Initialize with default seed
	TINYGS_HOST_DEVICE pcg32() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

	/// @brief Initialize with custom seed
	TINYGS_HOST_DEVICE pcg32(uint64_t initstate, uint64_t initseq = 1u) { seed(initstate, initseq); }

	/// @brief Seed the pseudorandom number generator
	/// @param initstate State initializer
	/// @param initseq Sequence selection constant
	TINYGS_HOST_DEVICE void seed(uint64_t initstate, uint64_t initseq = 1) {
		state = 0U;
		inc = (initseq << 1u) | 1u;
		next_uint();
		state += initstate;
		next_uint();
	}

	/// @brief Generate uniformly distributed 32-bit random number
	TINYGS_HOST_DEVICE uint32_t next_uint() {
		uint64_t oldstate = state;
		state = oldstate * PCG32_MULT + inc;
		uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
		uint32_t rot = (uint32_t) (oldstate >> 59u);
		return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
	}

	/// @brief Generate uniformly distributed number in range [0, bound)
	TINYGS_HOST_DEVICE uint32_t next_uint(uint32_t bound) {
		// To avoid bias, we need to make the range of the RNG a multiple of
		// bound, which we do by dropping output less than a threshold.
		// A naive scheme to calculate the threshold would be to do
		//
		//     uint32_t threshold = 0x100000000ull % bound;
		//
		// but 64-bit div/mod is slower than 32-bit div/mod (especially on
		// 32-bit platforms).  In essence, we do
		//
		//     uint32_t threshold = (0x100000000ull-bound) % bound;
		//
		// because this version will calculate the same modulus, but the LHS
		// value is less than 2^32.

		uint32_t threshold = (~bound+1u) % bound;

		// Uniformity guarantees that this loop will terminate.  In practice, it
		// should usually terminate quickly; on average (assuming all bounds are
		// equally likely), 82.25% of the time, we can expect it to require just
		// one iteration.  In the worst case, someone passes a bound of 2^31 + 1
		// (i.e., 2147483649), which invalidates almost 50% of the range.  In
		// practice, bounds are typically small and only a tiny amount of the range
		// is eliminated.
		for (;;) {
			uint32_t r = next_uint();
			if (r >= threshold)
				return r % bound;
		}
	}

	/// @brief Generate single precision float in [0, 1)
	TINYGS_HOST_DEVICE float next_float() {
		/* Trick from MTGP: generate an uniformly distributed
			single precision number in [1,2) and subtract 1. */
		union {
			uint32_t u;
			float f;
		} x;
		x.u = (next_uint() >> 9) | 0x3f800000u;
		return x.f - 1.0f;
	}

	/// @brief Generate double precision float in [0, 1)
	TINYGS_HOST_DEVICE double next_double() {
		/* Trick from MTGP: generate an uniformly distributed
			double precision number in [1,2) and subtract 1. */
		union {
			uint64_t u;
			double d;
		} x;
		x.u = ((uint64_t) next_uint() << 20) | 0x3ff0000000000000ULL;
		return x.d - 1.0;
	}

	/// @brief Multi-step advance function (jump-ahead, jump-back)
	/// @param delta_ Number of steps to advance (default 2^32)
	TINYGS_HOST_DEVICE void advance(int64_t delta_ = (1ll<<32)) {
		uint64_t
			cur_mult = PCG32_MULT,
			cur_plus = inc,
			acc_mult = 1u,
			acc_plus = 0u;

		/* Even though delta is an unsigned integer, we can pass a signed
			integer to go backwards, it just goes "the long way round". */
		uint64_t delta = (uint64_t) delta_;

		while (delta > 0) {
			if (delta & 1) {
				acc_mult *= cur_mult;
				acc_plus = acc_plus * cur_mult + cur_plus;
			}
			cur_plus = (cur_mult + 1) * cur_plus;
			cur_mult *= cur_mult;
			delta /= 2;
		}
		state = acc_mult * state + acc_plus;
	}

	/// @brief Compute distance between two PCG32 generators
	TINYGS_HOST_DEVICE int64_t operator-(const pcg32 &other) const {
		uint64_t
			cur_mult = PCG32_MULT,
			cur_plus = inc,
			cur_state = other.state,
			the_bit = 1u,
			distance = 0u;

		while (state != cur_state) {
			if ((state & the_bit) != (cur_state & the_bit)) {
				cur_state = cur_state * cur_mult + cur_plus;
				distance |= the_bit;
			}

			the_bit <<= 1;
			cur_plus = (cur_mult + 1ULL) * cur_plus;
			cur_mult *= cur_mult;
		}

		return (int64_t) distance;
	}

	/// @brief Equality operator
	TINYGS_HOST_DEVICE bool operator==(const pcg32 &other) const { return state == other.state && inc == other.inc; }

	/// @brief Inequality operator
	TINYGS_HOST_DEVICE bool operator!=(const pcg32 &other) const { return state != other.state || inc != other.inc; }

	uint64_t state;  // RNG state.  All values are possible.
	uint64_t inc;    // Controls which RNG sequence (stream) is selected. Must *always* be odd.
};

}
