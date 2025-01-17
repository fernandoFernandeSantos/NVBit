/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <stdio.h>
#include <cstdarg>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

extern "C" __device__ __noinline__
void record_reg_val(int pred, int opcode_id, uint64_t pchannel_dev, int32_t num_regs,
                    int32_t instruction_index, int32_t num_cbank, int32_t va_list_size ...) {
	if (!pred) {
		return;
	}

	unsigned active_mask = __ballot_sync(__activemask(), 1);
	const int laneid = (int32_t)get_laneid();
	const int first_laneid = __ffs((int32_t) active_mask) - 1;

	reg_info_t ri;

	int4 cta = get_ctaid();
	ri.cta_id_x = cta.x;
	ri.cta_id_y = cta.y;
	ri.cta_id_z = cta.z;
	ri.warp_id = (int32_t) get_warpid();
	/* Add new data to be logged */
	ri.lane_id = laneid;
	ri.sm_id = (int32_t) get_smid();
	int4 ncta = get_nctaid();
	ri.ncta_id_x = ncta.x;
	ri.ncta_id_y = ncta.y;
	ri.ncta_id_z = ncta.z;
	ri.global_warp_id = get_global_warp_id();
	/*-------------------*/
	ri.opcode_id = opcode_id;
	ri.num_regs = num_regs;
	ri.num_cbank = num_cbank;
    ri.instruction_index = instruction_index;

	if (va_list_size) {
		va_list vl;
		va_start(vl, va_list_size);

		if (num_regs) {
			for (int i = 0; i < num_regs; i++) {
				uint32_t val = va_arg(vl, uint32_t);

				/* collect register values from other threads */
				for (int tid = 0; tid < WARP_SIZE; tid++) {
					ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
				}
			}
		}
		/**************************************************************************
		 Edit: trying to load all the cbank values
		 **************************************************************************/
		if (num_cbank) {
			for (int i = 0; i < num_cbank; i++) {
				uint32_t val = va_arg(vl, uint32_t);

				/* collect cbank values from other threads */
				for (int tid = 0; tid < WARP_SIZE; tid++) {
					ri.cbank_vals[tid][i] = __shfl_sync(active_mask, val, tid);
				}
			}
		}
		/**************************************************************************/
		va_end(vl);
	}

	/* first active lane pushes information on the channel */
	if (first_laneid == laneid) {
		auto *channel_dev = (ChannelDev*) pchannel_dev;
		channel_dev->push(&ri, sizeof(reg_info_t));
	}
}

