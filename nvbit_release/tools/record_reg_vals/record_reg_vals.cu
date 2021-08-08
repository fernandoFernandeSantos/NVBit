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

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <vector>
#include <unordered_set>
#include <utility> // cbank list
#include <fstream> //final trace
#include <regex> // find cbank in the sass
#include <sstream>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

/**
 * Fernando mod
 * Final trace file
 */
constexpr char output_trace_file[] = "nvbit_trace_file.txt";
std::ofstream nvbit_trace_file;

/*get the c[bankid][bankoffset] list from the sass instruction*/
std::vector<std::pair<int32_t, int32_t>> extract_cbank_vector(const std::string& sass_line) {
	std::regex sass_regex(".*c\\[(0[xX][0-9a-fA-F]+)\\]\\[(0[xX][0-9a-fA-F]+)\\].*");
	std::smatch match;
	auto m = std::regex_match(sass_line, match, sass_regex);
	if (m == false && (sass_line.find("c[") != std::string::npos)) {
		std::cerr << "Problem when parsing the SASS line " << sass_line << std::endl;
		throw;
	}
	std::vector<std::pair<int32_t, int32_t>> cbank_list;
	for (uint32_t i = 1; i < match.size(); i += 2) {
		auto bank_id = std::stoi(match[i], nullptr, 16);
		auto bank_offset = std::stoi(match[i + 1], nullptr, 16);
		std::pair < int32_t, int32_t > cbank(bank_id, bank_offset);
		cbank_list.push_back(cbank);
	}
	return cbank_list;
}

/**************************************************************************/

void nvbit_at_init() {
	setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
	GET_VAR_INT(instr_begin_interval, "INSTR_BEGIN", 0,
			"Beginning of the instruction interval where to apply instrumentation");
	GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
			"End of the instruction interval where to apply instrumentation");
	GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
	//std::string pad(100, '-');
	//printf("%s\n", pad.c_str());
	nvbit_trace_file.open(output_trace_file, std::ios::out);
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
	/* Get related functions of the kernel (device function that can be
	 * called by the kernel) */
	std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

	/* add kernel itself to the related function vector */
	related_functions.push_back(func);

	/* iterate on function */
	for (auto f : related_functions) {
		/* "recording" function was instrumented, if set insertion failed
		 * we have already encountered this function */
		if (!already_instrumented.insert(f).second) {
			continue;
		}
		const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
		if (verbose) {
			printf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f),
					nvbit_get_func_addr(f));
		}

		uint32_t cnt = 0;
		/* iterate on all the static instructions in the function */
		for (auto instr : instrs) {
			if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
				cnt++;
				continue;
			}
			if (verbose) {
				instr->printDecoded();
			}

			if (sass_to_id_map.find(instr->getSass()) == sass_to_id_map.end()) {
				int opcode_id = sass_to_id_map.size();
				sass_to_id_map[instr->getSass()] = opcode_id;
				id_to_sass_map[opcode_id] = std::string(instr->getSass());
			}

			int opcode_id = sass_to_id_map[instr->getSass()];
			std::vector<int> reg_num_list;
			/* iterate on the operands */
			for (int i = 0; i < instr->getNumOperands(); i++) {
				/* get the operand "i" */
				const InstrType::operand_t *op = instr->getOperand(i);
				if (op->type == InstrType::OperandType::REG) {
					reg_num_list.push_back(op->u.reg.num);
				}
			}
			/* insert call to the instrumentation function with its
			 * arguments */
			nvbit_insert_call(instr, "record_reg_val", IPOINT_BEFORE);
			/* guard predicate value */
			nvbit_add_call_arg_guard_pred_val(instr);
			/* opcode id */
			nvbit_add_call_arg_const_val32(instr, opcode_id);
			/* add pointer to channel_dev*/
			nvbit_add_call_arg_const_val64(instr, (uint64_t) & channel_dev);
			/* how many register values are passed next */
			nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
			/**************************************************************************
			 * Edit: trying to load all the cbank values
			 **************************************************************************/
			// extract vector of pair with c[bankid][bankoffset]
			auto cbank_values = extract_cbank_vector(instr->getSass());
			// how many constant operands
			nvbit_add_call_arg_const_val32(instr, cbank_values.size());
			// For some reason I have to put the size of the operands at
			// the end of the var list
			nvbit_add_call_arg_const_val32(instr, reg_num_list.size() + cbank_values.size());

			//REGs FIRST as I will read them before the cbank values
			for (int num : reg_num_list) {
				/* last parameter tells it is a variadic parameter passed to
				 * the instrument function record_reg_val() */
				nvbit_add_call_arg_reg_val(instr, num, true);
			}

			//instrument the constant operands
			for (auto& cbank : cbank_values) {
//				std::cout << "SASS: " << instr->getSass() << " - c[" << cbank.first << "]["
//						<< cbank.second << "]\n";
				nvbit_add_call_arg_cbank_val(instr, cbank.first, cbank.second, true);
			}
			/**************************************************************************/
			cnt++;
		}
	}
}

__global__ void flush_channel() {
	/* push memory access with negative cta id to communicate the kernel is
	 * completed */
	reg_info_t ri;
	ri.cta_id_x = -1;
	channel_dev.push(&ri, sizeof(reg_info_t));

	/* flush channel */
	channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name,
		void *params, CUresult *pStatus) {
	if (skip_flag)
		return;

	if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel) {
		cuLaunchKernel_params *p = (cuLaunchKernel_params *) params;

		if (!is_exit) {
			int nregs;
			CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

			int shmem_static_nbytes;
			CUDA_SAFECALL(
					cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
							p->f));

			instrument_function_if_needed(ctx, p->f);

			nvbit_enable_instrumented(ctx, p->f, true);

//			printf("Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
//					"%d - shmem %d - cuda stream id %ld\n", nvbit_get_func_name(ctx, p->f),
//					p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ,
//					nregs, shmem_static_nbytes + p->sharedMemBytes, (uint64_t) p->hStream);
			recv_thread_receiving = true;

			/**
			 * Fernando mod
			 */
			nvbit_trace_file << "Kernel " << nvbit_get_func_name(ctx, p->f) << " gridsize "
					<< p->gridDimX << "," << p->gridDimY << "," << p->gridDimZ << " blocksize "
					<< p->blockDimX << "," << p->blockDimY << "," << p->blockDimZ << " nregs "
					<< nregs << " shmem " << shmem_static_nbytes + p->sharedMemBytes
					<< " cudastreamid " << (uint64_t) p->hStream << std::endl;

//			nvbit_trace_file << "cta_id_x,cta_id_y,cta_id_z,warp_id,global_warp_id,sm_id,lane_id,opcode\n";

		} else {
			/* make sure current kernel is completed */
			cudaDeviceSynchronize();
			cudaError_t kernelError = cudaGetLastError();
			if (kernelError != cudaSuccess) {
				printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
				assert(0);
			}

			/* make sure we prevent re-entry on the nvbit_callback when issuing
			 * the flush_channel kernel */
			skip_flag = true;

			/* issue flush of channel so we are sure all the memory accesses
			 * have been pushed */
			flush_channel<<<1, 1>>>();
			cudaDeviceSynchronize();
			assert(cudaGetLastError() == cudaSuccess);

			/* unset the skip flag */
			skip_flag = false;

			/* wait here until the receiving thread has not finished with the
			 * current kernel */
			while (recv_thread_receiving) {
				pthread_yield();
			}
		}
	}
}

void print_data(reg_info_t* ri) {
	printf("CTA %d,%d,%d - warp %d - %s:\n", ri->cta_id_x, ri->cta_id_y, ri->cta_id_z, ri->warp_id,
			id_to_sass_map[ri->opcode_id].c_str());
	for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
		printf("* ");
		for (int i = 0; i < WARP_SIZE; i++) {
			printf("Reg%d_T%d: 0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_data_csv(reg_info_t* ri) {
//	printf("CTA %d,%d,%d - NCTA %d,%d,%d - WARPID %d - GWARPID %d - SMID %d - LANEID %d - ",
//			ri->cta_id_x, ri->cta_id_y, ri->cta_id_z, // CTA
//			ri->ncta_id_x, ri->ncta_id_y, ri->ncta_id_z, // NCTA
//			ri->warp_id, ri->global_warp_id, ri->sm_id, ri->lane_id //WARP, global WARP, SM and LANE ID
//			);

	nvbit_trace_file << "CTA " << ri->cta_id_x << "," << ri->cta_id_y << "," << ri->cta_id_z
			<< // CTA
			" NCTA " << ri->ncta_id_x << "," << ri->ncta_id_y << "," << ri->ncta_id_z
			<< // NCTA
			" WARPID " << ri->warp_id << " GWARPID " << ri->global_warp_id << " SMID " << ri->sm_id
			<< " LANEID " << ri->lane_id << " " << id_to_sass_map[ri->opcode_id] << std::endl;

//	nvbit_trace_file << ri->cta_id_x << "," << ri->cta_id_y                   << "," << ri->cta_id_z
//				<< ri->warp_id  << "," << ri->global_warp_id             << "," << ri->sm_id
//				<< ri->lane_id  << "," << id_to_sass_map[ri->opcode_id]  << ",";

//	printf("%s\n", id_to_sass_map[ri->opcode_id].c_str());
//	nvbit_trace_file << id_to_sass_map[ri->opcode_id] << std::endl;
	char temp[128];
	for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
		for (int i = 0; i < WARP_SIZE; i++) {
//			printf("R%dT%d:0x%08x ", reg_idx, i, ri->reg_vals[i][reg_idx]);
			sprintf(temp, "R%dT%d:0x%08x ", reg_idx, ri->lane_id, ri->reg_vals[i][reg_idx]);
			nvbit_trace_file << temp;
		}
		nvbit_trace_file << std::endl;
	}

	/* Print to the file the constant values */
	for (int cbank_idx = 0; cbank_idx < ri->num_cbank; cbank_idx++) {
		for (int i = 0; i < WARP_SIZE; i++) {
			sprintf(temp, "C%dT%d:0x%08x ", cbank_idx, ri->lane_id, ri->cbank_vals[i][cbank_idx]);
			nvbit_trace_file << temp;
		}
		nvbit_trace_file << std::endl;
	}
}

void *recv_thread_fun(void *) {
	char *recv_buffer = (char *) malloc(CHANNEL_SIZE);

	while (recv_thread_started) {
		uint32_t num_recv_bytes = 0;
		if (recv_thread_receiving
				&& (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) > 0) {
			uint32_t num_processed_bytes = 0;
			while (num_processed_bytes < num_recv_bytes) {
				reg_info_t *ri = (reg_info_t *) &recv_buffer[num_processed_bytes];

				/* when we get this cta_id_x it means the kernel has completed
				 */
				if (ri->cta_id_x == -1) {
					recv_thread_receiving = false;
					break;
				}

				print_data_csv(ri);
				num_processed_bytes += sizeof(reg_info_t);
			}
		}
	}
	free(recv_buffer);
	return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
	recv_thread_started = true;
	channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
	pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {
	if (recv_thread_started) {
		recv_thread_started = false;
		pthread_join(recv_thread, NULL);
	}
}
