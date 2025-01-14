/*
 * Copyright (c) 2004-2006 The Regents of The University of Michigan
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/pred/2bit_correlating.hh"

#include "base/intmath.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/Fetch.hh"

namespace gem5
{

namespace branch_prediction
{

CorrelatingBP::CorrelatingBP(const CorrelatingBPParams &params)
    : BPredUnit(params),
      localPredictorSize(params.localPredictorSize),
      localCtrBits(params.localCtrBits),
      globalHistoryBits(params.globalHistoryBits),
      globalHistorySize(1U << globalHistoryBits),
      localPredictorSets(localPredictorSize / localCtrBits / globalHistorySize),
      localCtrs(localPredictorSets, std::vector<SatCounter8>(globalHistorySize, SatCounter8(localCtrBits))),
      globalHistory(params.numThreads, 0),
      indexMask(localPredictorSets - 1)
{
    if (!isPowerOf2(localPredictorSize)) {
        fatal("Invalid local predictor size!\n");
    }

    if (!isPowerOf2(localPredictorSets)) {
        fatal("Invalid number of local predictor sets! Check localCtrBits.\n");
    }

    globalHistoryMask = globalHistorySize - 1;

    DPRINTF(Fetch, "index mask: %#x\n", indexMask);

    DPRINTF(Fetch, "local predictor size: %i\n",
            localPredictorSize);

    DPRINTF(Fetch, "local counter bits: %i\n", localCtrBits);

    DPRINTF(Fetch, "instruction shift amount: %i\n",
            instShiftAmt);
}

void
CorrelatingBP::btbUpdate(ThreadID tid, Addr branch_addr, void * &bp_history)
{
// Place holder for a function that is called to update predictor history when
// a BTB entry is invalid or not found.
}


bool
CorrelatingBP::lookup(ThreadID tid, Addr branch_addr, void * &bp_history)
{
    bool taken;
    unsigned local_predictor_idx = getLocalIndex(branch_addr);

    DPRINTF(Fetch, "Looking up index %#x\n",
            local_predictor_idx);

    unsigned global_history_idx = globalHistory[tid] & globalHistoryMask;

    DPRINTF(Fetch, "Looking up global history %#x\n", global_history_idx);

    uint8_t counter_val = localCtrs[local_predictor_idx][global_history_idx];

    DPRINTF(Fetch, "prediction is %i.\n",
            (int)counter_val);

    taken = getPrediction(counter_val);

    return taken;
}

inline
void
CorrelatingBP::updateGlobalHistTaken(ThreadID tid)
{
    globalHistory[tid] = (globalHistory[tid] << 1) | 1;
}

inline
void
CorrelatingBP::updateGlobalHistNotTaken(ThreadID tid)
{
    globalHistory[tid] = (globalHistory[tid] << 1);
}

void
CorrelatingBP::update(ThreadID tid, Addr branch_addr, bool taken, void *bp_history,
                bool squashed, const StaticInstPtr & inst, Addr corrTarget)
{
    assert(bp_history == NULL);
    unsigned local_predictor_idx;

    // No state to restore, and we do not update on the wrong
    // path.
    if (squashed) {
        return;
    }

    // Update the local predictor.
    local_predictor_idx = getLocalIndex(branch_addr);

    DPRINTF(Fetch, "Looking up index %#x\n", local_predictor_idx);

    unsigned global_history_idx = globalHistory[tid] & globalHistoryMask;

    DPRINTF(Fetch, "Looking up global history %#x\n", global_history_idx);

    if (taken) {
        DPRINTF(Fetch, "Branch updated as taken.\n");
        localCtrs[local_predictor_idx][global_history_idx]++;
        updateGlobalHistTaken(tid);
    } else {
        DPRINTF(Fetch, "Branch updated as not taken.\n");
        localCtrs[local_predictor_idx][global_history_idx]--;
        updateGlobalHistNotTaken(tid);
    }
}

inline
bool
CorrelatingBP::getPrediction(uint8_t &count)
{
    // Get the MSB of the count
    return (count >> (localCtrBits - 1));
}

inline
unsigned
CorrelatingBP::getLocalIndex(Addr &branch_addr)
{
    return (branch_addr >> instShiftAmt) & indexMask;
}

void
CorrelatingBP::uncondBranch(ThreadID tid, Addr pc, void *&bp_history)
{
}

} // namespace branch_prediction
} // namespace gem5
