#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/workloads/workload_types.hpp>

namespace pipepye::workloads {

struct RefinerySchedulingParams {
    int num_units{3};
    int num_modes{2};
    int num_periods{6};
    uint64_t seed{42};
    InstanceScale scale{InstanceScale::Small};

    static RefinerySchedulingParams ForScale(InstanceScale scale) {
        RefinerySchedulingParams p;
        p.scale = scale;
        switch (scale) {
            case InstanceScale::Toy:
                p.num_units = 2; p.num_modes = 2; p.num_periods = 3; break;
            case InstanceScale::Small:
                p.num_units = 3; p.num_modes = 2; p.num_periods = 6; break;
            case InstanceScale::Medium:
                p.num_units = 5; p.num_modes = 3; p.num_periods = 12; break;
            case InstanceScale::Large:
                p.num_units = 8; p.num_modes = 3; p.num_periods = 24; break;
            case InstanceScale::Stress:
                p.num_units = 12; p.num_modes = 4; p.num_periods = 48; break;
        }
        return p;
    }
};

/// @brief Case C: Mixed-Integer Linear Program (MILP) for refinery unit scheduling.
/// Combines discrete unit on/off operating decisions with continuous mass flows,
/// yield matrices, inventory buffers, and demand targets.
class RefinerySchedulingGenerator {
public:
    static std::pair<model::LinearProgram, WorkloadMetadata> generate(
        const RefinerySchedulingParams& params = RefinerySchedulingParams::ForScale(InstanceScale::Small));
};

} // namespace pipepye::workloads
