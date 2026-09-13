#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/workloads/workload_types.hpp>

namespace pipepye::workloads {

struct UnitCommitmentParams {
    int num_generators{5};
    int num_periods{12};
    uint64_t seed{42};
    InstanceScale scale{InstanceScale::Small};

    static UnitCommitmentParams ForScale(InstanceScale scale) {
        UnitCommitmentParams p;
        p.scale = scale;
        switch (scale) {
            case InstanceScale::Toy:
                p.num_generators = 3; p.num_periods = 4; break;
            case InstanceScale::Small:
                p.num_generators = 5; p.num_periods = 12; break;
            case InstanceScale::Medium:
                p.num_generators = 10; p.num_periods = 24; break;
            case InstanceScale::Large:
                p.num_generators = 20; p.num_periods = 48; break;
            case InstanceScale::Stress:
                p.num_generators = 50; p.num_periods = 168; break;
        }
        return p;
    }
};

/// @brief Case D: Power system Unit Commitment & Economic Dispatch MILP.
/// Models thermal generator commitment, generation limits, ramp-rate dynamics,
/// system load balance, and spinning reserve requirements.
class UnitCommitmentGenerator {
public:
    static std::pair<model::LinearProgram, WorkloadMetadata> generate(
        const UnitCommitmentParams& params = UnitCommitmentParams::ForScale(InstanceScale::Small));
};

} // namespace pipepye::workloads
