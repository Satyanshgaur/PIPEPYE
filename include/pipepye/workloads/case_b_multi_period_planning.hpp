#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/workloads/workload_types.hpp>

namespace pipepye::workloads {

struct MultiPeriodPlanningParams {
    int num_periods{10};
    int num_products{10};
    int num_machines{5};
    uint64_t seed{42};
    InstanceScale scale{InstanceScale::Small};

    static MultiPeriodPlanningParams ForScale(InstanceScale scale) {
        MultiPeriodPlanningParams p;
        p.scale = scale;
        switch (scale) {
            case InstanceScale::Toy:
                p.num_periods = 5; p.num_products = 4; p.num_machines = 2; break;
            case InstanceScale::Small:
                p.num_periods = 10; p.num_products = 10; p.num_machines = 5; break;
            case InstanceScale::Medium:
                p.num_periods = 50; p.num_products = 20; p.num_machines = 10; break;
            case InstanceScale::Large:
                p.num_periods = 100; p.num_products = 30; p.num_machines = 15; break;
            case InstanceScale::Stress:
                p.num_periods = 250; p.num_products = 50; p.num_machines = 20; break;
        }
        return p;
    }
};

/// @brief Case B: Strongly structured multi-period production and inventory planning LP.
/// Exhibits canonical block-angular / staircase sparse matrix topology linking consecutive time steps.
class MultiPeriodPlanningGenerator {
public:
    static std::pair<model::LinearProgram, WorkloadMetadata> generate(
        const MultiPeriodPlanningParams& params = MultiPeriodPlanningParams::ForScale(InstanceScale::Small));
};

} // namespace pipepye::workloads
