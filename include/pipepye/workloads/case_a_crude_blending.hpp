#pragma once

#include <pipepye/model/lp_model.hpp>
#include <pipepye/workloads/workload_types.hpp>

namespace pipepye::workloads {

struct CrudeBlendingParams {
    int num_crudes{5};
    int num_products{3};
    int num_qualities{3};
    uint64_t seed{42};
    InstanceScale scale{InstanceScale::Small};

    static CrudeBlendingParams ForScale(InstanceScale scale) {
        CrudeBlendingParams p;
        p.scale = scale;
        switch (scale) {
            case InstanceScale::Toy:
                p.num_crudes = 3; p.num_products = 2; p.num_qualities = 2; break;
            case InstanceScale::Small:
                p.num_crudes = 5; p.num_products = 3; p.num_qualities = 3; break;
            case InstanceScale::Medium:
                p.num_crudes = 12; p.num_products = 6; p.num_qualities = 4; break;
            case InstanceScale::Large:
                p.num_crudes = 25; p.num_products = 10; p.num_qualities = 6; break;
            case InstanceScale::Stress:
                p.num_crudes = 50; p.num_products = 20; p.num_qualities = 8; break;
        }
        return p;
    }
};

/// @brief Case A: Industrially motivated continuous crude oil blending LP.
/// Models refinery stream blending with quality specifications, availability limits,
/// component conservation, and profit optimization.
class CrudeBlendingGenerator {
public:
    static std::pair<model::LinearProgram, WorkloadMetadata> generate(
        const CrudeBlendingParams& params = CrudeBlendingParams::ForScale(InstanceScale::Small));
};

} // namespace pipepye::workloads
