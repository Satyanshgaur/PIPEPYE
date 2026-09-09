India’s industrial optimization workloads depend on a small number of proprietary foreign solvers like CPLEX, Gurobi and Xpress, which introduce creating cost, licensing and dependency concerns. 
Existing open source solvers provide foundation, but fall short on large, sparse and difficult industrial optimization problems in complex MILPs. 
General-purpose solvers are also not specifically engineered or validated around relevant Indian industry.
There is also a lack of optimization engines that can intelligently determine how a workload should execute on CPU, GPU or a hybrid architecture for maximum throughput. 
Finally, there is limited publicly reproducible evidence showing how sovereign solvers perform on mathematical benchmarks and Indian industrial optimization models

We are building PIPEPYE to solve these gaps. 
