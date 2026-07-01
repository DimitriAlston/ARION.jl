# ARION.jl

ARION (**A**ccelerated and **R**easily-scalable **I**ntegrated CPU-GPU **O**ptimization for **N**onlinear programs) is an extension of the [EAGO solver](https://github.com/PSORLab/EAGO.jl) [1] that allows NVIDIA GPUs to be used to solve deterministic global optimization problems. This is accomplished in ARION by calculating lower bounds for multiple branch-and-bound (B&B) nodes simultaneously using GPU-accelerated McCormick relaxations and LP solving, with other processes (upper-bounding, fathoming, branching, etc.) occurring using EAGO's default serial CPU-based approaches. To enable GPU use in the lower problem, McCormick relaxations are calculated using [SourceCodeMcCormick (SCMC)](https://github.com/PSORLab/SourceCodeMcCormick.jl) [2,3], and relaxed LPs are solved using both a CPU-based LP subsolver ([GLPK](https://github.com/jump-dev/GLPK.jl) [4] by default) and a GPU-based batched LP solver ([BatchPDLP](https://github.com/PSORLab/BatchPDLP.jl)) [5].

Note that currently, ARION is designed to be used with variables defined using [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) [6], rather than [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl)/[JuMP](https://github.com/jump-dev/JuMP.jl) [7,8] variables. This difference is primarily for historical reasons and is expected to change in future versions of ARION.

## Basic Functionality

ARION is designed as an extension of EAGO, and consequently, optimization problems must be formulated using JuMP syntax with the EAGO optimizer called as the main optimizer and the selected ARION struct introduced as an extension of EAGO. As will be shown in the following example, to set up the extension, the optimization problem must be described a second time using Symbolics.jl-type variables (using the same order of variables and constraints), with all relevant variables, bounds, and constraints stored in a `Problem` struct. Because the underlying SCMC method relies on source code generation to create kernels that will be called during the B&B algorithm, this `Problem` struct must be converted into a `LoadedProblem`, which causes all necessary kernels to be generated. This `LoadedProblem` can then be passed to a `GroupMethod` or `KelleyMethod` struct, depending on which lower-bounding approach is desired, which is the extension that is passed to EAGO.

This example shows the above workflow as applied to the scalable Alpine02 global optimization problem:
```julia
using JuMP, EAGO, GLPK, SourceCodeMcCormick, CUDA, ARION

# Solving the Alpine02 problem with dimension 5:
# min -(prod(sqrt(x[i])*sin(x[i]) for i = 1:5)
# s.t. 0 <= x <= 10

# Prepare the ARION struct
SourceCodeMcCormick.@variables x[1:5]
vars = [x[1], x[2], x[3], x[4], x[5]]
obj = -prod(sqrt(x[i])*sin(x[i]) for i = 1:5)
lvbs = zeros(Float64, 5)
uvbs = 10.0 .* ones(Float64, 5)
Alpine02 = Problem("Alpine02", vars, obj, lvbs, uvbs)
LoadedAlpine02 = LoadedProblem(Alpine02)

# Set up the EAGO extension using the `GroupMethod` struct. 
extension = GroupMethod(
    :Sobol,                         # Specifies that linearization points are chosen from a Sobol sequence
    LoadedAlpine02,                 # The LoadedProblem struct from earlier
    max_parallel_nodes = 2048,      # The maximum number of B&B nodes to address per iteration
    num_eval_points = 400,          # The number of Sobol points to evaluate simultaneously per node
    PDLP_iteration_limit = 10000000,# The iteration limit for BatchPDLP
    perform_scan = true,            # Whether to perform the "MultiSobol" scan procedure
    abs_tol = 1E-6,                 # Absolute tolerance for BatchPDLP
    rel_tol = 1E-6,                 # Relative tolerance for BatchPDLP
    skip_hard_problems = true,      # Flag for BatchPDLP to solve "easy" LPs but pass "hard" LPs to the CPU subsolver
    cpu_solver = GLPK.Optimizer(),  # CPU-based LP subsolver to use
    );

# Set up the JuMP model with EAGO
model = JuMP.Model()
JuMP.set_optimizer(model, () -> EAGO.Optimizer(SubSolvers(; t = ext)))
JuMP.set_attribute(model, "enable_optimize_hook", true)        # Specify that the ARION extension should be used (required)
JuMP.set_attribute(model, "branch_variable", fill(true, 5))    # Specify that all variables should be branched on (currently required to do this)
JuMP.set_attribute(model, "branch_cvx_factor", 1.0)            # Helpful EAGO setting for this example
JuMP.set_attribute(model, "node_limit", 1000) # Enough to use BatchPDLP at some point, unless it solves too quickly (which is fine too)
lvbs = zeros(Float64, 5)
uvbs = 10.0 .* ones(Float64, 5)
@variable(model, lvbs[i] <= x[i=1:5] <= uvbs[i])
@objective(model, Min, -prod(sqrt(x[i])*sin(x[i]) for i = 1:5))

# Solve the problem
JuMP.optimize!(model)
```

In this example, the `GroupMethod` struct was used. When this struct is utilized, ARION determines a set of linearization points prior to starting the main B&B algorithm and then scales those points to each node's subdomain during the lower-bounding routine. If `:Sobol` is selected as the first argument, a total of `num_eval_points` linearization points are calculated from a Sobol sequence. The alternative for the first argument is `:nSimplex`, which instead calculates points using an n-simplex as described by Najman et al. [9]. In this case, the number of points is determined by the problem dimensionality, and the `num_eval_points` input is ignored. For both `:Sobol` and `:nSimplex`, if `perform_scan` is `false`, the relaxations and relaxation subgradients obtained for the set of linearization points on each B&B node will be used to generate constraints for a relaxed LP, which will be solved to obtain a lower bound for that node. Not all linearization points are added as LP constraints; for each expression in the NLP, up to `n+1` points are selected heuristically, where `n` is the number of variables participating in that expression. If `perform_scan` is `true`, the initial set of relaxations and relaxation subgradients are not used to create a relaxed LP, but are instead scanned to determine a subdomain within each node that likely contains a minimum of the relaxation. The linearization points are then rescaled to this subdomain, relaxations and relaxation subgradients are calculated for this new set of points, and up to `n+1` of these new linearization points are used to formulate a relaxed LP.

As an alternative to `GroupMethod`, `KelleyMethod` may be used as follows:
```julia
# Set up the EAGO extension using the `KelleyMethod` struct. 
extension = KelleyMethod(
    LoadedAlpine02,                 # The LoadedProblem struct from earlier
    max_parallel_nodes = 2048,      # The maximum number of B&B nodes to address per iteration
    max_cuts = 10,                  # The maximum number of Kelley's algorithm cuts to make per node (default = 8)
    abs_tol = 1E-6,                 # Absolute tolerance for BatchPDLP
    rel_tol = 1E-6,                 # Relative tolerance for BatchPDLP
    skip_hard_problems = true,      # Flag for BatchPDLP to solve "easy" LPs but pass "hard" LPs to the CPU subsolver
    cpu_solver = GLPK.Optimizer(),  # CPU-based LP subsolver to use
    );
```

`KelleyMethod` addresses the lower-bounding problem by applying an adapted version of Kelley's cutting plane algorithm [10] to each B&B node. This is the lower-bounding approach used in the CPU-based EAGO. This approach is expected to be less favorable than `GroupMethod` in ARION because the number of iterations of Kelley's algorithm is node-dependent. Specifically, this adapted version of Kelley's algorithm terminates early if the most recent cut did not result in an improvement of the LP objective value. Since ARION is designed to address many B&B nodes simultaneously, this generally results in some nodes terminating before others, which decreases the workload being sent to the GPU, which is relatively inefficient. Additionally, when Kelley's method is performed in EAGO, because each B&B node is processed individually, the LP subsolver may warm-start from the solution of the previous LP. In ARION, such warm-starting is currently not possible: when the GPU-based BatchPDLP is used, although the order of LPs is maintained as cuts are added, BatchPDLP does not utilize warm-starting. When the CPU-based LP subsolver is used instead, it is applied separately to each B&B node in sequence before the next cuts are added, which resets the subsolver between each LP and prevents automatic warm-starting. In contrast to the relative weaknesses of `KelleyMethod`, `GroupMethod` is designed to make better use of GPU resources by determining all linearization points a priori, which allows for a more consistent GPU workload.

## LP solving

Relaxed LPs created during the lower-bounding problem are solved using a combination of the GPU-based BatchPDLP and a CPU-based subsolver (GLPK by default). BatchPDLP shows strong performance when a large number of LPs are being solved together, but because only a small amount of GPU resources are dedicated to each LP, its performance is expected to be comparatively poor when only a small number of LPs are being solved. This is expected to be the case early in the B&B process as, e.g. the root node only results in one LP, its two descendents result in two LPs, their descendents result in four LPs, and so on. By default, BatchPDLP is used when more than 500 LPs are being solved, and the CPU subsolver is used otherwise. This can be adjusted using the `GPU_LP_break_point::Int` keyword argument in `KelleyMethod` and `GroupMethod`. Additionally, when using `KelleyMethod`, when more than 500 B&B nodes are being addressed, it is likely that after some number of Kelley's algorithm iterations, fewer than 500 B&B nodes will require further processing. In this case as well, ARION will switch to the CPU subsolver. 

As a final note, a key result from [5] is that the performance of BatchPDLP is strongly dependent on the speed of the slowest problems within each LP batch. I.e., some LPs are solved extremely quickly, but the runtime is effectively dominated by small numbers of extremely slow LPs. ARION and BatchPDLP can exploit this behavior using the `skip_hard_problems` flag. Setting this to `true` causes BatchPDLP to track the number of iterations required for each LP in a batch and preemptively terminate later LPs if their iteration counts are too high above the mean iteration count of other solved LPs. These unsolved LPs are then passed individually to the CPU LP subsolver to obtain B&B node lower bounds. From early testing, using BatchPDLP with this setting resulted in faster performance than both exclusively using BatchPDLP and exclusively using a CPU-based subsolver. 



## Citing ARION
Please cite the following paper when using ARION.jl. In plain text form this is:
```
Gottlieb, R.X. and Stuber, M.D. Integration of GPU-Parallel McCormick Relaxations and Batched PDLP for Accelerated Deterministic Global Optimization. Under Review (2026).
```
A BibTeX entry is given below:
```bibtex
@Article{
  author    = {Robert X. Gottlieb and Matthew D. Stuber},
  journal   = {Under Review},
  title     = {Integration of {GPU}-Parallel {McCormick} Relaxations and Batched {PDLP} for Accelerated Deterministic Global Optimization},
  year      = {2026},
}
```


## References
1. Wilhelm, M.E. and Stuber, M.D. EAGO.jl: Easy Advanced Global Optimization in Julia. Optimization Methods & Software 37 (2022), pp. 425-450. 
2. Gottlieb, R.X., Xu, P., and Stuber, M.D. Automatic source code generation for deterministic global optimization with parallel architectures, Optimization Methods and Software 41 (2026), pp. 308–346.
3. Gottlieb, R.X. and Stuber, M.D. Automatic generation of GPU kernels for evaluators of McCormick-based relaxations and subgradients, Under Revision (2025).
4. Makhorin, A. GLPK (GNU linear programming kit). http://www.gnu.org/s/glpk/glpk.html (2008).
5. Gottlieb, R.X., Alston, D., and Stuber, M.D, Re-architected PDLP for batch parallelization of linear programs, Under Revision (2026).
6. Gowda, S., Ma, Y., Cheli, A., Gwóźdź, M., Shah, V.B., Edelman, A., and Rackauckas, C. High-performance symbolic-numerics via multiple dispatch. ACM Communications in Computer Algebra 55(3) (2021), pp. 92-96.
7. Legat, B., Dowson, O., Garcia, J.D., and Lubin, M. MathOptInterface: A Data Structure for Mathematical Optimization Problems, INFORMS Journal on Computing 34(2) (2022), pp. 672-689.
8. Dunning, I., Huchette, J., and Lubin, M. JuMP: A Modeling Language for Mathematical Optimization, SIAM Review 59(2) (2017), pp. 295-320.
9. Najman, J., Bongartz, D., and Mitsos, A. Linearization of McCormick relaxations and hybridization with the auxiliary variable method, Journal of Global Optimization 80 (2021), pp. 731-756.
10. Kelley, J.E. Jr. The cutting-plane method for solving convex programs, Journal of the Society for Industrial and Applied Mathematics 8 (1960), pp. 703-712.