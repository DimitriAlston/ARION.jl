# This extension is meant to be used with SourceCodeMcCormick to overload
# EAGO's standard branch-and-bound algorithm, to enable the parallel
# evaluation of multiple B&B nodes.

abstract type ExtendGPU <: EAGO.ExtensionType end

mutable struct Problem
    name::String
    vars::Vector{Num}
    nvars::Int
    ncons::Int
    lvbs::Vector{Float64}
    uvbs::Vector{Float64}
    obj::Num
    eq::Vector{Num}
    leq::Vector{Num}
    geq::Vector{Num}
    sense::String
end

function Problem(name::String, vars::Vector{Num}, obj::Num, lvbs, uvbs; eq=Num[], leq=Num[], geq=Num[], sense="min")
    return Problem(name, vars, length(vars), length(eq)+length(leq)+length(geq), Float64.(lvbs), Float64.(uvbs), obj, eq, leq, geq, sense)
end

mutable struct LoadedProblem
    name::String
    vars::Vector{Num}
    nvars::Int
    ncons::Int
    lvbs::Vector{Float64}
    uvbs::Vector{Float64}
    obj_fun::Function
    eq_cons::Vector{Function}
    leq_cons::Vector{Function}
    geq_cons::Vector{Function}
    obj_sp::Vector{Bool}
    eq_sp::Vector{Vector{Bool}}
    leq_sp::Vector{Vector{Bool}}
    geq_sp::Vector{Vector{Bool}}
    sense::String
end

function LoadedProblem(prob::Problem; overwrite::Bool=false)
    return LoadedProblem(prob.name, prob.vars, prob.nvars, prob.ncons, prob.lvbs, prob.uvbs, 
        kgen(prob.obj, prob.vars, overwrite=overwrite),
        kgen.(prob.eq, Ref(prob.vars), overwrite=overwrite),
        kgen.(prob.leq, Ref(prob.vars), overwrite=overwrite),
        kgen.(prob.geq, Ref(prob.vars), overwrite=overwrite),
        [x in string.(pull_vars(prob.obj)) ? true : false for x in string.(prob.vars)],
        [[x in string.(pull_vars(prob.eq[i])) ? true : false for x in string.(prob.vars)] for i in eachindex(prob.eq)],
        [[x in string.(pull_vars(prob.leq[i])) ? true : false for x in string.(prob.vars)] for i in eachindex(prob.leq)],
        [[x in string.(pull_vars(prob.geq[i])) ? true : false for x in string.(prob.vars)] for i in eachindex(prob.geq)],
        prob.sense)
end

"""
$(TYPEDEF)

`GroupMethod` prepares a group of linearization points prior to running the B&B algorithm.
Within the lower-bounding routine, relaxations and subgradients are calculated at all
points within the set of linearization points, and up to `s+1` points are selected to
form LP constraints for each problem expression, where `s` is the number of non-zero
components in the expression's sparsity. If the `perform_scan` option is enabled, the
relaxation and subgradient results from the initial set of linearization points will
be used to identify a region of the node subdomain that likely contains a minimizer
for the objective function. The linearization points will then be scaled to match the
likely-minimizing region, relaxations and subgradients will be recalculated for these
scaled points, and the final set of points will be used for creating the relaxed LP.
Currently, this option does not affect relaxation calculations for problem constraints.

If the linearization method is chosen as `:Sobol`, `num_eval_points` linearization points
will be determined according to a Sobol sequence. If the linearization method is set
as `:nSimplex`, a total of `nvars+1` linearization points will be selected, which includes
the midpoint and `nvars` points selected using the nSimplex method described by:
Najman, J., Bongartz, D., and Mitsos, A. Linearization of McCormick relaxations and
hybridization with the auxiliary variable method. Journal of Global Optimization, 
80:731--356 (2021). https://doi.org/10.1007/s10898-020-00977-x

Note that unlike Najman et al., this method uses all points from the nSimplex rather
than removing points preemptively based on subgradient tightening, since points can
be evaluated quickly using SourceCodeMcCormick.jl.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct GroupMethod <: ExtendGPU
    "Linearization point selection method. Options are {`:Sobol`, `:nSimplex`}. The `:Sobol`
    method selects `num_eval_points` points from a Sobol sequence, and the `:nSimplex` method
    uses a variant of the nSimplex method described by:
    Najman, J., Bongartz, D., and Mitsos, A. Linearization of McCormick relaxations and
    hybridization with the auxiliary variable method. Journal of Global Optimization, 
    80:731--356 (2021). https://doi.org/10.1007/s10898-020-00977-x"
    linearization_method::Symbol
    "A SCMC/kgen-generated function taking arguments `[OUT, x...]` where each element of `x` is an
    `n-by-3`-dimensional `CuArray{Float64}` with columns corresponding to the evaluation point
    of interest, lower bound, and upper bound of a given variable. Each row of `x` relates to
    one point on which relaxations are desired, independent of the other rows/points. Results are
    saved to `OUT`, which is an `n-by-(4+2*n_vars)`-dimensional `CuArray{Float64}`, which holds
    in its columns, respectively, convex and concave relaxations, lower and upper bounds of an
    inclusion-monotonic interval extension, a subgradient of the convex relaxation, and a subgradient
    of the concave relaxation"
    obj_fun::Function
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    equality constraints in the original problem"
    eq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    less-than-or-equal-to constraints in the original problem"
    leq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    greater-than-or-equal-to constraints in the original problem"
    geq_cons::Vector{Function}
    "Sparsity pattern for the objective function"
    obj_sp::Vector{Bool}
    "Sparsity patterns for the equality constraints"
    eq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the LEQ constraints"
    leq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the GEQ constraints"
    geq_sp::Vector{Vector{Bool}}
    "The number of problem variables"
    n_vars::Int
    "The maximum number of nodes to consider simultaneously"
    max_parallel_nodes::Int64
    "Storage for inputs, outputs, buffers, and other intermediate fields needed during the PDLP algorithm"
    PDLP_data::BatchPDLP.PDLPData
    "Indicator to change lower solver behavior. If `use_dual_obj` = true, the dual
    objective value will be used instead of the primal objective value, for lower-bounding
    purposes (default = false)"
    use_dual_obj::Bool
    "The number of Sobol points to calculate as linearization points (default of 400). Only
    used if `linearization_method` is set to `:Sobol`"
    num_eval_points::Int32
    "The collection of all Sobol points being used"
    linearization_points::CuArray{Float64}
    "A vector of summary information used to compare linearization points. For constraints, this will
    be the absolute value of the convex/concave relaxations. For the objective function, this will be
    the sum of convex relaxation subgradient values squared"
    comparison_vector::CuArray{Float64}
    "A vector that stores the sum of S*(3^([var]-1)), where S is 0 if the subgradient in the [var]
    dimension is 0, 1 if the subgradient in the [var] dimension is positive, and 2 if the subgradient
    in the [var] dimension is negative. This gives guaranteed unique values for any combination of
    signs (or 0s) up to 39 variables (i.e., sum(2*3^(var-1) for var=1:39) is below typemax(Int64)). 
    Above 39 variables, Int64 will roll over into negative values, and it's possible (though very
    unlikely) to have a hash collision where two different subgradients give the same checksum value"
    subgradient_checksum::CuArray{Float64}
    "Storage for objective function/constraint evaluation results, for use immediately prior to adding
    information as constraints to the LP subproblems"
    result_storage::CuArray{Float64}
    "Storage for inputs to the objective function and constraint functions. For each variable, this stores
    the point to calculate and the bounds"
    input_storage::Vector{CuArray{Float64}}
    "Information about the evaluated points, not including bounds, for every variable. This also
    acts as a holder for PDLP solutions"
    eval_points::CuArray{Float64}
    "Objective values from PDLP"
    LP_objectives::CuArray{Float64}
    "Lower bound storage to hold calculated lower bounds for multiple nodes"
    lower_bound_storage::Vector{Float64} = Vector{Float64}()
    "Node storage to hold individual nodes outside of the main stack"
    node_storage::Vector{EAGO.NodeBB} = Vector{EAGO.NodeBB}()
    "An internal tracker of nodes in internal storage"
    node_len::Int = 0
    "Variable lower bounds to evaluate"
    all_lvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable upper bounds to evaluate"
    all_uvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable lower bounds for the nodes in storage stored on the GPU"
    lvbs_d::CuArray{Float64}
    "Variable upper bounds for the nodes in storage stored on the GPU"
    uvbs_d::CuArray{Float64}
    "Lower variable bounds for the multi-sobol procedure"
    lvbs_d_temp::CuArray{Float64}
    "Upper variable bounds for the multi-sobol procedure"
    uvbs_d_temp::CuArray{Float64}
    "Storage for LP solutions"
    LP_solutions::CuArray{Float64}
    "CPU storage for LP solutions"
    CPU_LP_solutions::Matrix{Float64}
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 100
    "Flag for performing the scan procedure for objective function relaxations"
    perform_scan::Bool = true
    "Number of nodes at which we switch from CPU (EAGO) to GPU (ARION). Default
    is to always use GPU"
    GPU_break_point::Int
    "Number of LPs at which we switch from CPU (EAGO subsolver) to GPU (BatchPDLP). 
    Default is to switch to BatchPDLP above 500 LPs"
    GPU_LP_break_point::Int
    "Number of GPU blocks available on the user's GPU"
    n_blocks::Int32
    "Timers for profiling"
    timers::Vector{Vector{Float64}}
    "Count of the number of nodes solved"
    nodes_solved::Vector{Int}
    "Count of the number of LPs solved"
    LPs_solved::Vector{Int}
    "Counts of the number of iterations within each call to PDLP"
    iterations::Vector{Matrix{Int32}}
    "Constraint matrix transfered from the GPU to the CPU, for CPU solving"
    cpu_constraint_mat::Matrix{Float64}
    "Right-hand side transfered from the GPU to the CPU, for CPU solving"
    cpu_rhs::Array{Float64}
    "Indicator for currently active constraints transfered from the GPU to the CPU, for CPU solving"
    cpu_active_constraint::Vector{Bool}
    "A vector of indicators for whether the CPU solver should be used on individual problems"
    cpu_solve_flag::Vector{Bool}
    "The CPU LP solver being used"
    cpu_solver
    "Flag to validate BatchPDLP results against the cpu_solver (primarily to
    be used as a diagnostic for BatchPDLP)"
    validate_vs_CPU::Bool
end

function GroupMethod(
    linearization_method::Symbol,
    problem::LoadedProblem;
    GPU_break_point::Int = 1,
    GPU_LP_break_point::Int = 500,
    max_parallel_nodes::Int = 10240,
    gc_freq::Int = 100, 
    PDLP_iteration_limit::Int = 10000000,
    num_eval_points::Int = 400,
    perform_scan::Bool = true,
    abs_tol::Float64 = 1E-8,
    rel_tol::Float64 = 1E-8,
    skip_hard_problems::Bool = true,
    validate_vs_CPU::Bool = false,
    use_dual_obj::Bool = false,
    nSimplex_r::Float64 = 0.725,
    nSimplex_deg::Float64 = 30.0,
    cpu_solver = GLPK.Optimizer(),
    )

    # Quick shortcut since the number of variables is used so often
    var_count = problem.nvars

    if linearization_method == :Sobol
        # Fill in linearization points for each variable using the Sobol.jl package
        linearization_points = Array{Float64}(undef, num_eval_points, var_count)
        s = SobolSeq(0.025 .* ones(var_count), 0.975 .* ones(var_count))

        # Always add the midpoint as a first point
        linearization_points[1,:] = round.(next!(s), digits=10)

        # Then skip the next (2^m - 1) points for the largest m where (2^m-1 <= n).
        # Note that Sobol.jl does this automatically when provided with "n", and
        # we use `num_eval_points - 1` because the first point is always the midpoint.
        # See Sobol.jl README for more details.
        skip(s, num_eval_points - 1)

        # Fill in linearization points with the next `num_eval_points-1` points
        for i in 2:num_eval_points
            linearization_points[i,:] .= round.(next!(s), digits=10)
        end
    elseif linearization_method == :nSimplex
        # Fill in linearization points for each variable using the nSimplex function
        linearization_points = vcat(zeros(Float64, 1, var_count), nSimplex(var_count, r=nSimplex_r, deg=nSimplex_deg))

        # Rescale linearization points to be between [0, 1]^n (from [-1, 1]^n)
        linearization_points .+= 1.0
        linearization_points ./= 2.0
        num_eval_points = size(linearization_points, 1)
    else
        error("Linearization method must be `:Sobol` or `:nSimplex`.")
    end

    # Determine the number of different types of constraints
    eq_len = length(problem.eq_cons)
    leq_len = length(problem.leq_cons)
    geq_len = length(problem.geq_cons)

    # Create the sparsity matrix. There will be one LP constraint for the lower bound,
    # and then for each expression (objective function or constraint), there will be
    # m+1 LP constraints, where m is the number of variables participating in that
    # expression. This way we use more points for more complex expressions, and
    # fewer points for less complex expressions.
    num_LP_constraints = 1 # lower bound
    num_LP_constraints += sum(problem.obj_sp) + 1
    for i = 1:eq_len
        num_LP_constraints += 2*(sum(problem.eq_sp[i]) + 1) # Double because equality constraints are split
    end
    for i = 1:leq_len
        num_LP_constraints += sum(problem.leq_sp[i]) + 1
    end
    for i = 1:geq_len
        num_LP_constraints += sum(problem.geq_sp[i]) + 1
    end
    sparsity = zeros(Bool, num_LP_constraints, var_count+1)

    # Fill in the sparsity matrix:
    # Lower bound, epigraph variable is always 1, other variables are 0
    next_row = 1
    sparsity[next_row, 1] = true
    next_row += 1

    # Objective function
    for _ = 1:(sum(problem.obj_sp) + 1)
        sparsity[next_row, :] .= [true, problem.obj_sp...]
        next_row += 1
    end

    # EQ constraints
    for i = 1:eq_len
        for _ = 1:2*(sum(problem.eq_sp[i]) + 1)
            sparsity[next_row, :] .= [false, problem.eq_sp[i]...]
            next_row += 1
        end
    end

    # LEQ constraints
    for i = 1:leq_len
        for _ = 1:sum(problem.leq_sp[i]) + 1
            sparsity[next_row, :] .= [false, problem.leq_sp[i]...]
            next_row += 1
        end
    end

    # GEQ constraints
    for i = 1:geq_len
        for _ = 1:sum(problem.geq_sp[i]) + 1
            sparsity[next_row, :] .= [false, problem.geq_sp[i]...]
            next_row += 1
        end
    end

    # Create a shorthand for the maximum number of relaxations that will need
    # to be evaluated at once
    max_eval_points = Int(max_parallel_nodes * num_eval_points)

    # Return the struct
    return GroupMethod(
                    linearization_method,
                    problem.obj_fun, 
                    problem.eq_cons, 
                    problem.leq_cons, 
                    problem.geq_cons,
                    problem.obj_sp,
                    problem.eq_sp,
                    problem.leq_sp,
                    problem.geq_sp,
                    problem.nvars, 
                    max_parallel_nodes, 
                    PDLPData(max_parallel_nodes, var_count+1, next_row-1,
                            iteration_limit=PDLP_iteration_limit,
                            sparsity=sparsity,
                            abs_tol=abs_tol, rel_tol=rel_tol,
                            skip_hard_problems=skip_hard_problems),
                    use_dual_obj,
                    Int32(num_eval_points),
                    cu(linearization_points),
                    CuArray{Float64}(undef, max_eval_points), # Comparison vector
                    CuArray{Float64}(undef, max_eval_points), # Subgradient checksum
                    CuArray{Float64}(undef, max_eval_points, 4+2*var_count), # Result storage
                    [CuArray{Float64}(undef, max_eval_points, 3) for j=1:var_count], # Input storage
                    CuArray{Float64}(undef, max_eval_points, var_count), # Evaluation points
                    CuArray{Float64}(undef, max_parallel_nodes, 1), # Objective results
                    Vector{Float64}(undef, max_parallel_nodes), # Space to transfer results to CPU (CPU)
                    Vector{NodeBB}(undef, max_parallel_nodes), # Node storage (CPU)
                    0, # Tracker for number of nodes
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All lower bounds (CPU)
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All upper bounds (CPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Node lower bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Node upper bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Temp lower bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Temp upper bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # (GPU) LP solution storage
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # (CPU) LP solution storage
                    gc_freq,
                    perform_scan,
                    GPU_break_point,
                    GPU_LP_break_point,
                    Int32(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)),
                    [Vector{Float64}() for i = 1:100], # Timers
                    Vector{Int}(undef, 0), # nodes solved
                    Vector{Int}(undef, 0), # LPs solved
                    Vector{Vector{Int32}}(undef, 0), # iterations
                    Matrix{Float64}(undef, size(sparsity, 1)*max_parallel_nodes, var_count+1), # CPU constraint matrix
                    Vector{Float64}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU RHS
                    Vector{Bool}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU active constraint list
                    Vector{Bool}(undef, max_parallel_nodes), # CPU solve flag
                    cpu_solver,
                    validate_vs_CPU,
    )
end

"""
$(TYPEDEF)

Similar to `GroupMethod`, but for systems with two GPUs. It is assumed that the
two GPUs are roughly equivalent.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct GroupMethod_MultiGPU <: ExtendGPU
    "Linearization point selection method. Options are {`:Sobol`, `:nSimplex`}. The `:Sobol`
    method selects `num_eval_points` points from a Sobol sequence, and the `:nSimplex` method
    uses a variant of the nSimplex method described by:
    Najman, J., Bongartz, D., and Mitsos, A. Linearization of McCormick relaxations and
    hybridization with the auxiliary variable method. Journal of Global Optimization, 
    80:731--356 (2021). https://doi.org/10.1007/s10898-020-00977-x"
    linearization_method::Symbol
    "A SCMC/kgen-generated function taking arguments `[OUT, x...]` where each element of `x` is an
    `n-by-3`-dimensional `CuArray{Float64}` with columns corresponding to the evaluation point
    of interest, lower bound, and upper bound of a given variable. Each row of `x` relates to
    one point on which relaxations are desired, independent of the other rows/points. Results are
    saved to `OUT`, which is an `n-by-(4+2*nvars)`-dimensional `CuArray{Float64}`, which holds
    in its columns, respectively, convex and concave relaxations, lower and upper bounds of an
    inclusion-monotonic interval extension, a subgradient of the convex relaxation, and a subgradient
    of the concave relaxation"
    obj_fun::Function
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    equality constraints in the original problem"
    eq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    less-than-or-equal-to constraints in the original problem"
    leq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    greater-than-or-equal-to constraints in the original problem"
    geq_cons::Vector{Function}
    "Sparsity pattern for the objective function"
    obj_sp::Vector{Bool}
    "Sparsity patterns for the equality constraints"
    eq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the LEQ constraints"
    leq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the GEQ constraints"
    geq_sp::Vector{Vector{Bool}}
    "The number of problem variables"
    n_vars::Int
    "The maximum number of nodes to consider simultaneously"
    max_parallel_nodes::Int64
    "Storage for inputs, outputs, buffers, and other intermediate fields needed during the PDLP algorithm"
    PDLP_data::Vector{PDLPData}
    "Indicator to change lower solver behavior. If `use_dual_obj` = true, the dual
    objective value will be used instead of the primal objective value, for lower-bounding
    purposes (default = false)"
    use_dual_obj::Bool
    "The number of Sobol points to calculate as linearization points (default of 400)"
    num_eval_points::Int32
    "The collection of all Sobol points being used"
    linearization_points::Vector{CuArray{Float64}}
    "A vector of summary information used to compare linearization points. For constraints, this will
    be the absolute value of the convex/concave relaxations. For the objective function, this will be
    the sum of convex relaxation subgradient values squared"
    comparison_vector::Vector{CuArray{Float64}}
    "A vector that stores the sum of S*(3^([var]-1)), where S is 0 if the subgradient in the [var]
    dimension is 0, 1 if the subgradient in the [var] dimension is positive, and 2 if the subgradient
    in the [var] dimension is negative. This gives guaranteed unique values for any combination of
    signs (or 0s) up to 39 variables (i.e., sum(2*3^(var-1) for var=1:39) is below typemax(Int64)). 
    Above 39 variables, Int64 will roll over into negative values, and it's possible (though very
    unlikely) to have a hash collision where two different subgradients give the same checksum value"
    subgradient_checksum::Vector{CuArray{Float64}}
    "Storage for objective function/constraint evaluation results, for use immediately prior to adding
    information as constraints to the LP subproblems"
    result_storage::Vector{CuArray{Float64}}
    "Storage for inputs to the objective function and constraint functions. For each variable, this stores
    the point to calculate and the bounds"
    input_storage::Vector{Vector{CuArray{Float64}}}
    "Information about the evaluated points, not including bounds, for every variable. This also
    acts as a holder for PDLP solutions"
    eval_points::Vector{CuArray{Float64}}
    "Objective values from PDLP"
    LP_objectives::Vector{CuArray{Float64}}
    "Lower bound storage to hold calculated lower bounds for multiple nodes"
    lower_bound_storage::Vector{Float64} = Vector{Float64}()
    "Node storage to hold individual nodes outside of the main stack"
    node_storage::Vector{EAGO.NodeBB} = Vector{EAGO.NodeBB}()
    "An internal tracker of nodes in internal storage"
    node_len::Int = 0
    "Variable lower bounds to evaluate"
    all_lvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable upper bounds to evaluate"
    all_uvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable lower bounds for the nodes in storage stored on the GPU"
    lvbs_d::Vector{CuArray{Float64}}
    "Variable upper bounds for the nodes in storage stored on the GPU"
    uvbs_d::Vector{CuArray{Float64}}
    "Lower variable bounds for the multi-sobol procedure"
    lvbs_d_temp::Vector{CuArray{Float64}}
    "Upper variable bounds for the multi-sobol procedure"
    uvbs_d_temp::Vector{CuArray{Float64}}
    "Storage for PDLP solutions"
    LP_solutions::Vector{CuArray{Float64}}
    "Storage for PDLP solutions"
    CPU_LP_solutions::Matrix{Float64}
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 100
    "Flag for performing the scan procedure for objective function relaxations"
    perform_scan::Bool = true
    "Number of LPs at which we switch from CPU to GPU"
    GPU_break_point::Int
    "Number of LPs at which we switch from CPU (EAGO subsolver) to GPU (BatchPDLP). 
    Default is to switch to BatchPDLP above 500 LPs"
    GPU_LP_break_point::Int
    "Number of GPU blocks available on the user's GPUs"
    n_blocks::Vector{Int32}
    "Timers for profiling"
    timers::Vector{Vector{Float64}}
    "Count of the number of nodes solved"
    nodes_solved::Vector{Int}
    "Count of the number of LPs solved"
    LPs_solved::Vector{Int}
    "Counts of the number of iterations within each call to PDLP"
    iterations::Vector{Matrix{Int32}}
    "Constraint matrix transfered from the GPU to the CPU, for CPU solving"
    cpu_constraint_mat::Matrix{Float64}
    "Right-hand side transfered from the GPU to the CPU, for CPU solving"
    cpu_rhs::Array{Float64}
    "Indicator for currently active constraints transfered from the GPU to the CPU, for CPU solving"
    cpu_active_constraint::Vector{Bool}
    "A vector of indicators for whether the CPU solver should be used on individual problems"
    cpu_solve_flag::Vector{Bool}
    "The CPU LP solver being used"
    cpu_solver
    "Flag to validate BatchPDLP results against the cpu_solver (primarily to
    be used as a diagnostic for BatchPDLP)"
    validate_vs_CPU::Bool
end

function GroupMethod_MultiGPU(
    linearization_method::Symbol,
    problem::LoadedProblem;
    GPU_break_point::Int = 1,
    GPU_LP_break_point::Int = 500,
    max_parallel_nodes::Int = 20480,
    gc_freq::Int = 15, 
    PDLP_iteration_limit::Int = 10000000,
    num_eval_points::Int = 400,
    perform_scan::Bool = true,
    abs_tol::Float64 = 1E-8,
    rel_tol::Float64 = 1E-8,
    skip_hard_problems::Bool = false,
    validate_vs_CPU::Bool = false,
    use_dual_obj::Bool = false,
    cpu_solver = GLPK.Optimizer(),
    )

    # Quick shortcut since the number of variables is used so often
    var_count = problem.nvars

    if linearization_method == :Sobol
        # Fill in linearization points for each variable using the Sobol.jl package
        linearization_points = Array{Float64}(undef, num_eval_points, var_count)
        s = SobolSeq(0.025 .* ones(var_count), 0.975 .* ones(var_count))

        # Always add the midpoint as a first point
        linearization_points[1,:] = round.(next!(s), digits=10)

        # Then skip the next (2^m - 1) points for the largest m where (2^m-1 <= n).
        # Note that Sobol.jl does this automatically when provided with "n", and
        # we use `num_eval_points - 1` because the first point is always the midpoint.
        # See Sobol.jl README for more details.
        skip(s, num_eval_points - 1)

        # Fill in linearization points with the next `num_eval_points-1` points
        for i in 2:num_eval_points
            linearization_points[i,:] .= round.(next!(s), digits=10)
        end
    elseif linearization_method == :nSimplex
        # Fill in linearization points for each variable using the nSimplex function
        linearization_points = vcat(zeros(Float64, 1, var_count), nSimplex(var_count, r=nSimplex_r, deg=nSimplex_deg))

        # Rescale linearization points to be between [0, 1]^n (from [-1, 1]^n)
        linearization_points .+= 1.0
        linearization_points ./= 2.0
        num_eval_points = size(linearization_points, 1)
    else
        error("Linearization method must be `:Sobol` or `:nSimplex`.")
    end

    # Calculate the sparsity for PDLP
    eq_len = length(problem.eq_cons)
    leq_len = length(problem.leq_cons)
    geq_len = length(problem.geq_cons)

    # Create the sparsity matrix. There will be one LP constraint for the lower bound,
    # and then for each expression (objective function or constraint), there will be
    # m+1 LP constraints, where m is the number of variables participating in that
    # expression. This way we use more points for more complex expressions, and
    # fewer points for less complex expressions.
    num_LP_constraints = 1 # lower bound
    num_LP_constraints += sum(problem.obj_sp) + 1
    for i = 1:eq_len
        num_LP_constraints += 2*(sum(problem.eq_sp[i]) + 1) # Double because equality constraints are split
    end
    for i = 1:leq_len
        num_LP_constraints += sum(problem.leq_sp[i]) + 1
    end
    for i = 1:geq_len
        num_LP_constraints += sum(problem.geq_sp[i]) + 1
    end
    sparsity = zeros(Bool, num_LP_constraints, var_count+1)

    # Fill in the sparsity matrix:
    # Lower bound, epigraph variable is always 1, other variables are 0
    next_row = 1
    sparsity[next_row, 1] = true
    next_row += 1

    # Objective function
    for _ = 1:(sum(problem.obj_sp) + 1)
        sparsity[next_row, :] .= [true, problem.obj_sp...]
        next_row += 1
    end

    # EQ constraints
    for i = 1:eq_len
        for _ = 1:2*(sum(problem.eq_sp[i]) + 1)
            sparsity[next_row, :] .= [false, problem.eq_sp[i]...]
            next_row += 1
        end
    end

    # LEQ constraints
    for i = 1:leq_len
        for _ = 1:sum(problem.leq_sp[i]) + 1
            sparsity[next_row, :] .= [false, problem.leq_sp[i]...]
            next_row += 1
        end
    end

    # GEQ constraints
    for i = 1:geq_len
        for _ = 1:sum(problem.geq_sp[i]) + 1
            sparsity[next_row, :] .= [false, problem.geq_sp[i]...]
            next_row += 1
        end
    end

    # Create a shorthand for the maximum number of relaxations that will need
    # to be evaluated at once
    max_eval_points = Int(max_parallel_nodes * num_eval_points)

    # Now we need to set up GPU information separately for each device.
    n_devices = length(devices())
    !isinteger(max_eval_points/n_devices) && error("max_parallel_nodes * num_eval_points must be divisible by $n_devices (the number of GPUs)")
    
    # Initialize storage locations
    PDLP_data = PDLPData[]
    linearization_points_cu = CuArray{Float64}[]
    comparison_vector = CuArray{Float64}[]
    subgradient_checksum = CuArray{Float64}[]
    result_storage = CuArray{Float64}[]
    input_storage = Vector{CuArray{Float64}}[]
    eval_points = CuArray{Float64}[]
    LP_objectives = CuArray{Float64}[]
    lvbs_d = CuArray{Float64}[]
    uvbs_d = CuArray{Float64}[]
    lvbs_d_temp = CuArray{Float64}[]
    uvbs_d_temp = CuArray{Float64}[]
    LP_solutions = CuArray{Float64}[]
    n_blocks = Vector{Int32}(undef, n_devices)

    # Fill storage locations for each device
    for (gpu, dev) in enumerate(devices())
        device!(dev)
        push!(PDLP_data, PDLPData(Int(max_parallel_nodes/n_devices), var_count+1, next_row-1,
                            iteration_limit=PDLP_iteration_limit,
                            sparsity=sparsity,
                            abs_tol=abs_tol, rel_tol=rel_tol,
                            skip_hard_problems=skip_hard_problems))
        push!(linearization_points_cu, cu(linearization_points))
        push!(comparison_vector, CuArray{Float64}(undef, Int(max_eval_points/n_devices)))
        push!(subgradient_checksum, CuArray{Float64}(undef, Int(max_eval_points/n_devices)))
        push!(result_storage, CuArray{Float64}(undef, Int(max_eval_points/n_devices), 4+2*var_count))
        push!(input_storage, [CuArray{Float64}(undef, Int(max_eval_points/n_devices), 3) for j=1:var_count])
        push!(eval_points, CuArray{Float64}(undef, Int(max_eval_points/n_devices), var_count))
        push!(LP_objectives, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), 1))
        push!(lvbs_d, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), var_count))
        push!(uvbs_d, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), var_count))
        push!(lvbs_d_temp, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), var_count))
        push!(uvbs_d_temp, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), var_count))
        push!(LP_solutions, CuArray{Float64}(undef, Int(max_parallel_nodes/n_devices), var_count))
        n_blocks[gpu] = Int32(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
    end

    return GroupMethod_MultiGPU(
                    linearization_method,
                    problem.obj_fun, 
                    problem.eq_cons, 
                    problem.leq_cons, 
                    problem.geq_cons,
                    problem.obj_sp,
                    problem.eq_sp,
                    problem.leq_sp,
                    problem.geq_sp,
                    problem.nvars, 
                    max_parallel_nodes, 
                    PDLP_data,
                    use_dual_obj,
                    Int32(num_eval_points),
                    linearization_points_cu,
                    comparison_vector, # Comparison vector
                    subgradient_checksum, # Subgradient checksum
                    result_storage, # Result storage
                    input_storage, # Input storage
                    eval_points, # Evaluation points
                    LP_objectives, # Objective results
                    Vector{Float64}(undef, max_parallel_nodes), # Space to transfer lower bounds to CPU (CPU)
                    Vector{EAGO.NodeBB}(undef, max_parallel_nodes), # Node storage (CPU)
                    0, # Tracker for number of nodes
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All lower bounds (CPU)
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All upper bounds (CPU)
                    lvbs_d, # Node lower bounds (GPU)
                    uvbs_d, # Node upper bounds (GPU)
                    lvbs_d_temp, # Temp lower bounds (GPU)
                    uvbs_d_temp, # Temp upper bounds (GPU)
                    LP_solutions, # BatchPDLP solution storage
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # CPU solution storage
                    gc_freq,
                    perform_scan,
                    GPU_break_point,
                    GPU_LP_break_point,
                    n_blocks,
                    [Vector{Float64}() for i = 1:100], # Timers
                    Vector{Int}(undef, 0), # nodes solved
                    Vector{Int}(undef, 0), # LPs solved
                    Vector{Vector{Int32}}(undef, 0), # iterations
                    Matrix{Float64}(undef, size(sparsity, 1)*max_parallel_nodes, var_count+1), # CPU constraint matrix
                    Vector{Float64}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU RHS
                    Vector{Bool}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU active constraint list
                    Vector{Bool}(undef, max_parallel_nodes), # CPU solve flag
                    cpu_solver,
                    validate_vs_CPU,
    )
end

function nSimplex(n::Int; r::Float64=0.725, deg::Real=-30.0)
    if n > 1
        # This function follows Algorithm 2 from Najman2021 (note that
        # the order of indices is flipped here relative to the paper)
        # Line 4
        V = zeros(Float64, n+1, n)

        # Line 5
        V[1,1] = r

        # Line 6
        for i = 2:n+1
            V[i,1] = -r/n
        end

        # Lines 7--16
        for i = 1:n-1
            t = 0.0
            for j = 1:i
                t += V[i+1,j]^2
            end
            V[i+1,i+1] = sqrt(r^2 - t)
            for j = i+2:n+1
                V[j,i+1] = -(V[i+1,i+1]/(n-i))
            end
        end

        # Now, we rotate by 30 degrees counterclockwise, from the unit vector pointing
        # in the first dimension to the unit vector pointing in every dimension but the
        # first
        rotation = zeros(Float64, n, n)
        n1 = zeros(Float64, n)
        n2 = zeros(Float64, n)
        n1[1] = 1.0
        n2[2:end] .= 1/sqrt(n-1)

        for i = 1:n
            rotation[i,i] = 1.0
        end

        # See the following for rotations between two n-dimensional vectors: 
        # https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
        # rotation += (n2*transpose(n1) - n1*transpose(n2))*sind(deg) + (n1*transpose(n1) + n2*transpose(n2))*(cosd(deg) - 1)
        temp1 = similar(rotation)
        temp2 = similar(rotation)
        mul!(temp1, n2, n1')
        mul!(temp2, n1, n2')
        temp1 .-= temp2
        temp1 .*= sind(deg)
        mul!(temp2, n1, n1')
        temp2 .+= n2 * n2'
        temp2 .*= (cosd(deg) - 1)
        rotation .+= temp1 .+ temp2

        # Perform the rotation
        temp1 = similar(V[1,:])
        for i = 1:n+1
            mul!(temp1, rotation, V[i,:])
            V[i,:] .= temp1
        end

        # If there are any near-zeros, just set them to 0
        V[abs.(V) .<= 1E-10] .= 0.0

        return V
    elseif n==1
        # If n==1, we use 3 specific points
        return [-1/3; 0.0; 1/3]
    else
        error("Simplex dimensionality must be a non-negative integer")
    end
end


"""
$(TYPEDEF)

`KelleyMethod` uses a variant of Kelley's algorithm to iteratively improve relaxations
of the main problem on each subdomain. This method is similar to what is done in the
base version of EAGO, but multiple nodes are evaluated at once using GPU resources.
The chosen CPU-based LP solver will be used in place of BatchPDLP if fewer than
`GPU_LP_break_point` LPs are being solved, which is the case if fewer than 
`GPU_LP_break_point` nodes are being evaluated, or if Kelley's algorithm terminates
on enough nodes that the remaining number of LPs is below this value.

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct KelleyMethod <: ExtendGPU
    "A SCMC/kgen-generated function taking arguments `[OUT, x...]` where each element of `x` is an
    `n-by-3`-dimensional `CuArray{Float64}` with columns corresponding to the evaluation point
    of interest, lower bound, and upper bound of a given variable. Each row of `x` relates to
    one point on which relaxations are desired, independent of the other rows/points. Results are
    saved to `OUT`, which is an `n-by-(4+2*n_vars)`-dimensional `CuArray{Float64}`, which holds
    in its columns, respectively, convex and concave relaxations, lower and upper bounds of an
    inclusion-monotonic interval extension, a subgradient of the convex relaxation, and a subgradient
    of the concave relaxation"
    obj_fun::Function
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    equality constraints in the original problem"
    eq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    less-than-or-equal-to constraints in the original problem"
    leq_cons::Vector{Function}
    "Similar to the `obj_fun` field, except this is a vector of kgen-generated functions representing
    greater-than-or-equal-to constraints in the original problem"
    geq_cons::Vector{Function}
    "Sparsity pattern for the objective function"
    obj_sp::Vector{Bool}
    "Sparsity patterns for the equality constraints"
    eq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the LEQ constraints"
    leq_sp::Vector{Vector{Bool}}
    "Sparsity patterns for the GEQ constraints"
    geq_sp::Vector{Vector{Bool}}
    "The number of problem variables"
    n_vars::Int
    "The maximum number of nodes to consider simultaneously"
    max_parallel_nodes::Int64
    "Storage for inputs, outputs, buffers, and other intermediate fields needed during the PDLP algorithm"
    PDLP_data::BatchPDLP.PDLPData
    "Indicator to change lower solver behavior. If `use_dual_obj` = true, the dual
    objective value will be used instead of the primal objective value, for lower-bounding
    purposes (default = false)"
    use_dual_obj::Bool
    "Kelley's algorithm generally continues until either a predetermined maximum number of cuts (this
    field) have been added, or there is no progress in improving the lower bound. `KelleyMethod` will
    halt individual subproblems separately if they stop making progress, and all subproblems will stop
    at at most `max_cuts` iterations regardless of progress"
    max_cuts::Int64
    "Storage for objective function/constraint evaluation results, for use immediately prior to adding
    information as constraints to the LP subproblems"
    result_storage::CuArray{Float64}
    "Storage for inputs to the objective function and constraint functions. For each variable, this stores
    the point to calculate and the bounds"
    input_storage::Vector{CuArray{Float64}}
    "Information about the evaluated points, not including bounds, for every variable. This also
    acts as a holder for PDLP solutions"
    eval_points::CuArray{Float64}
    "Objective values from PDLP"
    LP_objectives::CuArray{Float64}
    "Previous iteration objective values from PDLP, used to determine termination of Kelley's algorithm"
    previous_LP_objectives::CuArray{Float64}
    "Absolute tolerance to stop iterations of Kelley's algorithm (default = 1E-6)"
    cut_tolerance_abs::Float64 = 1E-6
    "Relative tolerance to stop iterations of Kelley's algorithm (default = 1E-3)"
    cut_tolerance_rel::Float64 = 1E-3
    "Lower bound storage to hold calculated lower bounds for multiple nodes"
    lower_bound_storage::Vector{Float64} = Vector{Float64}()
    "Node storage to hold individual nodes outside of the main stack"
    node_storage::Vector{EAGO.NodeBB} = Vector{EAGO.NodeBB}()
    "An internal tracker of nodes in internal storage"
    node_len::Int = 0
    "Variable lower bounds to evaluate"
    all_lvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable upper bounds to evaluate"
    all_uvbs::Matrix{Float64} = Matrix{Float64}()
    "Variable lower bounds for the nodes in storage stored on the GPU"
    lvbs_d::CuArray{Float64}
    "Variable upper bounds for the nodes in storage stored on the GPU"
    uvbs_d::CuArray{Float64}
    "Storage for LP solutions"
    LP_solutions::CuArray{Float64}
    "CPU storage for LP solutions"
    CPU_LP_solutions::Matrix{Float64}
    "Frequency of garbage collection (number of iterations)"
    gc_freq::Int = 100
    "Number of nodes at which we switch from CPU (EAGO) to GPU (ARION). Default
    is to always use GPU"
    GPU_break_point::Int
    "Number of LPs at which we switch from CPU (EAGO subsolver) to GPU (BatchPDLP). 
    Default is to switch to BatchPDLP above 500 LPs"
    GPU_LP_break_point::Int
    "Number of GPU blocks available on the user's GPU"
    n_blocks::Int32
    "Timers for profiling"
    timers::Vector{Vector{Float64}}
    "Count of the number of nodes solved"
    nodes_solved::Vector{Int}
    "Count of the number of LPs solved"
    LPs_solved::Vector{Int}
    "Counts of the number of iterations within each call to PDLP"
    iterations::Vector{Matrix{Int32}}
    "Constraint matrix transfered from the GPU to the CPU, for CPU solving"
    cpu_constraint_mat::Matrix{Float64}
    "Right-hand side transfered from the GPU to the CPU, for CPU solving"
    cpu_rhs::Array{Float64}
    "Indicator for currently active constraints transfered from the GPU to the CPU, for CPU solving"
    cpu_active_constraint::Vector{Bool}
    "A vector of indicators for whether the CPU solver should be used on individual problems"
    cpu_solve_flag::Vector{Bool}
    "The CPU LP solver being used"
    cpu_solver
end

function KelleyMethod(
    problem::LoadedProblem;
    GPU_break_point::Int = 1,
    GPU_LP_break_point::Int = 500,
    max_parallel_nodes::Int = 10240,
    max_cuts::Int = 8, 
    gc_freq::Int = 100, 
    PDLP_iteration_limit::Int = 10000000,
    abs_tol::Float64 = 1E-8,
    rel_tol::Float64 = 1E-8,
    cut_tolerance_abs::Float64 = 1E-6,
    cut_tolerance_rel::Float64 = 1E-3,
    skip_hard_problems::Bool = true,
    use_dual_obj::Bool = false,
    cpu_solver = GLPK.Optimizer(),
    )

    # Quick shortcut since the number of variables is used so often
    var_count = problem.nvars

    # Determine the number of different types of constraints
    eq_len = length(problem.eq_cons)
    leq_len = length(problem.leq_cons)
    geq_len = length(problem.geq_cons)

    # Create the sparsity matrix. There will be one LP constraint for the lower bound,
    # and then for each cut in Kelley's algorithm, there will be one constraint for the
    # objective function, two constraints for each EQ constraint, and one constraint
    # for each LE or GE constraint. 
    constraints_per_cut = 1 + 2*eq_len + leq_len + geq_len
    sparsity = zeros(Bool, 1 + constraints_per_cut * max_cuts, var_count+1)

    # Fill in the sparsity matrix:
    # Lower bound, epigraph variable is always 1, other variables are 0
    next_row = 1
    sparsity[next_row, 1] = true
    next_row += 1

    # Objective function
    for cut = 0:max_cuts-1
        sparsity[next_row + cut*constraints_per_cut, :] .= [true, problem.obj_sp...]
    end
    next_row += 1

    # EQ constraints
    for i = 1:eq_len
        for cut = 0:max_cuts-1
            sparsity[next_row + cut*constraints_per_cut, :] .= [false, problem.eq_sp[i]...]
        end
        next_row += 1
        for cut = 0:max_cuts-1
            sparsity[next_row + cut*constraints_per_cut, :] .= [false, problem.eq_sp[i]...]
        end
        next_row += 1
    end

    # LEQ constraints
    for i = 1:leq_len
        for cut = 0:max_cuts-1
            sparsity[next_row + cut*constraints_per_cut, :] .= [false, problem.leq_sp[i]...]
        end
        next_row += 1
    end

    # GEQ constraints
    for i = 1:geq_len
        for cut = 0:max_cuts-1
            sparsity[next_row + cut*constraints_per_cut, :] .= [false, problem.geq_sp[i]...]
        end
        next_row += 1
    end

    # Return the struct
    return KelleyMethod(
                    problem.obj_fun, 
                    problem.eq_cons, 
                    problem.leq_cons, 
                    problem.geq_cons,
                    problem.obj_sp,
                    problem.eq_sp,
                    problem.leq_sp,
                    problem.geq_sp,
                    problem.nvars, 
                    max_parallel_nodes, 
                    PDLPData(max_parallel_nodes, var_count+1, size(sparsity, 1),
                            iteration_limit=PDLP_iteration_limit,
                            sparsity=sparsity,
                            abs_tol=abs_tol, rel_tol=rel_tol,
                            skip_hard_problems=skip_hard_problems),
                    use_dual_obj,
                    max_cuts,
                    CuArray{Float64}(undef, max_parallel_nodes, 4+2*var_count), # Result storage
                    [CuArray{Float64}(undef, max_parallel_nodes, 3) for j=1:var_count], # Input storage
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Evaluation points
                    CuArray{Float64}(undef, max_parallel_nodes, 1), # Objective results
                    CuArray{Float64}(undef, max_parallel_nodes, 1), # Previous objective results
                    cut_tolerance_abs,
                    cut_tolerance_rel,
                    Vector{Float64}(undef, max_parallel_nodes), # Space to transfer results to CPU (CPU)
                    Vector{NodeBB}(undef, max_parallel_nodes), # Node storage (CPU)
                    0, # Tracker for number of nodes
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All lower bounds (CPU)
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # All upper bounds (CPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Node lower bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # Node upper bounds (GPU)
                    CuArray{Float64}(undef, max_parallel_nodes, var_count), # (GPU) LP solution storage
                    Matrix{Float64}(undef, max_parallel_nodes, var_count), # (CPU) LP solution storage
                    gc_freq,
                    GPU_break_point,
                    GPU_LP_break_point,
                    Int32(CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)),
                    [Vector{Float64}() for i = 1:100], # Timers
                    Vector{Int}(undef, 0), # nodes solved
                    Vector{Int}(undef, 0), # LPs solved
                    Vector{Vector{Int32}}(undef, 0), # iterations
                    Matrix{Float64}(undef, size(sparsity, 1)*max_parallel_nodes, var_count+1), # CPU constraint matrix
                    Vector{Float64}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU RHS
                    Vector{Bool}(undef, size(sparsity, 1)*max_parallel_nodes), # CPU active constraint list
                    Vector{Bool}(undef, max_parallel_nodes), # CPU solve flag
                    cpu_solver,
    )
end