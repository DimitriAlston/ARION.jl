# Overload EAGO's B&B algorithm through optimize_hook!
function EAGO.optimize_hook!(::ExtendGPU, m::Optimizer)    
    m._global_optimizer._parameters.branch_variable = m._parameters.branch_variable
    EAGO.initial_parse!(m)

    optimize_gpu!(m)
end

# Analogous to `optimize!()` 
# @ EAGO/src/eago_optimizer/optimize/optimize_nonconvex.jl:492
function optimize_gpu!(m::Optimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType}
    solve_gpu!(m._global_optimizer)
    EAGO.unpack_global_solution!(m)
    return
end

# Analogous to `global_solve()
# @ EAGO/src/eago_optimizer/optimize/optimize_nonconvex.jl:373
function solve_gpu!(ext::T, m::EAGO.GlobalOptimizer) where T <: ExtendGPU
    # Turn off garbage collection
    # GC.enable(false)

    # Set node count to 1
    m._node_count = 1

    # Prepare to run branch-and-bound
    EAGO.parse_global!(m)
    EAGO.presolve_global!(m)
    EAGO.print_preamble!(m)

    # Flag for whether the epigraph variable was created...
    # _epigraph_occurred

    # Reset tracking/profiling counts and timers
    gpu_iteration_count = 0
    nodes_explored = 0
    empty!.(ext.timers)
    push!.(ext.timers, 0.0)
    ext.iterations = Vector{Matrix{Int32}}(undef, 0)
    empty!(ext.LPs_solved)
    this_time = 0.0

    # Run branch and bound; terminate when the stack is empty or when some
    # tolerance or limit is hit
    while !EAGO.termination_check(m)
        # Update iteration counter
        m._iteration_count += 1

        # Perform fathoming
        EAGO.fathom!(m)

        # Based on the current node count, perform either a CPU or GPU iteration
        if m._node_count < ext.GPU_break_point
            ########################################################################
            ############################ CPU iteration #############################
            ########################################################################

            # Iterate the number of nodes explored
            nodes_explored += 1

            # Pick a node and temporarily remove it from the stack
            EAGO.node_selection!(m)
            EAGO.print_node!(m)

            # Perform prepocessing and log the time
            this_time = @elapsed EAGO.preprocess!(m)
            m._last_preprocess_time += this_time
            ext.timers[12] .+= this_time

            # Continue if the node has not been proven infeasible
            if m._preprocess_feasibility

                # Solve the lower bounding problem and log the time
                this_time = @elapsed EAGO.lower_problem!(m)
                m._last_lower_problem_time += this_time
                ext.timers[9] .+= this_time
                EAGO.print_results!(m, true)

                # Continue if lower problem is not infeasible and problem
                # problem has not yet converged
                if m._lower_feasibility && !EAGO.convergence_check(m)

                    # Solve the upper bounding problem and log the time
                    this_time = @elapsed EAGO.upper_problem!(m)
                    m._last_upper_problem_time += this_time
                    ext.timers[13] .+= this_time
                    EAGO.print_results!(m, false)

                    # Update the global upper bound if necessary
                    EAGO.store_candidate_solution!(m)

                    # Perform post processing and log the time
                    this_time = @elapsed EAGO.postprocess!(m)
                    m._last_postprocessing_time += this_time
                    ext.timers[14] .+= this_time

                    # Continue if the node is not infeasible after postprocessing
                    if m._postprocess_feasibility

                        # If the node is to be repeatedly evaluated, add it back
                        # onto the stack. If not, branch on the node and add the
                        # two new nodes to the stack
                        this_time = @elapsed EAGO.repeat_check(m) ? EAGO.single_storage!(m) : EAGO.branch_node!(m)
                        ext.timers[15] .+= this_time
                    end
                end
            else
                # "Disqualify" the node (TODO: Not strictly necessary, since it's
                # not added back to the stack? This node will simply be overwritten
                # in `node_selection!`)
                m._lower_objective_value = Inf
                m._lower_feasibility = false
                m._upper_feasibility = false
            end

            # Update the global lower bound if necessary
            EAGO.set_global_lower_bound!(m)

            # Adjust algorithm run times
            m._run_time = time() - m._start_time
            m._time_left = m._parameters.time_limit - m._run_time

            # Log and print information as needed
            EAGO.log_iteration!(m)
            EAGO.print_iteration!(m, false)
        else
            ########################################################################
            ############################ GPU iteration #############################
            ########################################################################
            # Garbage collect every gc_freq iterations
            # if mod(m._iteration_count, EAGO._ext(m).gc_freq)==0
            #     GC.enable(true)
            #     GC.gc(false)
            #     GC.enable(false)
            # end

            # Increment the GPU iteration counter
            gpu_iteration_count += 1

            # Extract up to `max_parallel_nodes` nodes from the main problem stack
            count = 0
            while !iszero(length(m._stack))
                EAGO.node_selection!(m)
                this_time = @elapsed EAGO.preprocess!(m)
                m._last_preprocess_time += this_time
                ext.timers[12] .+= this_time

                # If the node has not been proven infeasible, add to the GPU stack
                this_time = @elapsed if m._preprocess_feasibility
                    count += 1
                    ext.node_storage[count] = m._current_node
                    ext.all_lvbs[count,:] .= @view ext.node_storage[count].lower_variable_bounds[1:end-m._epigraph_occurred]
                    ext.all_uvbs[count,:] .= @view ext.node_storage[count].upper_variable_bounds[1:end-m._epigraph_occurred]
                    if count == ext.max_parallel_nodes
                        break
                    end
                end
                ext.timers[16] .+= this_time
            end
            nodes_explored += count
            ext.node_len = count

            # Solve the lower problem for all nodes in parallel
            this_time = @elapsed lower_problem_gpu!(m)
            m._last_lower_problem_time += this_time
            ext.timers[17] .+= this_time
            
            EAGO.print_results!(m, true)
            for _ in 1:ext.node_len
                make_current_node!(m)

                # Check for infeasibility and store the solution
                if m._lower_feasibility && !EAGO.convergence_check(m)

                    # Solve the upper bounding problem and log the time
                    this_time = @elapsed EAGO.upper_problem!(m)
                    m._last_upper_problem_time += this_time
                    ext.timers[13] .+= this_time
                    EAGO.print_results!(m, false)

                    # Update the global upper bound if necessary
                    EAGO.store_candidate_solution!(m)

                    # Note: SKIP postprocessing, because variable_dbbt!() requires 
                    #       ConstraintDual() information from the lower solver. We
                    #       don't specifically calculate that with PDLP. We might
                    #       be able to, but we currently don't, so we can't do
                    #       duality-based bounds tightening, which is the only
                    #       postprocessing technique currently used by EAGO.
                    # Perform post processing and log the time
                    # this_time = @elapsed EAGO.postprocess!(m)
                    # m._last_postprocessing_time += this_time
                    # ext.timers[14] .+= this_time
                    m._postprocess_feasibility = true

                    # Continue if the node is not infeasible after postprocessing
                    if m._postprocess_feasibility

                        # If the node is to be repeatedly evaluated, add it back
                        # onto the stack. If not, branch on the node and add the
                        # two new nodes to the stack
                        this_time = @elapsed EAGO.repeat_check(m) ? EAGO.single_storage!(m) : EAGO.branch_node!(m)
                        ext.timers[15] .+= this_time
                    end
                end
            end

            # Update the global lower bound if necessary
            EAGO.set_global_lower_bound!(m)

            # Adjust algorithm run times
            m._run_time = time() - m._start_time
            m._time_left = m._parameters.time_limit - m._run_time

            # Log and print information as needed
            EAGO.log_iteration!(m)
            print_gpu_iteration!(m, gpu_iteration_count, false)
        end
    end

    # Since the algorithm has terminated, convert EAGO's end status into
    # MOI.TerminationStatusCode and MOI.ResultStatusCode
    EAGO.set_termination_status!(m)
    EAGO.set_result_status!(m)

    # Print final information about the solution
    EAGO.print_iteration!(m, true)
    EAGO.print_solution!(m)

    # Print extra profiling information
    println("  Process               |  Time (s)")
    println("=======================================")
    println("  EAGO Runtime:         | $(round(m._run_time, digits=6))")
    println("      Preprocessing:        | $(round(ext.timers[12][1], digits=6))")
    println("      Lower Problem:        | $(round(ext.timers[9][1] + ext.timers[17][1], digits=6))")
    println("          CPU Lower-Bounding:   | $(round(ext.timers[9][1], digits=6))")
    println("          GPU Lower-Bounding:   | $(round(ext.timers[17][1], digits=6))")
    # println("              Miscellaneous Setup:  | $(round(sum(ext.timers[2]), digits=6))")
    # println("              Data Transfer:        | $(round(sum(ext.timers[3]), digits=6))")
    # println("              Initial Point Setup:  | $(round(sum(ext.timers[4]), digits=6))")
    # println("              Relaxations:          | $(round(sum(ext.timers[5]), digits=6))")
    # println("              MultiSobol Rescaling: | $(round(sum(ext.timers[6]), digits=6))")
    # println("              Adding Constraints:   | $(round(sum(ext.timers[7]), digits=6))")
    # println("              Running PDLP:         | $(round(sum(ext.timers[8]), digits=6))")
    # println("              (Alt.) Preparing GLPK:| $(round(sum(ext.timers[10]), digits=6))")
    # println("              (Alt.) Running GLPK:  | $(round(sum(ext.timers[11]), digits=6))")
    # println("              Timing:               | $(round(sum(ext.timers[9]), digits=6))")
    # println("              [Untimed]:            | $(round(m._last_lower_problem_time - sum(sum.(ext.timers[1:11])), digits=6))")
    println("      Upper Problem:        | $(round(ext.timers[13][1], digits=6))")
    println("      Postprocessing:       | $(round(ext.timers[14][1], digits=6))")
    println("=======================================")
    println("")
    println("  Diagnostic          |  Value")
    println("=======================================")
    # println("  Iterations:         | $(m._iteration_count)")
    # println("      GPU Iterations: | $(gpu_iteration_count)")
    # println("      CPU Iterations: | $(m._iteration_count - gpu_iteration_count)")
    # println("  LPs Solved:         | $(sum(ext.LPs_solved))")
    # println("  Avg LPs / Iteration:| $(round(sum(ext.LPs_solved) / sum(ext.nodes_solved), digits=2))")
    println("=======================================")

    # Turn back on garbage collection
    GC.enable(true)
    GC.gc()
end
solve_gpu!(m::EAGO.GlobalOptimizer) = solve_gpu!(EAGO._ext(m), m)

function print_gpu_iteration!(m::EAGO.GlobalOptimizer, gpu_iteration_count::Int, end_flag::Bool)
    if EAGO._verbosity(m) > 0

        # Print an iteration summary on mod(gpu_iteration_count, 10)==1, OR if end_flag==true
        # if mod(gpu_iteration_count, 10)==0 || end_flag
        if true
            # Print start
            print_str = "| "

            # Print iteration number (with a star next to it, to indicate a GPU iteration)
            max_len = 11
            temp_str = string(m._iteration_count) * "*"
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print node count
            max_len = 11
            temp_str = string(m._node_count)
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print lower bound
            max_len = 11
            temp_str = string(round(m._global_lower_bound, digits=3))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print upper bound
            max_len = 11
            temp_str = string(round(m._global_upper_bound, digits=3))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print absolute gap between lower and upper bound
            max_len = 11
            temp_str = string(round(abs(m._global_upper_bound - m._global_lower_bound), digits=3))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print relative gap between lower and upper bound
            max_len = 11
            temp_str = string(round(EAGO.relative_gap(m._global_lower_bound, m._global_upper_bound), digits=3))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print run time
            max_len = 11
            temp_str = string(round(m._run_time, digits=2))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" | "

            # Print time remaining
            max_len = 11
            temp_str = string(round(m._time_left, digits=2))
            len_str = length(temp_str)
            print_str *= (" "^(max_len - len_str))*temp_str*" |"

            println(print_str)

            # Update printed iteration
            m._last_printed_iteration = m._iteration_count
        end
        # if end_flag
        #     println("-----------------------------------------------------------------------------------------------------------------")
        # end
    end

    return
end

make_current_node!(m::EAGO.GlobalOptimizer{R,S,Q}) where {R,S,Q<:EAGO.ExtensionType} = make_current_node!(EAGO._ext(m), m)
function make_current_node!(t::T, m::EAGO.GlobalOptimizer) where T <: ExtendGPU
    prev = copy(t.node_storage[t.node_len])
    new_lower = t.lower_bound_storage[t.node_len]
    m._lower_objective_value = max(prev.lower_bound, new_lower)
    m._lower_solution[1:end-1] .= t.CPU_LP_solutions[t.node_len,:]
    t.node_len -= 1
    m._current_node = NodeBB(prev.lower_variable_bounds, prev.upper_variable_bounds,
                             prev.is_integer, prev.continuous, new_lower, prev.upper_bound,
                             prev.depth, prev.cont_depth, prev.id, prev.branch_direction,
                             prev.last_branch, prev.branch_extent)
    if new_lower != Inf
        m._lower_feasibility = true
    else
        m._lower_feasibility = false
    end
end

# Lower-bounding routine specific to the PDLP MultiSobol method with one GPU
function lower_problem_gpu!(t::GroupMethod, m::EAGO.GlobalOptimizer)
    ################################################################################
    ##########  Step 0) Miscellaneous setup and tracking
    ################################################################################
    # Set time trackers to 0
    misc_setup = 0.0
    data_transfer = 0.0
    initial_linearization_setup = 0.0
    calculating_relaxations = 0.0
    multisobol_rescaling = 0.0
    adding_constraints = 0.0
    pdlp_solves = 0.0
    cpu_solver_setup = 0.0
    cpu_solve_time = 0.0
    
    # Add information about how many nodes were solved
    misc_setup += @elapsed push!(t.nodes_solved, t.node_len)


    ################################################################################
    ##########  Step 1) Determine problem parameters
    ################################################################################
    misc_setup += @elapsed begin
        n_LPs = t.node_len 
        n_points = t.node_len * t.num_eval_points
        var_count = t.n_vars
        geq_len = length(t.geq_cons)
        leq_len = length(t.leq_cons)
        eq_len = length(t.eq_cons)
    end


    ################################################################################
    ##########  Step 2) Set up initial evaluation points
    ################################################################################
    # Load in lower and upper bounds from t.all_lvbs / t.all_uvbs
    data_transfer += @elapsed CUDA.@sync begin
        # Load in lower and upper bounds
        copyto!(t.lvbs_d, t.all_lvbs)
        copyto!(t.uvbs_d, t.all_uvbs)

        # Copy lower and upper bounds to the temp storage for the MultiSobol method
        copyto!(t.lvbs_d_temp, t.lvbs_d)
        copyto!(t.uvbs_d_temp, t.uvbs_d)
    end

    # Load in the linearization points for every variable, for every LP
    initial_linearization_setup += @elapsed CUDA.@sync for var in 1:var_count
        CUDA.@sync @cuda blocks = t.n_blocks threads=Int32(1024) load_linearization_points_sobol_kernel(
            t.input_storage[var],
            t.eval_points,
            t.lvbs_d,
            t.uvbs_d,
            n_LPs,
            t.linearization_points, 
            t.num_eval_points,
            var
        )
    end


    ################################################################################
    ##########  Step 3) Reset the PDLP struct
    ################################################################################

    misc_setup += @elapsed CUDA.@sync begin
        # Set up a convenient reference for the original_problem field
        LPs = t.PDLP_data.original_problem

        # Reset bounds to [-Inf, [lvbs]...], [Inf, [uvbs]...], where the first index
        # is for the epigraph variable
        @views begin
            LPs.variable_lower_bounds[1:n_LPs,1] .= -Inf
            LPs.variable_lower_bounds[1:n_LPs,2:end] .= t.lvbs_d[1:n_LPs,:]
            LPs.variable_upper_bounds[1:n_LPs,1] .= Inf
            LPs.variable_upper_bounds[1:n_LPs,2:end] .= t.uvbs_d[1:n_LPs,:]
        end

        # Reset the constraint matrix and right-hand side to zeros
        CUDA.fill!(LPs.constraint_matrix, 0.0)
        CUDA.fill!(LPs.right_hand_side, 0.0)

        # Reset the objective vector to minimize the epigraph variable
        LPs.objective_vector[1:n_LPs,1] .= 1.0
        LPs.objective_vector[1:n_LPs,2:end] .= 0.0
        CUDA.fill!(LPs.objective_constant, 0.0)

        # Set the current LP length to 0, and identify the number of LPs
        t.PDLP_data.dims.current_LP_length = Int32(0) # Zero, until we add constraints
        t.PDLP_data.dims.n_LPs = Int32(n_LPs)

        # Reset the skip flag to be `false` for 1:n_LPs and `true` for n_LPs+1:end
        t.PDLP_data.skip_flag[1:n_LPs] .= false
        t.PDLP_data.skip_flag[n_LPs+1:end] .= true

        # Set the comparison vector and subgradient checksum back to zeros
        CUDA.fill!(t.comparison_vector, 0.0)
        CUDA.fill!(t.subgradient_checksum, 0.0)

        # Reset active constraint flag to be false (gets set to true when new
        # constraints are added)
        CUDA.fill!(t.PDLP_data.active_constraint, false)
    end
    

    ################################################################################
    ##########  Step 4) Calculate relaxations and add constraints
    ################################################################################

    # Calculate relaxations for the objective function 
    calculating_relaxations += @elapsed CUDA.@sync @views t.obj_fun(t.result_storage[1:n_points,:], [t.input_storage[j][1:n_points,:] for j=1:var_count]...)

    # Based on the relaxations in `result_storage`, determine an improved set
    # of linearization points and recalculate the objective function relaxations
    # at those points
    if t.perform_scan
        multisobol_rescaling += @elapsed CUDA.@sync for var in 1:var_count
            CUDA.@sync @cuda blocks=t.n_blocks threads=Int32(1024) rescale_linearization_points_kernel(
                t.input_storage[var],
                t.eval_points,
                t.result_storage,
                t.lvbs_d_temp,
                t.uvbs_d_temp,
                n_LPs,
                t.num_eval_points,
                Int32(var),
            )
        end
        calculating_relaxations += @elapsed CUDA.@sync @views t.obj_fun(t.result_storage[1:n_points,:], [t.input_storage[j][1:n_points,:] for j=1:var_count]...)
    end

    # Add LP constraints based on the objective function relaxation results
    adding_constraints += @elapsed CUDA.@sync begin
        # Update the comparison vector with the sum of the squares of convex relaxation
        # subgradient values, scaled by domain size in each dimension
        sum_subgradients(t.comparison_vector, t.lvbs_d, t.uvbs_d, t.result_storage, n_points, t.num_eval_points, var_count, t.n_blocks)

        # Get the unique indicator for subgradient signs to not add similar constraints later
        extract_subgradient_sign(t.subgradient_checksum, t.result_storage, n_points, var_count, t.n_blocks)

        # Add an LP constraint based on the interval extension (equal for all linearization
        # points, currently. Could change in the future if subgradient propagation is used)
        @views BatchPDLP.add_multiple_LP_lower_bound(LPs, t.result_storage[1:n_points,:], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points)

        # Add LP constraints from the objective function, with the number of constraints
        # to add based on the sparsity of the objective function
        @views BatchPDLP.add_best_obj_LP_constraints(LPs, t.result_storage[1:n_points,:], t.eval_points[1:n_points,:], t.comparison_vector[1:n_points], t.subgradient_checksum[1:n_points], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points, sum(t.obj_sp)+1)
    end

    # Scale the input storage points back to the original domain
    if t.perform_scan
        multisobol_rescaling += @elapsed CUDA.@sync for var in 1:var_count
            CUDA.@sync @cuda blocks=t.n_blocks threads=Int32(1024) scale_back_linearization_points_kernel(
                t.input_storage[var],
                t.eval_points,
                t.lvbs_d_temp,
                t.uvbs_d_temp,
                n_LPs,
                t.num_eval_points,
                Int32(var),
            )
        end
    end

    # EQ constraints
    for i = 1:eq_len
        # Calculate relaxations for the i-th equality constraint
        calculating_relaxations += @elapsed CUDA.@sync @views t.eq_cons[i](t.result_storage[1:n_points,:], [t.input_storage[j][1:n_points,:] for j=1:var_count]...)

        adding_constraints += @elapsed CUDA.@sync begin
            # Update the comparison vector with the value of the CONVEX relaxations (for the LEQ side of the EQ constraint)
            extract_convex_relaxation(t.comparison_vector, t.result_storage, n_points, t.n_blocks)
            
            # Get the unique indicator for subgradient signs to not add similar constraints later
            extract_subgradient_sign(t.subgradient_checksum, t.result_storage, n_points, var_count, t.n_blocks)

            # Add the LEQ constraint(s)
            @views add_best_cons_LP_constraints(LPs, t.result_storage[1:n_points,:], t.eval_points[1:n_points,:], t.comparison_vector[1:n_points], t.subgradient_checksum[1:n_points], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=false)

            # Update the comparison vector with the value of the CONCAVE relaxations (for the GEQ side of the EQ constraint)
            extract_concave_relaxation(t.comparison_vector, t.result_storage, n_points, t.n_blocks)
            
            # Get the unique indicator for subgradient signs to not add similar constraints later
            extract_subgradient_sign(t.subgradient_checksum, t.result_storage, n_points, var_count, t.n_blocks, concave=true)

            # Add the GEQ constraint(s)
            @views add_best_cons_LP_constraints(LPs, t.result_storage[1:n_points,:], t.eval_points[1:n_points,:], t.comparison_vector[1:n_points], t.subgradient_checksum[1:n_points], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=true)
        end
    end

    # LEQ constraints
    for i = 1:leq_len
        calculating_relaxations += @elapsed CUDA.@sync @views t.leq_cons[i](t.result_storage[1:n_points,:], [t.input_storage[j][1:n_points,:] for j=1:var_count]...)
        adding_constraints += @elapsed CUDA.@sync begin
            extract_convex_relaxation(t.comparison_vector, t.result_storage, n_points, t.n_blocks)
            extract_subgradient_sign(t.subgradient_checksum, t.result_storage, n_points, var_count, t.n_blocks)
            @views add_best_cons_LP_constraints(LPs, t.result_storage[1:n_points,:], t.eval_points[1:n_points,:], t.comparison_vector[1:n_points], t.subgradient_checksum[1:n_points], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points, sum(t.leq_sp[i])+1, geq=false)
        end
    end

    # GEQ constraints
    for i = 1:geq_len
        calculating_relaxations += @elapsed CUDA.@sync @views t.geq_cons[i](t.result_storage[1:n_points,:], [t.input_storage[j][1:n_points,:] for j=1:var_count]...)
        adding_constraints += @elapsed CUDA.@sync begin
            extract_concave_relaxation(t.comparison_vector, t.result_storage, n_points, t.n_blocks)
            extract_subgradient_sign(t.subgradient_checksum, t.result_storage, n_points, var_count, t.n_blocks, concave=true)
            @views add_best_cons_LP_constraints(LPs, t.result_storage[1:n_points,:], t.eval_points[1:n_points,:], t.comparison_vector[1:n_points], t.subgradient_checksum[1:n_points], t.PDLP_data.dims, t.PDLP_data.active_constraint, t.num_eval_points, sum(t.geq_sp[i])+1, geq=true)
        end
    end


    ################################################################################
    ##########  Step 5) Run PDLP
    ################################################################################

    # Run BatchPDLP if we have enough parallel LPs (otherwise, use the CPU solver)
    if t.node_len > t.GPU_LP_break_point
        # Run BatchPDLP
        pdlp_solves += @elapsed PDLP(t.PDLP_data, solutions=t.LP_solutions, objectives=t.LP_objectives, return_dual_obj=t.use_dual_obj, global_upper_bound = m._global_upper_bound)
        misc_setup += @elapsed push!(t.LPs_solved, t.node_len)

        # Copy objectives and solutions to the CPU (If hard problems are being
        # skipped, some of these values will be -Inf from BatchPDLP, to be filled
        # in by the CPU solver shortly)
        data_transfer += @elapsed CUDA.@sync begin
            copyto!(t.lower_bound_storage, t.LP_objectives)
            copyto!(t.CPU_LP_solutions, t.LP_solutions)
        end

        # If hard problems are being skipped, solve the hard problems using
        # the designated CPU solver
        if t.PDLP_data.parameters.skip_hard_problems
            data_transfer += @elapsed CUDA.@sync begin
                copyto!(t.cpu_constraint_mat, t.PDLP_data.original_problem.constraint_matrix)
                copyto!(t.cpu_rhs, t.PDLP_data.original_problem.right_hand_side)
                copyto!(t.cpu_solve_flag, t.PDLP_data.termination_reason .== BatchPDLP.TERMINATION_REASON_IMPATIENCE)
                copyto!(t.cpu_active_constraint, t.PDLP_data.active_constraint)
            end

            for i = 1:t.node_len
                if !t.cpu_solve_flag[i]
                    continue
                end

                cpu_solver_setup += @elapsed begin
                    # Reset the solver
                    MOI.empty!(t.cpu_solver)

                    # Relaxations will be added using an epigraph variable
                    epi = MOI.add_variable(t.cpu_solver)

                    # Create the variables, using the bounds for the i-th problem
                    vi = Vector{MOI.VariableIndex}(undef, var_count)
                    for j = 1:var_count
                        vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                    end

                    # Identify the start point within the constraint/RHS fields
                    start = (i-1)*t.PDLP_data.dims.total_LP_length

                    # Add the constraints
                    for j = 1:t.PDLP_data.dims.current_LP_length
                        if t.cpu_active_constraint[start+j]
                            @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                    sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                    MOI.GreaterThan(t.cpu_rhs[start+j]))
                        end
                    end

                    # Add the objective function (always already in epigraph form)
                    MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                    MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
                end

                # Optimize
                cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

                # Move answers to lower-bound storage
                cpu_solver_setup += @elapsed begin
                    term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                    if term == MOI.OPTIMAL
                        if t.use_dual_obj
                            t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                        else
                            t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                        end
                        t.CPU_LP_solutions[i,:] .= MOI.get.(t.cpu_solver, MOI.VariablePrimal(), vi)
                    else
                        t.lower_bound_storage[i] = Inf
                    end
                end
            end
        end

        # If the `validate_vs_CPU` option is `true`, compare the BatchPDLP results
        # against the default CPU-based solver
        if t.validate_vs_CPU
            batchPDLP_termination_reason = Array(t.PDLP_data.termination_reason)
            batchPDLP_bound = Array(t.LP_objectives)
            data_transfer += @elapsed CUDA.@sync begin
                copyto!(t.cpu_constraint_mat, t.PDLP_data.original_problem.constraint_matrix)
                copyto!(t.cpu_rhs, t.PDLP_data.original_problem.right_hand_side)
                copyto!(t.cpu_solve_flag, t.PDLP_data.termination_reason .== BatchPDLP.TERMINATION_REASON_IMPATIENCE)
                copyto!(t.cpu_active_constraint, t.PDLP_data.active_constraint)
            end
            for i = 1:t.node_len
                cpu_solver_setup += @elapsed begin
                    # Reset the solver
                    MOI.empty!(t.cpu_solver)

                    # Relaxations will be added using an epigraph variable
                    epi = MOI.add_variable(t.cpu_solver)

                    # Create the variables, using the bounds for the i-th problem
                    vi = Vector{MOI.VariableIndex}(undef, var_count)
                    for j = 1:var_count
                        vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                    end

                    # Identify the start point within the constraint/RHS fields
                    start = (i-1)*t.PDLP_data.dims.total_LP_length

                    # Add the constraints
                    for j = 1:t.PDLP_data.dims.current_LP_length
                        if t.cpu_active_constraint[start+j]
                            @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                    sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                    MOI.GreaterThan(t.cpu_rhs[start+j]))
                        end
                    end

                    # Add the objective function (always already in epigraph form)
                    MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                    MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
                end

                # Optimize
                cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

                # Compare termination status and results with BatchPDLP
                term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                if term == MOI.OPTIMAL
                    cpu_bound = 0.0
                    if t.use_dual_obj
                        cpu_bound =  MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                    else
                        cpu_bound =  MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                    end
                    if batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_OPTIMAL
                        if !(batchPDLP_termination_reason[i] == BatchPDLP.TERMINATION_REASON_GLOBAL_UPPER_BOUND_HIT ||
                             batchPDLP_termination_reason[i] == BatchPDLP.TERMINATION_REASON_IMPATIENCE)
                             println("For $i: (wrong termination)")
                             println("CPU solver says $term ($cpu_bound)")
                             println("BatchPDLP says $(batchPDLP_termination_reason[i]), ($(batchPDLP_bound[i]))")
                        end
                    elseif !isapprox(cpu_bound, batchPDLP_bound[i], atol=1e-6, rtol=1e-6)
                        if batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_ITERATION_LIMIT
                            println("For $i: (wrong solution)")
                            println("CPU solver says $cpu_bound")
                            println("BatchPDLP says $(batchPDLP_bound[i])")
                        end
                    end
                else
                    if (batchPDLP_bound[i] != Inf) && (batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_ITERATION_LIMIT)
                        println("For $i: (wrong termination)")
                        println("CPU solver says $term")
                        println("BatchPDLP says $(batchPDLP_termination_reason[i])")
                    end
                end
            end
        end
    else
        # Only run the CPU solver
        data_transfer += @elapsed CUDA.@sync begin
            copyto!(t.cpu_constraint_mat, t.PDLP_data.original_problem.constraint_matrix)
            copyto!(t.cpu_rhs, t.PDLP_data.original_problem.right_hand_side)
            copyto!(t.cpu_active_constraint, t.PDLP_data.active_constraint)
        end

        for i = 1:t.node_len
            cpu_solver_setup += @elapsed CUDA.@sync begin
                # Reset the solver
                MOI.empty!(t.cpu_solver)

                # Relaxations will be added using an epigraph variable
                epi = MOI.add_variable(t.cpu_solver)

                # Create the variables, using the bounds for the i-th problem
                vi = Vector{MOI.VariableIndex}(undef, var_count)
                for j = 1:var_count
                    vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                end

                # Identify the start point within the constraint/RHS fields
                start = (i-1)*t.PDLP_data.dims.total_LP_length

                # Add the constraints
                for j = 1:t.PDLP_data.dims.current_LP_length
                    if t.cpu_active_constraint[start+j]
                        @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                MOI.GreaterThan(t.cpu_rhs[start+j]))
                    end
                end

                # Add the objective function (always already in epigraph form)
                MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
            end

            # Optimize
            cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

            # Move answers to lower-bound storage
            cpu_solver_setup += @elapsed begin
                term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                if term == MOI.OPTIMAL
                    if t.use_dual_obj
                        t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                    else
                        t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                    end
                    t.CPU_LP_solutions[i,:] .= MOI.get.(t.cpu_solver, MOI.VariablePrimal(), vi)
                else
                    t.lower_bound_storage[i] = Inf
                end
            end
        end
        misc_setup += @elapsed push!(t.LPs_solved, t.node_len)
    end

    # Push profiling information to timers
    push!(t.timers[2], misc_setup)
    push!(t.timers[3], data_transfer)
    push!(t.timers[4], initial_linearization_setup)
    push!(t.timers[5], calculating_relaxations)
    push!(t.timers[6], multisobol_rescaling)
    push!(t.timers[7], adding_constraints)
    push!(t.timers[8], pdlp_solves)
    push!(t.timers[10], cpu_solver_setup)
    push!(t.timers[11], cpu_solve_time)
    return nothing
end
lower_problem_gpu!(m::EAGO.GlobalOptimizer) = lower_problem_gpu!(EAGO._ext(m), m)

# Lower-bounding routine specific to the PDLP MultiSobol method with two GPUs
function lower_problem_gpu!(t::GroupMethod_MultiGPU, m::EAGO.GlobalOptimizer)
    ################################################################################
    ##########  Step 0) Miscellaneous setup and tracking
    ################################################################################
    # Set time trackers to 0
    misc_setup = 0.0
    data_transfer = 0.0
    pdlp_solves = 0.0
    initial_linearization_setup = 0.0
    calculating_relaxations = 0.0
    multisobol_rescaling = 0.0
    adding_constraints = 0.0
    cpu_solver_setup = 0.0
    cpu_solve_time = 0.0
    
    # Add information about how many nodes were solved
    misc_setup += @elapsed push!(t.nodes_solved, t.node_len)


    ################################################################################
    ##########  Step 1) Determine problem parameters
    ################################################################################
    misc_setup += @elapsed begin
        n_LPs = [Int(ceil(t.node_len/2)), Int(t.node_len - ceil(t.node_len/2))]
        n_points = n_LPs .* t.num_eval_points
        var_count = t.n_vars
        geq_len = length(t.geq_cons)
        leq_len = length(t.leq_cons)
        eq_len = length(t.eq_cons)
    end


    ################################################################################
    ##########  Step 2) Set up initial evaluation points
    ################################################################################
    # Load in lower and upper bounds from t.all_lvbs / t.all_uvbs
    #    NOTE: It's difficult to split the bounds appropriately between the two
    #          GPUs, because "copyto!" doesn't like to work with slices of data.
    #          So, even if t.all_lvbs isn't full, I still copy the first L rows
    #          to the first GPU's memory, and then start at one plus the actual
    #          number of bounds that should've been in the first GPU, for the
    #          second GPU. In effect, some bound values are technically doubled,
    #          and appear in both GPUs, but in later steps the first GPU will
    #          simply stop doing calculations before reaching these repeated values.
    L = size(t.lvbs_d[1], 1)
    data_transfer += @elapsed CUDA.@sync begin
        @sync begin
            @async begin
                device!(0)
                gpu = 1

                # Load in lower and upper bounds
                copyto!(t.lvbs_d[gpu], t.all_lvbs[1:L,:])
                copyto!(t.uvbs_d[gpu], t.all_uvbs[1:L,:])

                # Copy lower and upper bounds to the temp storage for the MultiSobol method
                copyto!(t.lvbs_d_temp[gpu], t.lvbs_d[gpu])
                copyto!(t.uvbs_d_temp[gpu], t.uvbs_d[gpu])
            end
            @async begin
                device!(1)
                gpu = 2

                # Load in lower and upper bounds
                copyto!(t.lvbs_d[gpu], t.all_lvbs[n_LPs[1]+1:n_LPs[1]+L,:])
                copyto!(t.uvbs_d[gpu], t.all_uvbs[n_LPs[1]+1:n_LPs[1]+L,:])

                # Copy lower and upper bounds to the temp storage for the multi-sobol method
                copyto!(t.lvbs_d_temp[gpu], t.lvbs_d[gpu])
                copyto!(t.uvbs_d_temp[gpu], t.uvbs_d[gpu])
            end
        end
    end

    # Load in the linearization points for every variable, for every LP
    initial_linearization_setup += @elapsed CUDA.@sync begin
        @sync begin
            @async begin
                device!(0)
                gpu = 1

                for var in 1:var_count
                    CUDA.@sync @cuda blocks = t.n_blocks[gpu] threads=Int32(1024) load_linearization_points_sobol_kernel(
                        t.input_storage[gpu][var],
                        t.eval_points[gpu],
                        t.lvbs_d[gpu],
                        t.uvbs_d[gpu],
                        n_LPs[gpu],
                        t.linearization_points[gpu], 
                        t.num_eval_points,
                        var
                    )
                end
            end
            @async begin
                device!(1)
                gpu = 2

                for var in 1:var_count
                    CUDA.@sync @cuda blocks = t.n_blocks[gpu] threads=Int32(1024) load_linearization_points_sobol_kernel(
                        t.input_storage[gpu][var],
                        t.eval_points[gpu],
                        t.lvbs_d[gpu],
                        t.uvbs_d[gpu],
                        n_LPs[gpu],
                        t.linearization_points[gpu], 
                        t.num_eval_points,
                        var
                    )
                end
            end
        end
    end


    ################################################################################
    ##########  Step 3) Reset the PDLP struct
    ################################################################################

    misc_setup += @elapsed CUDA.@sync begin
        @sync begin
            @async begin
                device!(0)
                gpu = 1

                # Set up a convenient reference for the original_problem field
                LPs = t.PDLP_data[gpu].original_problem

                # Reset bounds to [-Inf, [lvbs]...], [Inf, [uvbs]...], where the first index
                # is for the epigraph variable
                @views begin
                    LPs.variable_lower_bounds[1:n_LPs[gpu],1] .= -Inf
                    LPs.variable_lower_bounds[1:n_LPs[gpu],2:end] .= t.lvbs_d[gpu][1:n_LPs[gpu],:]
                    LPs.variable_upper_bounds[1:n_LPs[gpu],1] .= Inf
                    LPs.variable_upper_bounds[1:n_LPs[gpu],2:end] .= t.uvbs_d[gpu][1:n_LPs[gpu],:]
                end

                # Reset the constraint matrix and right-hand side to zeros
                CUDA.fill!(LPs.constraint_matrix, 0.0)
                CUDA.fill!(LPs.right_hand_side, 0.0)

                # Reset the objective vector to minimize the epigraph variable
                LPs.objective_vector[1:n_LPs[gpu],1] .= 1.0
                LPs.objective_vector[1:n_LPs[gpu],2:end] .= 0.0
                CUDA.fill!(LPs.objective_constant, 0.0)

                # Set the current LP length to 0, and identify the number of LPs
                t.PDLP_data[gpu].dims.current_LP_length = Int32(0) # Zero, until we add constraints
                t.PDLP_data[gpu].dims.n_LPs = Int32(n_LPs[gpu])

                # Reset the skip flag to be `false` for 1:n_LPs and `true` for n_LPs+1:end
                t.PDLP_data[gpu].skip_flag[1:n_LPs[gpu]] .= false
                t.PDLP_data[gpu].skip_flag[n_LPs[gpu]+1:end] .= true
                
                # Set the comparison vector and subgradient checksum back to zeros
                CUDA.fill!(t.comparison_vector[gpu], 0.0)
                CUDA.fill!(t.subgradient_checksum[gpu], 0.0)

                # Reset active constraint flag to be false (gets set to true when new
                # constraints are added)
                CUDA.fill!(t.PDLP_data[gpu].active_constraint, false)
            end
            @async begin
                device!(1)
                gpu = 2

                # Set up a convenient reference for the original_problem field
                LPs = t.PDLP_data[gpu].original_problem

                # Reset bounds to [-Inf, [lvbs]...], [Inf, [uvbs]...], where the first index
                # is for the epigraph variable
                @views begin
                    LPs.variable_lower_bounds[1:n_LPs[gpu],1] .= -Inf
                    LPs.variable_lower_bounds[1:n_LPs[gpu],2:end] .= t.lvbs_d[gpu][1:n_LPs[gpu],:]
                    LPs.variable_upper_bounds[1:n_LPs[gpu],1] .= Inf
                    LPs.variable_upper_bounds[1:n_LPs[gpu],2:end] .= t.uvbs_d[gpu][1:n_LPs[gpu],:]
                end

                # Reset the constraint matrix and right-hand side to zeros
                CUDA.fill!(LPs.constraint_matrix, 0.0)
                CUDA.fill!(LPs.right_hand_side, 0.0)

                # Reset the objective vector to minimize the epigraph variable
                LPs.objective_vector[1:n_LPs[gpu],1] .= 1.0
                LPs.objective_vector[1:n_LPs[gpu],2:end] .= 0.0
                CUDA.fill!(LPs.objective_constant, 0.0)

                # Set the current LP length to 0, and identify the number of LPs
                t.PDLP_data[gpu].dims.current_LP_length = Int32(0) # Zero, until we add constraints
                t.PDLP_data[gpu].dims.n_LPs = Int32(n_LPs[gpu])

                # Reset the skip flag to be `false` for 1:n_LPs and `true` for n_LPs+1:end
                t.PDLP_data[gpu].skip_flag[1:n_LPs[gpu]] .= false
                t.PDLP_data[gpu].skip_flag[n_LPs[gpu]+1:end] .= true
                
                # Set the comparison vector and subgradient checksum back to zeros
                CUDA.fill!(t.comparison_vector[gpu], 0.0)
                CUDA.fill!(t.subgradient_checksum[gpu], 0.0)

                # Reset active constraint flag to be false (gets set to true when new
                # constraints are added)
                CUDA.fill!(t.PDLP_data[gpu].active_constraint, false)
            end
        end
    end


    ################################################################################
    ##########  Step 4) Calculate relaxations and add constraints
    ################################################################################

    # Calculate relaxations for the objective function
    calculating_relaxations += @elapsed CUDA.@sync begin
        @sync begin
            @async begin
                device!(0)
                gpu = 1

                CUDA.@sync @views t.obj_fun(t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
            end
            @async begin
                device!(1)
                gpu = 2

                CUDA.@sync @views t.obj_fun(t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
            end
        end
    end


    # Based on the relaxations in `result_storage`, determine an improved set
    # of linearization points and recalculate the objective function relaxations
    # at those points
    if t.perform_scan
        multisobol_rescaling += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    for var in 1:var_count
                        CUDA.@sync @cuda blocks=t.n_blocks[gpu] threads=Int32(1024) rescale_linearization_points_kernel(
                            t.input_storage[gpu][var],
                            t.eval_points[gpu],
                            t.result_storage[gpu],
                            t.lvbs_d_temp[gpu],
                            t.uvbs_d_temp[gpu],
                            n_LPs[gpu],
                            t.num_eval_points,
                            Int32(var),
                        )
                    end
                end
                @async begin
                    device!(1)
                    gpu = 2

                    for var in 1:var_count
                        CUDA.@sync @cuda blocks=t.n_blocks[gpu] threads=Int32(1024) rescale_linearization_points_kernel(
                            t.input_storage[gpu][var],
                            t.eval_points[gpu],
                            t.result_storage[gpu],
                            t.lvbs_d_temp[gpu],
                            t.uvbs_d_temp[gpu],
                            n_LPs[gpu],
                            t.num_eval_points,
                            Int32(var),
                        )
                    end
                end
            end
        end
        calculating_relaxations += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    CUDA.@sync @views t.obj_fun(t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    CUDA.@sync @views t.obj_fun(t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
            end
        end
    end

    # Add LP constraints based on the objective function relaxation results
    adding_constraints += @elapsed CUDA.@sync begin
        @sync begin
            @async begin
                device!(0)
                gpu = 1

                # Update the comparison vector with the sum of the squares of convex relaxation
                # subgradient values, scaled by domain size in each dimension
                sum_subgradients(t.comparison_vector[gpu], t.lvbs_d[gpu], t.uvbs_d[gpu], t.result_storage[gpu], n_points[gpu], t.num_eval_points, var_count, t.n_blocks[gpu])

                # Get the unique indicator for subgradient signs to not add similar constraints later
                extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu])

                # Add an LP constraint based on the interval extension (equal for all linearization
                # points, currently. Could change in the future if subgradient propagation is used)
                @views BatchPDLP.add_multiple_LP_lower_bound(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points)

                # Add LP constraints from the objective function, with the number of constraints
                # to add based on the sparsity of the objective function
                @views BatchPDLP.add_best_obj_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.obj_sp)+1)
            end
            @async begin
                device!(1)
                gpu = 2

                sum_subgradients(t.comparison_vector[gpu], t.lvbs_d[gpu], t.uvbs_d[gpu], t.result_storage[gpu], n_points[gpu], t.num_eval_points, var_count, t.n_blocks[gpu])
                extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu])
                @views BatchPDLP.add_multiple_LP_lower_bound(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points)
                @views BatchPDLP.add_best_obj_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.obj_sp)+1)
            end
        end
    end

    # Scale the input storage points back to the original domain
    if t.perform_scan
        multisobol_rescaling += @elapsed CUDA.@sync begin
            @sync begin 
                @async begin
                    device!(0)
                    gpu = 1
                    
                    for var in 1:var_count
                        CUDA.@sync @cuda blocks=t.n_blocks[gpu] threads=Int32(1024) scale_back_linearization_points_kernel(
                            t.input_storage[gpu][var],
                            t.eval_points[gpu],
                            t.lvbs_d_temp[gpu],
                            t.uvbs_d_temp[gpu],
                            n_LPs[gpu],
                            t.num_eval_points,
                            Int32(var),
                        )
                    end
                end
                @async begin
                    device!(1)
                    gpu = 2
                    
                    for var in 1:var_count
                        CUDA.@sync @cuda blocks=t.n_blocks[gpu] threads=Int32(1024) scale_back_linearization_points_kernel(
                            t.input_storage[gpu][var],
                            t.eval_points[gpu],
                            t.lvbs_d_temp[gpu],
                            t.uvbs_d_temp[gpu],
                            n_LPs[gpu],
                            t.num_eval_points,
                            Int32(var),
                        )
                    end
                end
            end
        end
    end

    # EQ constraints
    for i = 1:eq_len
        # Calculate relaxations for the i-th equality constraint
        calculating_relaxations += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    CUDA.@sync @views t.eq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    CUDA.@sync @views t.eq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
            end
        end

        adding_constraints += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    # Update the comparison vector with the value of the CONVEX relaxations (for the LEQ side of the EQ constraint)
                    extract_convex_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    
                    # Calculate something that depends on the sign of the subgradient terms
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu])

                    # Add the LEQ constraints
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=false)

                    # Update the comparison vector with the value of the CONCAVE relaxations (for the GEQ side of the EQ constraint)
                    extract_concave_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    
                    # Calculate something that depends on the sign of the subgradient terms
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu], concave=true)

                    # Add the GEQ constraints
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=true)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    extract_convex_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu])
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=false)
                    extract_concave_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu], concave=true)
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.eq_sp[i])+1, geq=true)
                end
            end
        end
    end

    # LEQ constraints
    for i = 1:leq_len
        calculating_relaxations += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    CUDA.@sync @views t.leq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    CUDA.@sync @views t.leq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
            end
        end

        adding_constraints += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    extract_convex_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu]) 
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.leq_sp[i])+1, geq=false)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    extract_convex_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu]) 
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.leq_sp[i])+1, geq=false)
                end
            end
        end
    end

    # GEQ constraints
    for i = 1:geq_len
        calculating_relaxations += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    CUDA.@sync @views t.geq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    CUDA.@sync @views t.geq_cons[i](t.result_storage[gpu][1:n_points[gpu],:], [t.input_storage[gpu][j][1:n_points[gpu],:] for j=1:var_count]...)
                end
            end
        end

        adding_constraints += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1

                    extract_concave_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu], concave=true)
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.geq_sp[i])+1, geq=true)
                end
                @async begin
                    device!(1)
                    gpu = 2

                    extract_concave_relaxation(t.comparison_vector[gpu], t.result_storage[gpu], n_points[gpu], t.n_blocks[gpu])
                    extract_subgradient_sign(t.subgradient_checksum[gpu], t.result_storage[gpu], n_points[gpu], var_count, t.n_blocks[gpu], concave=true)
                    @views add_best_cons_LP_constraints(t.PDLP_data[gpu].original_problem, t.result_storage[gpu][1:n_points[gpu],:], t.eval_points[gpu][1:n_points[gpu],:], t.comparison_vector[gpu][1:n_points[gpu]], t.subgradient_checksum[gpu][1:n_points[gpu]], t.PDLP_data[gpu].dims, t.PDLP_data[gpu].active_constraint, t.num_eval_points, sum(t.geq_sp[i])+1, geq=true)
                end
            end
        end
        
    end


    ################################################################################
    ##########  Step 5) Run PDLP
    ################################################################################

    # Run BatchPDLP if we have enough parallel LPs (otherwise, use the CPU solver)
    if t.node_len > t.GPU_LP_break_point
        # Run BatchPDLP
        pdlp_solves += @elapsed CUDA.@sync begin
            @sync begin
                @async begin
                    device!(0)
                    gpu = 1
                    
                    PDLP(t.PDLP_data[gpu], solutions=t.LP_solutions[gpu], objectives=t.LP_objectives[gpu], return_dual_obj=t.use_dual_obj, global_upper_bound = m._global_upper_bound)
                end
                @async begin
                    device!(1)
                    gpu = 2
                    
                    PDLP(t.PDLP_data[gpu], solutions=t.LP_solutions[gpu], objectives=t.LP_objectives[gpu], return_dual_obj=t.use_dual_obj, global_upper_bound = m._global_upper_bound)
                end
            end
        end
        misc_setup += @elapsed push!(t.LPs_solved, t.node_len)
        
        # Copy objectives and solutions to the CPU (If hard problems are being
        # skipped, some of these values will be -Inf from BatchPDLP, to be filled
        # in by the CPU solver shortly)
        data_transfer += @elapsed CUDA.@sync begin
            device!(0)
            t.lower_bound_storage[1:n_LPs[1]] .= Array(t.LP_objectives[1])[1:n_LPs[1]]
            t.CPU_LP_solutions[1:n_LPs[1],:] .= Array(t.LP_solutions[1])[1:n_LPs[1]]
            device!(1)
            t.lower_bound_storage[n_LPs[1]+1:t.node_len] .= Array(t.LP_objectives[2])[1:n_LPs[2]]
            t.CPU_LP_solutions[n_LPs[1]+1:t.node_len,:] .= Array(t.LP_solutions[2])[1:n_LPs[2]]
        end

        # If hard problems are being skipped, solve the hard problems using
        # the designated CPU solver
        if t.PDLP_data[1].parameters.skip_hard_problems # Same for PDLP_data[1] and PDLP_data[2]
            data_transfer += @elapsed CUDA.@sync begin
                device!(0)
                t.cpu_constraint_mat[1:Int(end/2),:] .= Array(t.PDLP_data[1].original_problem.constraint_matrix)
                t.cpu_rhs[1:Int(end/2)] .= Array(t.PDLP_data[1].original_problem.right_hand_side)
                t.cpu_solve_flag[1:Int(end/2)] .= Array(t.PDLP_data[1].termination_reason) .== BatchPDLP.TERMINATION_REASON_IMPATIENCE
                t.cpu_active_constraint[1:Int(end/2)] .= Array(t.PDLP_data[1].active_constraint)
                device!(1)
                t.cpu_constraint_mat[Int(end/2)+1:end,:] .= Array(t.PDLP_data[2].original_problem.constraint_matrix)
                t.cpu_rhs[Int(end/2)+1:end] .= Array(t.PDLP_data[2].original_problem.right_hand_side)
                t.cpu_solve_flag[Int(end/2)+1:end] .= Array(t.PDLP_data[2].termination_reason) .== BatchPDLP.TERMINATION_REASON_IMPATIENCE
                t.cpu_active_constraint[Int(end/2)+1:end] .= Array(t.PDLP_data[2].active_constraint)
            end
            device!(0)

            LP_length = t.PDLP_data[1].dims.total_LP_length
            for i = 1:size(t.cpu_solve_flag, 1)
                if !t.cpu_solve_flag[i]
                    continue
                end

                cpu_solver_setup += @elapsed begin
                    # Reset the solver
                    MOI.empty!(t.cpu_solver)

                    # Relaxations will be added using an epigraph variable
                    epi = MOI.add_variable(t.cpu_solver)

                    # Create the variables, using the bounds for the i-th problem
                    vi = Vector{MOI.VariableIndex}(undef, var_count)
                    for j = 1:var_count
                        vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                    end

                    # Identify the start point within the constraint/RHS fields
                    start = (i-1)*LP_length

                    # Add the constraints
                    for j = 1:LP_length
                        if t.cpu_active_constraint[start+j]
                            @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                        sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                        MOI.GreaterThan(t.cpu_rhs[start+j]))
                        end
                    end

                    # Add the objective function (always already in epigraph form)
                    MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                    MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
                end

                # Optimize
                cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

                # Move answers to lower-bound storage
                cpu_solver_setup += @elapsed begin
                    term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                    if term == MOI.OPTIMAL
                        if t.use_dual_obj
                            t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                        else
                            t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                        end
                        t.CPU_LP_solutions[i,:] .= MOI.get.(t.cpu_solver, MOI.VariablePrimal(), vi)
                    else
                        t.lower_bound_storage[i] = Inf
                    end
                end
            end
        end

        if t.validate_vs_CPU
            data_transfer += @elapsed CUDA.@sync begin
                batchPDLP_termination_reason = hcat(Array(t.PDLP_data[1].termination_reason), Array(t.PDLP_data[2].termination_reason))
                batchPDLP_bound = hcat(Array(t.LP_objectives[1]), Array(t.LP_objectives[2]))
                device!(0)
                t.cpu_constraint_mat[1:Int(end/2),:] .= Array(t.PDLP_data[1].original_problem.constraint_matrix)
                t.cpu_rhs[1:Int(end/2)] .= Array(t.PDLP_data[1].original_problem.right_hand_side)
                t.cpu_active_constraint[1:Int(end/2)] .= Array(t.PDLP_data[1].active_constraint)
                device!(1)
                t.cpu_constraint_mat[Int(end/2)+1:end,:] .= Array(t.PDLP_data[2].original_problem.constraint_matrix)
                t.cpu_rhs[Int(end/2)+1:end] .= Array(t.PDLP_data[2].original_problem.right_hand_side)
                t.cpu_active_constraint[Int(end/2)+1:end] .= Array(t.PDLP_data[2].active_constraint)
            end
            device!(0)
            LP_length = t.PDLP_data[1].dims.total_LP_length
            for i = 1:size(t.cpu_solve_flag)
                cpu_solver_setup += @elapsed begin
                    # Reset the solver
                    MOI.empty!(t.cpu_solver)

                    # Relaxations will be added using an epigraph variable
                    epi = MOI.add_variable(t.cpu_solver)

                    # Create the variables, using the bounds for the i-th problem
                    vi = Vector{MOI.VariableIndex}(undef, var_count)
                    for j = 1:var_count
                        vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                    end

                    # Identify the start point within the constraint/RHS fields
                    start = (i-1)*LP_length

                    # Add the constraints
                    for j = 1:LP_length
                        if t.cpu_active_constraint[start+j]
                            @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                        sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                        MOI.GreaterThan(t.cpu_rhs[start+j]))
                        end
                    end

                    # Add the objective function (always already in epigraph form)
                    MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                    MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
                end

                # Optimize
                cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

                # Compare termination status and results with BatchPDLP
                term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                if term == MOI.OPTIMAL
                    cpu_bound = 0.0
                    if t.use_dual_obj
                        cpu_bound =  MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                    else
                        cpu_bound =  MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                    end
                    if batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_OPTIMAL
                        if !(batchPDLP_termination_reason[i] == BatchPDLP.TERMINATION_REASON_GLOBAL_UPPER_BOUND_HIT ||
                             batchPDLP_termination_reason[i] == BatchPDLP.TERMINATION_REASON_IMPATIENCE)
                             println("For $i: (wrong termination)")
                             println("CPU solver says $term ($cpu_bound)")
                             println("BatchPDLP says $(batchPDLP_termination_reason[i]), ($(batchPDLP_bound[i]))")
                        end
                    elseif !isapprox(cpu_bound, batchPDLP_bound[i], atol=1e-6, rtol=1e-6)
                        if batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_ITERATION_LIMIT
                            println("For $i: (wrong solution)")
                            println("CPU solver says $cpu_bound")
                            println("BatchPDLP says $(batchPDLP_bound[i])")
                        end
                    end
                else
                    if (batchPDLP_bound[i] != Inf) && (batchPDLP_termination_reason[i] != BatchPDLP.TERMINATION_REASON_ITERATION_LIMIT)
                        println("For $i: (wrong termination)")
                        println("CPU solver says $term")
                        println("BatchPDLP says $(batchPDLP_termination_reason[i])")
                    end
                end
            end
        end
    else
        # Only run the CPU solver
        data_transfer += @elapsed CUDA.@sync begin
            device!(0)
            t.cpu_constraint_mat[1:Int(end/2),:] .= Array(t.PDLP_data[1].original_problem.constraint_matrix)
            t.cpu_rhs[1:Int(end/2)] .= Array(t.PDLP_data[1].original_problem.right_hand_side)
            t.cpu_solve_flag[1:Int(end/2)] .= (!).(Array(t.PDLP_data[1].skip_flag))
            t.cpu_active_constraint[1:Int(end/2)] .= Array(t.PDLP_data[1].active_constraint)
            device!(1)
            t.cpu_constraint_mat[Int(end/2)+1:end,:] .= Array(t.PDLP_data[2].original_problem.constraint_matrix)
            t.cpu_rhs[Int(end/2)+1:end] .= Array(t.PDLP_data[2].original_problem.right_hand_side)
            t.cpu_solve_flag[Int(end/2)+1:end] .=  (!).(Array(t.PDLP_data[2].skip_flag))
            t.cpu_active_constraint[Int(end/2)+1:end] .= Array(t.PDLP_data[2].active_constraint)
        end

        LP_length = t.PDLP_data[1].dims.total_LP_length
        for i = 1:size(t.cpu_solve_flag)
            if !t.cpu_solve_flag[i]
                continue
            end
            cpu_solver_setup += @elapsed begin
                # Reset the solver
                MOI.empty!(t.cpu_solver)

                # Relaxations will be added using an epigraph variable
                epi = MOI.add_variable(t.cpu_solver)

                # Create the variables, using the bounds for the i'th problem
                vi = Vector{MOI.VariableIndex}(undef, var_count)
                for j = 1:var_count
                    vi[j], (_,_) = MOI.add_constrained_variable(t.cpu_solver, (MOI.GreaterThan(t.all_lvbs[i,j]), MOI.LessThan(t.all_uvbs[i,j])))
                end

                # Identify the start point within the constraint/RHS fields
                start = (i-1)*LP_length

                # Add the constraints
                for j = 1:LP_length
                    if t.cpu_active_constraint[start+j]
                        @views MOI.add_constraint(t.cpu_solver, t.cpu_constraint_mat[start+j,1]*epi + 
                                                    sum(t.cpu_constraint_mat[start+j,2:end].*vi[1:var_count]),
                                                    MOI.GreaterThan(t.cpu_rhs[start+j]))
                    end
                end

                # Add the objective function (always already in epigraph form)
                MOI.set(t.cpu_solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
                MOI.set(t.cpu_solver, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), 0.0+epi)
            end

            # Optimize
            cpu_solve_time += @elapsed MOI.optimize!(t.cpu_solver)

            # Move answers to lower-bound storage
            cpu_solver_setup += @elapsed begin
                term = MOI.get(t.cpu_solver, MOI.TerminationStatus())
                if term == MOI.OPTIMAL
                    if t.use_dual_obj
                        t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.DualObjectiveValue())
                    else
                        t.lower_bound_storage[i] = MOI.get(t.cpu_solver, MOI.ObjectiveValue())
                    end
                    t.CPU_LP_solutions[i,:] .= MOI.get.(t.cpu_solver, MOI.VariablePrimal(), vi)
                else
                    t.lower_bound_storage[i] = Inf
                end
            end
        end
        misc_setup += @elapsed push!(t.LPs_solved, t.node_len)
    end

    # Push profiling information to timers
    push!(t.timers[2], misc_setup)
    push!(t.timers[3], data_transfer)
    push!(t.timers[4], initial_linearization_setup)
    push!(t.timers[5], calculating_relaxations)
    push!(t.timers[6], multisobol_rescaling)
    push!(t.timers[7], adding_constraints)
    push!(t.timers[8], pdlp_solves)
    push!(t.timers[10], cpu_solver_setup)
    push!(t.timers[11], cpu_solve_time)
    return nothing
end

# Helper function to print an LP constraint matrix for diagnostic purposes. 
# Use by calling, e.g.,:
# print_constraint_matrix(
#       Array(LPs.constraint_matrix), 
#       Array(LPs.right_hand_side), 
#       t.max_parallel_nodes,
#       1)
function print_constraint_matrix(constraint_matrix, right_hand_side, rows_per_LP, LP_to_print)
    display(hcat(constraint_matrix[(LP_to_print-1)*rows_per_LP+1:LP_to_print*rows_per_LP,:], fill(">=", rows_per_LP), right_hand_side[(LP_to_print-1)*rows_per_LP+1:LP_to_print*rows_per_LP]))
end