# Given points and domains:
# P = `[eval_points[:,var], lvbs[:,var], uvbs[:,var]]`
# as well as linearization points to use from the Sobol sequence:
# L = `linearization_points[1:n_points,var]`
# Fill in `input_storage` to be `n_points`-times longer than
# `eval_points` with the linearization points used in place of
# the initial `eval_points[:,1]`.
function load_linearization_points_sobol_kernel(input_storage, eval_points, lvbs, uvbs, n_LPs, linearization_points, n_points, var)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_LPs
        point = Int32(1)
        diff = (uvbs[idx,var] - lvbs[idx,var])
        while point <= n_points
            eval_points[(idx - Int32(1))*n_points + point, var] = linearization_points[point,var] * diff + lvbs[idx,var]
            input_storage[(idx - Int32(1))*n_points + point, Int32(1)] = linearization_points[point,var] * diff + lvbs[idx,var]
            input_storage[(idx - Int32(1))*n_points + point, Int32(2)] = lvbs[idx,var]
            input_storage[(idx - Int32(1))*n_points + point, Int32(3)] = uvbs[idx,var]
            point += Int32(1)
        end
        idx += stride
    end
    return nothing
end

# Identify a region likely containing a minimum and scale points
# and bounds to the new region
function rescale_linearization_points_kernel(input_storage, eval_points, result_storage, curr_lvbs, curr_uvbs, n_LPs, points_per_LP, var)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_LPs
        # For the objective function, we want to identify a region that
        # possibly contains the minimum. We can do this by checking each
        # variable for the latest point with a negative subgradient and
        # the earliest point with a positive subgradient. In the simplest
        # case of a symmetric relaxation with a unique minimum, this method
        # will identify the correct region. If it's not this simple case,
        # we can still learn useful information.
        point = Int32(1)
        LBD = -Inf
        UBD = Inf
        while point <= points_per_LP
            pt = input_storage[(idx - Int32(1))*points_per_LP + point, Int32(1)]
            subgrad = result_storage[(idx - Int32(1))*points_per_LP + point, Int32(4) + var]
            if subgrad > 0 && pt < UBD
                UBD = pt
            elseif subgrad < 0 && pt > LBD
                LBD = pt
            end
            point += Int32(1)
        end

        # If LBD > UBD, we have a less trivial case. This can happen if, e.g.,
        # minima are found in a line/band that is diagonal relative to a variable
        # of interest
        if LBD > UBD
            temp = UBD
            UBD = LBD
            LBD = temp
        end

        # Save old bounds
        old_lbd = curr_lvbs[idx, var]
        old_ubd = curr_uvbs[idx, var]
        
        # Determine new bounds
        LBD = max(LBD, old_lbd)
        UBD = min(UBD, old_ubd)

        # Put the new bounds in the curr_*vbs storage
        curr_lvbs[idx, var] = LBD
        curr_uvbs[idx, var] = UBD

        # Now apply the subgradient information to rescale the input storage (and eval points)
        point = Int32(1)
        while point <= points_per_LP
            curr = (idx - Int32(1))*points_per_LP + point
            if old_lbd == old_ubd
                new_pt = old_lbd
            else
                new_pt = ((input_storage[curr, Int32(1)] - old_lbd)/(old_ubd - old_lbd))*(UBD - LBD) + LBD
            end
            eval_points[curr, var] = new_pt
            input_storage[curr, Int32(1)] = new_pt
            point += Int32(1)
        end
        idx += stride
    end
    return nothing
end

# Given that rescaling has already occurred, scale linearization
# points back to the original domain in preparation for other
# relaxation calculations
function scale_back_linearization_points_kernel(input_storage, eval_points, curr_lvbs, curr_uvbs, n_LPs, points_per_LP, var)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_LPs
        # We've scaled input_storage and eval_points. Let's scale them back to what
        # they were before.
        point = Int32(1)
        old_lbd = curr_lvbs[idx, var]
        old_ubd = curr_uvbs[idx, var]
        LBD = input_storage[(idx - Int32(1))*points_per_LP + point, Int32(2)]
        UBD = input_storage[(idx - Int32(1))*points_per_LP + point, Int32(3)]
        while point <= points_per_LP
            curr = (idx - Int32(1))*points_per_LP + point
            if old_lbd == old_ubd
                new_pt = old_lbd
            else
                new_pt = ((input_storage[curr, Int32(1)] - old_lbd)/(old_ubd - old_lbd))*(UBD - LBD) + LBD
            end
            eval_points[curr, var] = new_pt
            input_storage[curr, Int32(1)] = new_pt
            point += Int32(1)
        end
        idx += stride
    end
    return nothing
end

# For the objective function relaxations, store the sum of the subgradient elements squared
function sum_subgradients(subgradient_checksum, lvbs, uvbs, result_storage, n_points, points_per_LP, n_vars, n_blocks)
    CUDA.@sync @cuda blocks=n_blocks threads=1024 sum_subgradients_kernel(subgradient_checksum, lvbs, uvbs, result_storage, n_points, points_per_LP, n_vars)
    return nothing
end
function sum_subgradients_kernel(subgradient_checksum, lvbs, uvbs, result_storage, n_points, points_per_LP, n_vars)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_points
        var = Int32(1)
        LP = Int32(floor((idx-1)/points_per_LP)+1)
        subgradient_checksum[idx] = 0.0
        while var <= n_vars
            subgradient_checksum[idx] += result_storage[idx, 4+var]^2 * (uvbs[LP, var] - lvbs[LP, var])
            var += Int32(1)
        end
        idx += stride
    end
    return nothing
end

# Given a results matrix, extract information summarizing the signs of
# subgradient elements.
function extract_subgradient_sign(subgradient_checksum, result_storage, n_points, n_vars, n_blocks; concave::Bool=false)
    CUDA.@sync @cuda blocks=n_blocks threads=1024 extract_subgradient_sign_kernel(subgradient_checksum, result_storage, n_points, n_vars, concave)
    return nothing
end
function extract_subgradient_sign_kernel(subgradient_checksum, result_storage, n_points, n_vars, concave::Bool)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_points
        var = Int32(1)
        subgradient_checksum[idx] = 0.0
        while var <= n_vars
            if result_storage[idx, 4+var+(concave*n_vars)] < 0
                subgradient_checksum[idx] += 3^(var-1)
            elseif result_storage[idx, 4+var+(concave*n_vars)] > 0
                subgradient_checksum[idx] += 2*3^(var-1)
            end
            var += Int32(1)
        end
        idx += stride
    end
    return nothing
end

# For constraints, use the convex or concave relaxation values for comparisons
function extract_convex_relaxation(comparison_vector, result_storage, n_points, n_blocks)
    CUDA.@sync @cuda blocks=n_blocks threads=1024 extract_convex_relaxation_kernel(comparison_vector, result_storage, n_points)
    return nothing
end
function extract_convex_relaxation_kernel(comparison_vector, result_storage, n_points)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_points
        comparison_vector[idx] = result_storage[idx, Int32(1)]
        idx += stride
    end
    return nothing
end
function extract_concave_relaxation(comparison_vector, result_storage, n_points, n_blocks)
    CUDA.@sync @cuda blocks=n_blocks threads=1024 extract_concave_relaxation_kernel(comparison_vector, result_storage, n_points)
    return nothing
end
function extract_concave_relaxation_kernel(comparison_vector, result_storage, n_points)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_points
        comparison_vector[idx] = result_storage[idx, Int32(2)]
        idx += stride
    end
    return nothing
end

# For KelleyMethod, load in the midpoint for each variable, for each node
function load_midpoints_kernel(input_storage, eval_points, lvbs, uvbs, n_LPs, var)
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x
    while idx <= n_LPs
        eval_points[idx,var] = (lvbs[idx,var] + uvbs[idx,var])/2
        input_storage[idx,Int32(1)] = (lvbs[idx,var] + uvbs[idx,var])/2
        input_storage[idx,Int32(2)] = lvbs[idx,var]
        input_storage[idx,Int32(3)] = uvbs[idx,var]
        idx += stride
    end
    return nothing
end

# For KelleyMethod, offset solutions by 5% away from the bounds
function bound_offset_kernel(
    LP_solutions,
    input_storage,
    eval_points,
    lvbs,
    uvbs,
    len::Int32,
    var::Int32;
    offset::Float64 = 0.05 #0.05
    )
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x

    while idx <= len
        diff = (uvbs[idx,var] - lvbs[idx,var])
        result = max(lvbs[idx,var] + offset*diff, 
                    min(uvbs[idx,var] - offset*diff,
                    LP_solutions[idx,var]))
        eval_points[idx,var] = result
        input_storage[idx,Int32(1)] = result
        idx += stride
    end
    return nothing
end

# A kernel to remove NaN/Inf rows from the constraint matrix/RHS
function remove_NaNInf_kernel(
    constraint_matrix,
    right_hand_side,
    active_constraint,
    total_rows,
    matrix_width,
    )
    idx = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    stride = blockDim().x * gridDim().x

    while idx <= total_rows
        # Scan through the row
        flag = isnan(right_hand_side[idx]) || isinf(right_hand_side[idx])
        col = Int32(1)
        while col <= matrix_width
            flag |= (isnan(constraint_matrix[idx,col]) || isinf(constraint_matrix[idx,col]))
            flag && break
            col += Int32(1)
        end

        # If the flag is true, the whole row should get set to zeros
        if flag
            active_constraint[idx] = false
            right_hand_side[idx] = 0.0
            col = Int32(1)
            while col <= matrix_width
                constraint_matrix[idx,col] = 0.0
                col += Int32(1)
            end
        end
        idx += stride
    end
    return nothing
end