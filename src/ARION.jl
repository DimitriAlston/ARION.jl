

module ARION
    using CUDA, DocStringExtensions, LinearAlgebra, Sobol, MultiFloats, Printf
    using SourceCodeMcCormick, BatchPDLP
    using EAGO, GLPK
    import EAGO: optimize_hook!
    import MathOptInterface as MOI

    include("./extension.jl")
    include("./kernels.jl")
    include("./subroutines.jl")

    export GroupMethod, GroupMethod_MultiGPU, KelleyMethod
    export Problem, LoadedProblem
end