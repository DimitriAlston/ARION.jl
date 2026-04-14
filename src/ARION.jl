

module ARION
    using CUDA, DocStringExtensions, LinearAlgebra, Sobol, MultiFloats, MathOptInterface
    using SourceCodeMcCormick, BatchPDLP
    using EAGO, GLPK
    import EAGO: optimize_hook!

    const MOI = MathOptInterface
    
    include("./extension.jl")
    include("./kernels.jl")
    include("./subroutines.jl")

    export PDLP_MultiSobol
end