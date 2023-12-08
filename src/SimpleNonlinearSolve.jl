module SimpleNonlinearSolve

import PrecompileTools: @compile_workload, @setup_workload, @recompile_invalidations

@recompile_invalidations begin
    using ADTypes,
        ArrayInterface, ConcreteStructs, DiffEqBase, Reexport, LinearAlgebra, SciMLBase

    import DiffEqBase: AbstractNonlinearTerminationMode,
        AbstractSafeNonlinearTerminationMode, AbstractSafeBestNonlinearTerminationMode,
        NonlinearSafeTerminationReturnCode, get_termination_mode,
        NONLINEARSOLVE_DEFAULT_NORM
    using FiniteDiff, ForwardDiff
    import ForwardDiff: Dual
    import MaybeInplace: @bb, setindex_trait, CanSetindex, CannotSetindex
    import SciMLBase: AbstractNonlinearAlgorithm, build_solution, isinplace
    import StaticArraysCore: StaticArray, SVector, SMatrix, SArray, MArray, MMatrix, Size
end

@reexport using ADTypes, SciMLBase

abstract type AbstractSimpleNonlinearSolveAlgorithm <: AbstractNonlinearAlgorithm end
abstract type AbstractBracketingAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end
abstract type AbstractNewtonAlgorithm <: AbstractSimpleNonlinearSolveAlgorithm end

include("utils.jl")

## Nonlinear Solvers
include("nlsolve/raphson.jl")
include("nlsolve/broyden.jl")
include("nlsolve/lbroyden.jl")
include("nlsolve/klement.jl")
include("nlsolve/trustRegion.jl")
include("nlsolve/halley.jl")
include("nlsolve/dfsane.jl")
include("nlsolve/extension_algs.jl")

## Interval Nonlinear Solvers
include("bracketing/bisection.jl")
include("bracketing/falsi.jl")
include("bracketing/ridder.jl")
include("bracketing/brent.jl")
include("bracketing/alefeld.jl")
include("bracketing/itp.jl")

# AD
include("ad.jl")

## Default algorithm

# Set the default bracketing method to ITP
function SciMLBase.solve(prob::IntervalNonlinearProblem; kwargs...)
    return solve(prob, ITP(); kwargs...)
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Nothing,
        args...; kwargs...)
    return solve(prob, ITP(), args...; kwargs...)
end

@setup_workload begin
    for T in (Float32, Float64)
        prob_no_brack_scalar = NonlinearProblem{false}((u, p) -> u .* u .- p, T(0.1), T(2))
        prob_no_brack_iip = NonlinearProblem{true}((du, u, p) -> du .= u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))
        prob_no_brack_oop = NonlinearProblem{false}((u, p) -> u .* u .- p,
            T.([1.0, 1.0, 1.0]), T(2))

        algs = [SimpleNewtonRaphson(), SimpleBroyden(), SimpleKlement(), SimpleDFSane(),
            SimpleTrustRegion(), SimpleLimitedMemoryBroyden(; threshold = 2)]

        algs_no_iip = [SimpleHalley()]

        @compile_workload begin
            for alg in algs
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_iip, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end

            for alg in algs_no_iip
                solve(prob_no_brack_scalar, alg, abstol = T(1e-2))
                solve(prob_no_brack_oop, alg, abstol = T(1e-2))
            end
        end

        prob_brack = IntervalNonlinearProblem{false}((u, p) -> u * u - p,
            T.((0.0, 2.0)), T(2))
        algs = [Bisection(), Falsi(), Ridder(), Brent(), Alefeld(), ITP()]
        @compile_workload begin
            for alg in algs
                solve(prob_brack, alg, abstol = T(1e-2))
            end
        end
    end
end

export SimpleBroyden, SimpleDFSane, SimpleGaussNewton, SimpleHalley, SimpleKlement,
    SimpleLimitedMemoryBroyden, SimpleNewtonRaphson, SimpleTrustRegion
export SimpleJNFK # Extension
export Alefeld, Bisection, Brent, Falsi, ITP, Ridder

end # module
