"""
    SimpleJFNK()

Low overhead Jacobian-free Newton-Krylov method. Uses Jacobian-Free Linear Solve using
`Krylov.jl` `gmres!` routine.

`JFNK` usually requires pre-conditioning to work well. This is unfortunately not supported
here to keep the implementation simple. Users are recommended to use
`NewtonRaphson(; linsolve = KrylovJL_GMRES())` from `NonlinearSolve.jl` instead for most
workflows.

Additionally, this method uses `ForwardDiff` to compute the `JVP` operator. We will
currently ignore `jvp` passed in by the user. This might be supported in the future.

!!! note

    This algorithm is only available if `Krylov.jl` is installed and loaded.
"""
struct SimpleJNFK <: AbstractNewtonAlgorithm
    function SimpleJNFK()
        if Base.get_extension(@__MODULE__, :SimpleNonlinearSolveKrylovExt) === nothing
            error("SimpleJNFK requires Krylov.jl to be loaded")
        end
        return new()
    end
end
