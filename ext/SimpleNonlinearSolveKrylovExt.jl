module SimpleNonlinearSolveKrylovExt

using Krylov, SciMLBase, SimpleNonlinearSolve
import ConcreteStructs: @concrete
import ForwardDiff: Dual, Partials, Tag, partials

# JVP Operator
struct SimpleJNFKJacVecTag end

function jvp_forwarddiff(f, x::AbstractArray{T}, v) where {T}
    v_ = reshape(v, axes(x))
    y = (Dual{Tag{SimpleJNFKJacVecTag, T}, T, 1}).(x, Partials.(tuple.(v_)))
    return vec(partials.(vec(f(y)), 1))
end
jvp_forwarddiff!(r, f, x, v) = copyto!(r, jvp_forwarddiff(f, x, v))


end