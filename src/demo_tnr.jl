
using Distributions

iter = 100
lr = 0.00001
d = 100
n = 500
T = 10
trueRank = convert(Int, round(min(d, T) / 2))
lambda = 0.01


function main()
  trueW = generateLowRankMatrix(d, T, trueRank)
  X, Y = generateSynthetic(trueW, n)
  W = randn(d, T)  # initialize
  for i in range(0, iter)
    println("$i th iteration...")
    W = proxGrad(W, X, Y, lambda, lr)
    diff = norm(W - trueW)
    println("|| W - trueW || = $diff")
  end
  println("Finish!")
end


function generateLowRankMatrix(d1::Int, d2::Int, rank::Int)
  @assert min(d1, d2) > rank
  U = randn(d1, rank)
  V = randn(d2, rank)
  # noise = rand(Normal(0, 1), d1, d2)
  noise = randn(d1, d2)
  return U * V' + noise
end


function generateSynthetic(W, num::Int)
  X = randn(num, size(W, 1))
  Y = X * W + randn(num, size(W, 2))
  return X, Y
end


function proxGrad(W, X, Y, lambda, eta)
  grad = X' * (X * W - Y)
  return proxTraceNorm(W - eta * grad, lambda)
end


function proxTraceNorm(W, lambda)
  U, sigma, V = svd(W)
  return U * Diagonal(max(sigma - lambda, 0)) * V'
end

main()
