function prev_rng = seed_rand(seed)
if nargin < 1
    seed = 3;
end
prev_rng = rng;
rng(seed, 'twister')
