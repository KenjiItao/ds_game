// =====================================================================
// ds_game_multi_player.cpp
//
// Evolutionary simulation of the M-player resource game (original model).
// C++ counterpart of ds_game_multi_player.py: same model, same parameters,
// same CSV output. Use this for production runs and the Python script for
// small runs and for plotting (python ds_game_multi_player.py --plot-from).
//
// ---------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------
// A population of N groups of M players. Every generation, each group
// plays a game of T steps on its own resource x.
//
// Game (one group, one step t)
//   z_i(t)   = x(t) + S_i M y_i(t) + A_i M <y(t)>   <y>: group mean incl. self
//   h_i(t)   = 1 if z_i(t) > 0 else 0              harvest or not
//   H(t)     = sum_i h_i(t)
//   r(x)     = x + alpha (x - x^2)                 alpha = 1
//   harvest  = min(beta H, 1) r(x)                 beta  = 0.9 / M
//   x(t+1)   = r(x) - harvest
//   p(t)     = harvest / H                         (0 if H = 0)
//   y_i(t+1) = (1 - kappa) y_i(t) + p(t) h_i(t)    kappa = 0.25
//
// Initial state (every generation)
//   x(0) = 0.1,   y_i(0) = (1 + eta_i) / M,   eta_i ~ N(0, 0.1)
//
// Fitness
//   f_i = mean of y_i(t) over t in [0.1 T, T)      (first 10 % discarded)
//
// Reproduction
//   Exactly N M offspring; each offspring draws its parent with probability
//   proportional to f_i (multinomial / Wright-Fisher sampling). Offspring
//   inherit (S, A) plus independent N(0, mu) mutations, mu = 0.1.
//   Players are reassigned to groups at random every generation.
//
// Initial population
//   S_i, A_i ~ N(0, 0.1)
//
// Total fitness Q = M x (mean fitness per player).
//
// ---------------------------------------------------------------------
// Build and run
// ---------------------------------------------------------------------
//   g++ -O3 -std=c++17 -pthread ds_game_multi_player.cpp -o ds_game_multi_player
//
//   ./ds_game_multi_player                          # N = 100, M = 5, trial 0
//   ./ds_game_multi_player --N=100 --M=5 --trials=10 --threads=10
//   ./ds_game_multi_player --gens=500 --steps=500   # quick test
//
// Options (defaults in brackets)
//   --N=<int>        number of groups                      [100]
//   --M=<int>        players per group                     [5]
//   --trials=<int>   number of independent runs            [1]
//   --trial0=<int>   index of the first run                [0]
//   --threads=<int>  runs executed in parallel             [1]
//   --gens=<int>     generations                           [3000]
//   --steps=<int>    game steps per generation, T          [1000]
//   --seed=<int>     extra seed, mixed into every run seed [0]
//   --out=<dir>      output directory                      [Mplayer_out]
//   --no-detail      write only gen_* and summary_* files
//
// Each run (N, M, trial, seed) has its own deterministic seed, so any run
// can be reproduced on its own and runs can be spread over processes with
// --trial0.
//
// ---------------------------------------------------------------------
// Output (in <out>/res/)
// ---------------------------------------------------------------------
//   gen_N{N}_M{M}_trial{t}.csv   one row per generation: Q, mean fitness,
//                                harvesting frequency, mean resource,
//                                mean/sd of S and A, median S/A, fraction
//                                of players with -1.4 < S/A < -1.0
//   snap_N{N}_M{M}_trial{t}.csv  recorded generations: S, A, y_i(0), fitness
//                                and harvest rate of every player
//   dyn_N{N}_M{M}_trial{t}.csv   recorded generations: resource, y and h of
//                                groups 0-2 over the last 300 game steps
//   summary_N{N}_M{M}.csv        one row per run: averages over the last
//                                500 generations
// A group of a recorded generation can be replayed exactly from the
// (S, A, y0) columns of snap_*.csv.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

// =====================================================================
// parameters
// =====================================================================

static int    num_steps       = 1000;   // T
static int    num_generations = 3000;
static double alpha_growth    = 1.0;    // resource growth rate
static double kappa_decay     = 0.25;   // decay rate of y
static double beta_M          = 0.9;    // beta = beta_M / M
static double mutation_rate   = 0.1;    // sd of the mutation of S and A
static double x0_resource     = 0.1;    // x(0) in every generation
static double fit_transient   = 0.1;    // fraction of T excluded from fitness
static double init_sd         = 0.1;    // sd of S and A in generation 0
static int    tail_gens       = 500;    // window of the summary averages

// recorded generations (1-based labels; the last generation is always added)
static std::vector<int> detail_gens = {100, 300, 1000, 3000};
static int  detail_steps  = 300;        // last steps stored per recorded gen.
static int  detail_groups = 3;          // groups stored per recorded gen.
static bool no_detail     = false;

// reference band of the evolved decision rule: -1.4 < S/A < -1.0
static const double band_lo = -1.4;
static const double band_hi = -1.0;

// =====================================================================
// random numbers (one generator per run)
// =====================================================================

static inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

struct Rng
{
    std::mt19937_64 mt;
    std::uniform_real_distribution<double> u{0.0, 1.0};

    double uniform01() { return u(mt); }
    double normal(double mean, double sd)
    {
        if (!(sd > 0.0))
            return mean;
        std::normal_distribution<double> d(mean, sd);
        return d(mt);
    }
};

// =====================================================================
// player and per-generation records
// =====================================================================

struct Player
{
    double S = 0.0, A = 0.0;   // strategy
    double y = 0.0;            // richness
    double y0 = 0.0;           // y at the start of this generation
    double fit = 0.0;          // fitness of this generation
    double harvest = 0.0;      // number of steps harvested
};

struct GenStats
{
    double Q, mean_fitness, sd_fitness, mean_action, mean_resource;
    double mean_S, sd_S, mean_A, sd_A, ratio_med, band_frac;
    double fitness_total;
};

// game trace of the first few groups, values BEFORE the update of each step
struct Trace
{
    int step0 = 0, groups = 0, len = 0, M = 0;
    std::vector<double> x;     // [t][g]
    std::vector<double> y, h;  // [t][g][seat]
    double &X(int t, int g) { return x[t * groups + g]; }
    double &Y(int t, int g, int k) { return y[(t * groups + g) * M + k]; }
    double &Hh(int t, int g, int k) { return h[(t * groups + g) * M + k]; }
};

static double median(std::vector<double> v)
{
    if (v.empty())
        return std::nan("");
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// =====================================================================
// one generation of the game (no reproduction)
// players.size() must be N * M; group g consists of players g*M ... g*M+M-1
// =====================================================================

static GenStats play_generation(std::vector<Player> &players, int M, Rng &rng,
                                Trace *trace)
{
    const int n = static_cast<int>(players.size());
    const int groups = n / M;
    const double beta = beta_M / M;
    const double keep = 1.0 - kappa_decay;
    const int fit_from = std::min(
        std::max(0, static_cast<int>(std::llround(fit_transient * num_steps))),
        num_steps - 1);
    const int fit_len = num_steps - fit_from;

    const double inv_M = 1.0 / M;
    for (Player &p : players)
    {
        p.y0 = (1.0 + rng.normal(0.0, 0.1)) * inv_M;
        p.y = p.y0;
        p.harvest = 0.0;
    }

    if (trace)
    {
        trace->M = M;
        trace->groups = std::min(detail_groups, groups);
        trace->step0 = std::max(0, num_steps - detail_steps);
        trace->len = num_steps - trace->step0;
        trace->x.assign(static_cast<size_t>(trace->len) * trace->groups, 0.0);
        trace->y.assign(trace->x.size() * M, 0.0);
        trace->h = trace->y;
    }

    std::vector<double> resource(groups, x0_resource);
    std::vector<double> fit_sum(n, 0.0);
    std::vector<double> h(M, 0.0);
    double action_sum = 0.0, resource_sum = 0.0;

    for (int step = 0; step < num_steps; ++step)
    {
        for (int g = 0; g < groups; ++g)
        {
            const int base = g * M;
            double y_sum = 0.0;
            for (int i = 0; i < M; ++i)
                y_sum += players[base + i].y;
            const double peer = static_cast<double>(M) * y_sum / M;  // M <y>
            const double x = resource[g];

            // decisions
            double H = 0.0;
            for (int i = 0; i < M; ++i)
            {
                const Player &p = players[base + i];
                const double z = x + p.S * M * p.y + p.A * peer;
                h[i] = z > 0.0 ? 1.0 : 0.0;
                H += h[i];
            }

            // resource growth and harvest
            const double grown = x + alpha_growth * (x - x * x);
            const double harvested = std::min(beta * H, 1.0) * grown;
            const double per_unit = H > 0.0 ? harvested / H : 0.0;
            resource[g] = grown - harvested;

            // richness update
            const bool rec = trace && g < trace->groups && step >= trace->step0;
            for (int i = 0; i < M; ++i)
            {
                Player &p = players[base + i];
                if (step >= fit_from)
                    fit_sum[base + i] += p.y;    // y(t) before the update
                p.harvest += h[i];
                if (rec)
                {
                    const int t = step - trace->step0;
                    trace->Y(t, g, i) = p.y;
                    trace->Hh(t, g, i) = h[i];
                    if (i == 0)
                        trace->X(t, g) = x;
                }
                p.y = keep * p.y + per_unit * h[i];
            }
            action_sum += H;
            resource_sum += resource[g];
        }
    }

    // fitness and summary statistics
    GenStats s{};
    double f_tot = 0.0, f_sq = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double f = fit_sum[i] / fit_len;
        if (!(f > 0.0) || !std::isfinite(f))
            f = 0.0;
        players[i].fit = f;
        f_tot += f;
        f_sq += f * f;
    }
    s.fitness_total = f_tot;
    s.mean_fitness = f_tot / n;
    s.sd_fitness = std::sqrt(std::max(0.0, f_sq / n - s.mean_fitness * s.mean_fitness));
    s.Q = M * s.mean_fitness;
    s.mean_action = action_sum / (static_cast<double>(num_steps) * n);
    s.mean_resource = resource_sum / (static_cast<double>(num_steps) * groups);

    double sS = 0, sS2 = 0, sA = 0, sA2 = 0;
    int band = 0;
    std::vector<double> ratios;
    ratios.reserve(n);
    for (const Player &p : players)
    {
        sS += p.S;
        sS2 += p.S * p.S;
        sA += p.A;
        sA2 += p.A * p.A;
        if (std::fabs(p.A) > 1.0e-12)
        {
            const double r = p.S / p.A;
            ratios.push_back(r);
            if (r > band_lo && r < band_hi)
                ++band;
        }
    }
    s.mean_S = sS / n;
    s.sd_S = std::sqrt(std::max(0.0, sS2 / n - s.mean_S * s.mean_S));
    s.mean_A = sA / n;
    s.sd_A = std::sqrt(std::max(0.0, sA2 / n - s.mean_A * s.mean_A));
    s.band_frac = static_cast<double>(band) / n;
    s.ratio_med = median(ratios);
    return s;
}

// =====================================================================
// reproduction: exactly n offspring, parents drawn in proportion to fitness
// =====================================================================

static void reproduce(std::vector<Player> &players, Rng &rng)
{
    const int n = static_cast<int>(players.size());
    std::vector<double> cum(n);
    double run = 0.0;
    for (int i = 0; i < n; ++i)
    {
        run += players[i].fit;
        cum[i] = run;
    }
    std::vector<Player> next(n);
    for (int j = 0; j < n; ++j)
    {
        // if every fitness is zero, parents are drawn uniformly
        const int i =
            run > 0.0
                ? std::min(static_cast<int>(std::lower_bound(cum.begin(), cum.end(),
                                                             rng.uniform01() * run) -
                                            cum.begin()),
                           n - 1)
                : static_cast<int>(rng.uniform01() * n) % n;
        next[j].S = players[i].S + rng.normal(0.0, mutation_rate);
        next[j].A = players[i].A + rng.normal(0.0, mutation_rate);
    }
    players.swap(next);
}

// =====================================================================
// one run
// =====================================================================

struct Summary
{
    int N, M, trial;
    double Q, S, A, ratio_med, band, action, resource;
};

static std::string tag_of(int N, int M, int trial)
{
    std::ostringstream o;
    o << "N" << N << "_M" << M << "_trial" << trial;
    return o.str();
}

static Summary run_trial(int N, int M, int trial, uint64_t seed,
                         const std::string &res_dir, std::mutex &io)
{
    Rng rng;
    rng.mt.seed(splitmix64(0x0902F16200000000ULL ^ (seed << 48) ^
                           (static_cast<uint64_t>(N) << 32) ^
                           (static_cast<uint64_t>(M) << 16) ^
                           static_cast<uint64_t>(trial)));

    std::vector<Player> players(static_cast<size_t>(N) * M);
    for (Player &p : players)
    {
        p.S = rng.normal(0.0, init_sd);
        p.A = rng.normal(0.0, init_sd);
    }

    // recorded generations: 0-based index -> 1-based label
    std::map<int, int> rec;
    for (int g : detail_gens)
    {
        const int lab = std::min(std::max(1, g), num_generations);
        rec[lab - 1] = lab;
    }
    rec[num_generations - 1] = num_generations;

    const std::string tag = tag_of(N, M, trial);
    std::ofstream fgen(res_dir + "/gen_" + tag + ".csv");
    std::ofstream fsnap, fdyn;
    fgen << std::setprecision(8);
    fgen << "generation,Q,mean_fitness,sd_fitness,mean_action,mean_resource,"
            "mean_S,sd_S,mean_A,sd_A,ratio_med,band_frac\n";
    if (!no_detail)
    {
        fsnap.open(res_dir + "/snap_" + tag + ".csv");
        fdyn.open(res_dir + "/dyn_" + tag + ".csv");
        fsnap << std::setprecision(10);
        fdyn << std::setprecision(10);
        fsnap << "generation,label,M,index,group,seat,S,A,y0,fitness,harvest_rate\n";
        fdyn << "generation,label,group,step,resource,seat,y,h\n";
    }

    double tQ = 0, tS = 0, tA = 0, tband = 0, tact = 0, tres = 0;
    std::vector<double> tratio;
    int tn = 0;

    for (int gen = 0; gen < num_generations; ++gen)
    {
        std::shuffle(players.begin(), players.end(), rng.mt);   // regroup
        const bool is_rec = !no_detail && rec.count(gen) > 0;
        Trace trace;
        const GenStats s = play_generation(players, M, rng, is_rec ? &trace : nullptr);

        fgen << gen << "," << s.Q << "," << s.mean_fitness << "," << s.sd_fitness
             << "," << s.mean_action << "," << s.mean_resource << "," << s.mean_S
             << "," << s.sd_S << "," << s.mean_A << "," << s.sd_A << ","
             << s.ratio_med << "," << s.band_frac << "\n";

        if (is_rec)
        {
            const int lab = rec[gen];
            for (size_t i = 0; i < players.size(); ++i)
            {
                const Player &p = players[i];
                fsnap << gen << "," << lab << "," << M << "," << i << "," << i / M
                      << "," << i % M << "," << p.S << "," << p.A << "," << p.y0
                      << "," << p.fit << "," << p.harvest / num_steps << "\n";
            }
            for (int g = 0; g < trace.groups; ++g)
                for (int t = 0; t < trace.len; ++t)
                    for (int k = 0; k < M; ++k)
                        fdyn << gen << "," << lab << "," << g << ","
                             << trace.step0 + t << "," << trace.X(t, g) << "," << k
                             << "," << trace.Y(t, g, k) << ","
                             << static_cast<int>(trace.Hh(t, g, k)) << "\n";
        }

        if (gen >= num_generations - tail_gens)
        {
            ++tn;
            tQ += s.Q;
            tS += s.mean_S;
            tA += s.mean_A;
            tband += s.band_frac;
            tact += s.mean_action;
            tres += s.mean_resource;
            if (std::isfinite(s.ratio_med))
                tratio.push_back(s.ratio_med);
        }
        if (gen % 500 == 0 || gen == num_generations - 1)
        {
            std::lock_guard<std::mutex> lk(io);
            std::cout << "  N=" << N << " M=" << M << " trial=" << trial
                      << " gen=" << std::setw(5) << gen << "  Q=" << std::fixed
                      << std::setprecision(4) << s.Q << "  S=" << std::showpos
                      << std::setprecision(3) << s.mean_S << "  A=" << s.mean_A
                      << std::noshowpos << std::defaultfloat << "\n";
        }

        reproduce(players, rng);
    }

    const double k = tn ? 1.0 / tn : std::nan("");
    return {N, M, trial, tQ * k, tS * k, tA * k, median(tratio), tband * k,
            tact * k, tres * k};
}

// =====================================================================
// main
// =====================================================================

static std::vector<int> parse_ints(const std::string &s)
{
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty())
            v.push_back(std::atoi(tok.c_str()));
    return v;
}

int main(int argc, char **argv)
{
    int N = 100, M = 5, trials = 1, trial0 = 0, threads = 1;
    uint64_t seed = 0;
    std::string out = "Mplayer_out";

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto val = [&](const char *key) -> const char * {
            const size_t n = std::string(key).size();
            return a.compare(0, n, key) == 0 ? a.c_str() + n : nullptr;
        };
        if (const char *v = val("--N=")) N = std::atoi(v);
        else if (const char *v = val("--M=")) M = std::atoi(v);
        else if (const char *v = val("--trials=")) trials = std::atoi(v);
        else if (const char *v = val("--trial0=")) trial0 = std::atoi(v);
        else if (const char *v = val("--threads=")) threads = std::max(1, std::atoi(v));
        else if (const char *v = val("--gens=")) num_generations = std::atoi(v);
        else if (const char *v = val("--steps=")) num_steps = std::atoi(v);
        else if (const char *v = val("--seed=")) seed = std::strtoull(v, nullptr, 10);
        else if (const char *v = val("--out=")) out = v;
        else if (const char *v = val("--detail-gens=")) detail_gens = parse_ints(v);
        else if (a == "--no-detail") no_detail = true;
        else
        {
            std::cerr << "unknown option: " << a
                      << "\nsee the header of ds_game_multi_player.cpp for usage\n";
            return 1;
        }
    }
    if (N < 1 || M < 1 || num_steps < 1 || num_generations < 1)
    {
        std::cerr << "N, M, --steps and --gens must be positive\n";
        return 1;
    }

    const std::string res_dir = out + "/res";
    mkdir(out.c_str(), 0755);
    mkdir(res_dir.c_str(), 0755);

    // runs are distributed over the threads; each run has its own Rng
    std::vector<Summary> results(trials);
    std::mutex io;
    std::vector<std::thread> pool;
    for (int w = 0; w < std::min(threads, trials); ++w)
        pool.emplace_back([&, w]() {
            for (int j = w; j < trials; j += threads)
                results[j] = run_trial(N, M, trial0 + j, seed, res_dir, io);
        });
    for (std::thread &t : pool)
        t.join();

    const std::string spath = res_dir + "/summary_N" + std::to_string(N) + "_M" +
                              std::to_string(M) + ".csv";
    const bool is_new = !std::ifstream(spath).good();
    std::ofstream fs(spath, std::ios::app);
    fs << std::setprecision(8);
    if (is_new)
        fs << "N,M,trial,Q_tail,S_tail,A_tail,ratio_med_tail,band_tail,"
              "action_tail,resource_tail\n";
    for (const Summary &r : results)
    {
        fs << r.N << "," << r.M << "," << r.trial << "," << r.Q << "," << r.S << ","
           << r.A << "," << r.ratio_med << "," << r.band << "," << r.action << ","
           << r.resource << "\n";
        std::cout << "trial " << r.trial << ": Q_tail=" << std::fixed
                  << std::setprecision(4) << r.Q << "  S_tail=" << std::showpos
                  << std::setprecision(3) << r.S << "  A_tail=" << r.A
                  << "  S/A=" << r.ratio_med << std::noshowpos << std::defaultfloat
                  << "\n";
    }
    return 0;
}
