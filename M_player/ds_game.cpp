#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <map>
#include <tuple>
#include <string>
#include <set>
#include <cstdlib>
using namespace std;
//==================================================
// グローバル設定
//==================================================

// 1世代あたりのステップ数
static int num_steps = 1000;
// 世代数
static int num_generations = 3000;

// どの世代のステップ履歴を記録するか（必要に応じて変更）
//static std::set<int> record_generations = {30, 100, 300, 1000, 3000};

//==================================================
// 乱数生成
//==================================================
std::random_device rd;
std::mt19937 mt(rd());

// template <class T>
// void print(const T &value)
// {
//     std::cout << value << std::endl;
// }

// 正規分布乱数 (stddev<=0 の場合は常に mean を返す)
inline double normal_random(double mean, double stddev)
{
    if (stddev <= 0.0)
        return mean;
    std::normal_distribution<double> dist(mean, std::fabs(stddev));
    return dist(mt);
}

// ポアソン分布乱数
inline int poisson_random(double lambda)
{
    std::poisson_distribution<int> dist(lambda);
    return dist(mt);
}

//==================================================
// プレイヤークラス
//==================================================
// 各プレイヤーは内部状態 state、
// 自分自身への影響重み S、
// 他者平均への重み A、
// 他者標準偏差への重み V、
// および適応度 fitness を持つ。
class Player
{
public:
    double state;   // 内部状態
    double S;       // 自分の状態への影響重み
    double A;       // 他者平均への重み
    double fitness; // 適応度

    // コンストラクタ: 内部状態は平均1.0, stddev=0.1 で初期化
    Player(double S_in, double A_in)
        : state(normal_random(1.0, 0.1)), S(S_in), A(A_in), fitness(0.0)
    {
    }
};

//==================================================
// payoff 関数
//==================================================
// 各グループにおいて、まず資源を以下のように変換します:
//    transformed = 2.0 * resource - resource * resource;
// 次に、グループ内で行動したプレイヤー数 total_actions に対し、
// 収穫割合として「total_actions × beta」を適用します。
// ただし harvested の上限は transformed（すなわち100%収穫）とし、
// 収穫した総量を行動者で均等分配し、残りを次ターンの資源とします。
inline void payoff(double &resource, int total_actions, double beta, double &payoff_per_actor)
{
    double transformed = 2.0 * resource - resource * resource;
    if (total_actions > 0)
    {
        double harvested = total_actions * beta * transformed;
//        double harvested = transformed * (1.0 - 1.0 / pow(3.0, total_actions));
        payoff_per_actor = harvested / total_actions;
        resource = transformed - harvested;
    }
    else
    {
        payoff_per_actor = 0.0;
        resource = transformed;
    }
}

//==================================================
// 1世代分のゲーム実行と再生産（グループ版）
// np人のプレイヤーをグループサイズ group_size (m) ごとに分けてゲームを実施する。
// ※ここで、世代開始時に players をランダムシャッフルし、
//    最後のグループが m 人未満なら players の先頭から補填して m 人にする。
// 戻り値は (aveAction, aveResource, aveFitness, mean_S, mean_A, mean_V, phaseVal)
//==================================================
#include <sstream>   // 追加: stringstream を使用するため
double average_synchrony(const std::vector<std::vector<int>>& action_ls) {
    int M = action_ls.size();
    if (M < 2) return 1.0; // 個体数が1の場合は完全に同期しているとみなす
    double sum_similarity = 0.0;
    int count = 0;
    int L = action_ls[0].size();  // 全て同じ長さであると仮定
    for (int i = 0; i < M - 1; i++) {
        for (int j = i + 1; j < M; j++) {
            int matching = 0;
            for (int k = 0; k < L; k++) {
                if (action_ls[i][k] == action_ls[j][k])
                    matching++;
            }
            double similarity = static_cast<double>(matching) / L;
            sum_similarity += similarity;
            count++;
        }
    }
    return sum_similarity / count;
}

// 定数の定義
const int UNCLASSIFIED = -2;
const int NOISE = -1;

// 2つの系列間のハミング距離（＝1 - 一致率）を計算
double hamming_distance(const vector<int>& x, const vector<int>& y) {
    if (x.size() != y.size()) return 1.0; // サイズが異なる場合は最大距離とする
    int count = 0;
    int len = x.size();
    for (int i = 0; i < len; i++) {
        if (x[i] == y[i]) count++;
    }
    double match_rate = static_cast<double>(count) / len;
    return 1.0 - match_rate;
}

// 指定した点 index の近傍のインデックス（距離が eps 以下）を返す
vector<int> regionQuery(int index, const vector<vector<double>>& D, double eps) {
    vector<int> neighbors;
    int M = D.size();
    for (int j = 0; j < M; j++) {
        if (D[index][j] <= eps)
            neighbors.push_back(j);
    }
    return neighbors;
}

// --- DBSCAN --- //
// 距離行列 D を用いて DBSCAN を実行する関数
vector<int> dbscan(const vector<vector<double>>& D, double eps, int min_samples) {
    int M = D.size();
    vector<int> labels(M, UNCLASSIFIED);
    int cluster_id = 0;
    for (int i = 0; i < M; i++) {
        if (labels[i] != UNCLASSIFIED)
            continue;
        vector<int> neighbors = regionQuery(i, D, eps);
        if (neighbors.size() < static_cast<size_t>(min_samples)) {
            labels[i] = NOISE;
        } else {
            cluster_id++;
            labels[i] = cluster_id;
            vector<int> seeds = neighbors;
            // 拡大処理
            for (size_t idx = 0; idx < seeds.size(); idx++) {
                int j = seeds[idx];
                if (labels[j] == NOISE)
                    labels[j] = cluster_id;
                if (labels[j] != UNCLASSIFIED)
                    continue;
                labels[j] = cluster_id;
                vector<int> j_neighbors = regionQuery(j, D, eps);
                if (j_neighbors.size() >= static_cast<size_t>(min_samples)) {
                    // 新たな隣接点をシードリストに追加（重複は問題にならない）
                    seeds.insert(seeds.end(), j_neighbors.begin(), j_neighbors.end());
                }
            }
        }
    }
    return labels;
}

// --- ラベル再割り当て --- //
// DBSCANなどで得られたラベル配列を変換する関数
// ・クラスタ（-1以外）の各グループは 1 から連番に再割り当て
// ・ノイズ（-1）の各点は個別のラベルに変更
// 例: [-1, -1, 0, 0, 0] → [2, 3, 1, 1, 1]
vector<int> reassign_labels(const vector<int>& labels) {
    vector<int> new_labels(labels.size());
    map<int, int> cluster_mapping;
    int next_cluster = 1;
    // クラスタ（ノイズ以外）の再割り当て
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] != NOISE) {
            if (cluster_mapping.find(labels[i]) == cluster_mapping.end()) {
                cluster_mapping[labels[i]] = next_cluster;
                next_cluster++;
            }
            new_labels[i] = cluster_mapping[labels[i]];
        }
    }
    // ノイズ（-1）の各点に対しては個別のラベルを付与
    int noise_label = next_cluster;
    for (size_t i = 0; i < labels.size(); i++) {
        if (labels[i] == NOISE) {
            new_labels[i] = noise_label;
            noise_label++;
        }
    }
    return new_labels;
}

// --- 各クラスタの要素数カウント --- //
// 新たに割り当てたラベル配列から、各クラスタに属する要素数を（ラベル昇順で）返す
vector<int> count_clusters(const vector<int>& new_labels) {
    map<int, int> counts;
    for (int label : new_labels) {
        counts[label]++;
    }
    vector<int> cluster_counts;
    for (auto& kv : counts) {
        cluster_counts.push_back(kv.second);
    }
    return cluster_counts;
}

// --- action_clustering --- //
// 入力：action_ls_sorted（各系列は同じ長さの 0,1 の配列）
// 処理：各系列を末尾300要素にトリム（必要なら）、距離行列を作成、DBSCAN実行、ラベル再割り当て、各クラスタ要素数のカウント
// 出力：再割り当て後のラベル配列とクラスタ要素数（ペアで返す）
vector<int> action_clustering(const vector<vector<int>>& action_ls_sorted) {
    // 各系列を末尾300要素にトリム（各系列が300以上の場合）
    vector<vector<int>> data;
    for (const auto& series : action_ls_sorted) {
        if (series.size() > 300) {
            vector<int> trimmed(series.end() - 300, series.end());
            data.push_back(trimmed);
        } else {
            data.push_back(series);
        }
    }
    int M = data.size();
    // 距離行列 D の作成（各系列間のハミング距離）
    vector<vector<double>> D(M, vector<double>(M, 0.0));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            D[i][j] = hamming_distance(data[i], data[j]);
        }
    }

    // DBSCAN の実行（eps = 0.1, min_samples = 2）
    double eps = 0.1;
    int min_samples = 2;
    vector<int> labels = dbscan(D, eps, min_samples);
    vector<int> new_labels = reassign_labels(labels);
    vector<int> cluster_counts = count_clusters(new_labels);
    return cluster_counts;
}

// vector<int> を区切り文字 '_' で連結した文字列に変換するヘルパー関数
string vector_to_string(const vector<int>& v) {
    ostringstream oss;
    for (size_t i = 0; i < v.size(); i++) {
        oss << v[i];
        if (i != v.size() - 1)
            oss << "_";
    }
    return oss.str();
}

// cluster_counts を100個並べたリストから最頻値（mode）を返す関数
vector<int> mode_cluster_counts(const vector<vector<int>>& cluster_counts_list) {
    unordered_map<string, pair<int, vector<int>>> freq;

    // 各 cluster_counts の vector<int> を文字列に変換して頻度をカウント
    for (const auto &counts : cluster_counts_list) {
        string key = vector_to_string(counts);
        if (freq.find(key) == freq.end()) {
            freq[key] = make_pair(1, counts);
        } else {
            freq[key].first++;
        }
    }

    // 最頻出（出現回数最大）の cluster_counts を求める
    int max_count = 0;
    vector<int> mode;
    for (const auto &entry : freq) {
        if (entry.second.first > max_count) {
            max_count = entry.second.first;
            mode = entry.second.second;
        }
    }
    return mode;
}


//==================================================
// do_one_generation の高速化版（グループ情報をインデックス演算で処理）
//==================================================

inline std::tuple<double, double, double, double, double, double>
do_one_generation(std::vector<Player> &players,
                  int generation_idx,
                  int num_players,
                  double mutation,
                  int group_size)
{
    int n = players.size();
    double group_beta = 0.9 / group_size;

    // 1) ランダムシャッフル & 補填
    std::shuffle(players.begin(), players.end(), mt);
    int remainder = n % group_size;
    int needed = (remainder != 0) ? (group_size - remainder) : 0;
    for (int i = 0; i < needed; i++) players.push_back(players[i]);
    int N = players.size();
    int num_groups = N / group_size;

    std::vector<double> fitness_sum(N, 0.0);
    double total_action_all_steps = 0.0;
    double total_resource_all_steps = 0.0;
    std::vector<double> group_resources(num_groups, 0.1);

    // 2) ゲームステップ
    for (int step = 0; step < num_steps; step++) {
        for (int g = 0; g < num_groups; g++) {
            int start = g * group_size;
            int end = start + group_size;
            double sum_s = 0.0;
            for (int i = start; i < end; i++) sum_s += players[i].state;
            double avg_s = sum_s / group_size;

            std::vector<int> actions(group_size);
            int total_actions = 0;
            for (int i = start; i < end; i++) {
                double val = group_resources[g]
                             + players[i].S * players[i].state
                             + players[i].A * avg_s;
                int act = (val > 0.0) ? 1 : 0;
                actions[i - start] = act;
                total_actions += act;
                if (step > 499) fitness_sum[i] += players[i].state;
            }

            double payoff_actor = 0.0;
            payoff(group_resources[g], total_actions, group_beta, payoff_actor);
            for (int i = start; i < end; i++) {
                double add = actions[i - start] ? payoff_actor : 0.0;
                players[i].state = 0.75 * players[i].state + add;
            }
            total_action_all_steps += total_actions;
            total_resource_all_steps += group_resources[g];
        }
    }

    // 3) fitness 確定
    std::vector<double> fitnesses(N);
    for (int i = 0; i < N; i++) {
        players[i].fitness = fitness_sum[i] / 500.0;
        fitnesses[i] = players[i].fitness;
    }

    // 全体平均 fitness
    double sumF = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
    double aveFitness = sumF / N;

    // 4) グループごとの fitness 分散を計算
    double sumGroupVar = 0.0;
    for (int g = 0; g < num_groups; g++) {
        int start = g * group_size;
        int end = start + group_size;
        double grpSum = 0.0;
        for (int i = start; i < end; i++) grpSum += fitnesses[i];
        double grpMean = grpSum / group_size;
        double grpVar = 0.0;
        for (int i = start; i < end; i++) {
            double d = fitnesses[i] - grpMean;
            grpVar += d * d;
        }
        sumGroupVar += grpVar / group_size;
    }
    double avgVarFitness = sumGroupVar / num_groups;

    // 平均アクション／資源
    double aveAction   = total_action_all_steps / (num_steps * N);
    double aveResource = total_resource_all_steps / (num_steps * num_groups);

    // パラメータ平均
    double sumS = 0.0, sumA = 0.0;
    for (auto &p : players) { sumS += p.S; sumA += p.A; }
    double meanS = sumS / N;
    double meanA = sumA / N;

    std::vector<Player> nextPlayers;
    nextPlayers.reserve(N * 2);
    double normalizer = sumF / num_players;
    for (int i = 0; i < N; i++)
    {
        double lambda = players[i].fitness / normalizer;
        int num_offspring = poisson_random(lambda);
        for (int j = 0; j < num_offspring; j++)
        {
            double newS = players[i].S + normal_random(0.0, mutation);
            double newA = players[i].A + normal_random(0.0, mutation);
            nextPlayers.emplace_back(newS, newA);
        }
    }
    if (nextPlayers.empty())
        nextPlayers = players;
    if (nextPlayers.size() == 1)
        nextPlayers.push_back(nextPlayers[0]);
    players.swap(nextPlayers);

    return {aveAction, aveResource, aveFitness, meanS, meanA, avgVarFitness};
}



//==================================================
// シミュレーション実行関数
//==================================================
// 引数: プレイヤー数, 突然変異率, グループサイズ m, trial 番号
inline void run_simulation(int num_groups, double mutation, int group_size, int trial)
{
    // 出力ファイル名例:
    // "res_N300_mu3pc_m2_trialX.csv"
    int mut_pc = (int)std::round(mutation * 100.0);
    std::string summary_fname = "res_N" + std::to_string(num_groups) + "_mu" + std::to_string(mut_pc) + "pc_m" + std::to_string(group_size) + "_trial" + std::to_string(trial) + ".csv";
    std::string summary_path = "res/" + summary_fname;

    // 初期世代のプレイヤー作成: S, A, V は平均0, stddev=0.1 で初期化
    std::vector<Player> players;
    int num_players = num_groups * group_size;
    players.reserve(num_players);
    for (int i = 0; i < num_players; i++)
    {
        double S_init = normal_random(0.0, mutation);
        double A_init = normal_random(0.0, mutation);
        players.emplace_back(S_init, A_init);
    }

    // 各世代のサマリ記録用バッファ
    std::vector<double> actRecord(num_generations);
    std::vector<double> resourceRecord(num_generations);
    std::vector<double> fitnessRecord(num_generations);
    std::vector<double> SRecord(num_generations);
    std::vector<double> ARecord(num_generations);
    std::vector<double> var_fitnessRecord(num_generations);

    // 各世代を順次進める
    for (int gen = 0; gen < num_generations; gen++)
    {
        auto [aveAction, aveResource, aveFitness, meanS, meanA, var_fitness] = do_one_generation(players, gen, num_players, mutation, group_size);

        actRecord[gen] = aveAction;
        resourceRecord[gen] = aveResource;
        fitnessRecord[gen] = aveFitness;
        SRecord[gen] = meanS;
        ARecord[gen] = meanA;
        var_fitnessRecord[gen] = var_fitness;
    }

    // サマリを CSV 出力
    std::ofstream ofs(summary_path);
    if (!ofs.is_open())
    {
        std::cerr << "Error: cannot open file " << summary_path << "\n";
        return;
    }
    ofs << "generation,mean_S,mean_A,mean_fitness,mean_action,mean_resource,var_fitness\n";
    for (int gen = 0; gen < num_generations; gen++)
    {
        ofs << gen << ","
            << SRecord[gen] << ","
            << ARecord[gen] << ","
            << fitnessRecord[gen] << ","
            << actRecord[gen] << ","
            << resourceRecord[gen] <<","
            << var_fitnessRecord[gen] << "\n";
    }
    ofs.close();
}

//==================================================
// main 関数
//==================================================
int main(int argc, char *argv[])
{
    // グループ数、突然変異率、グループサイズ (m) の候補リスト
    std::vector<int> groups_list = {10, 16, 25, 40, 63, 100, 160, 250, 400, 630, 1000};
//    std::vector<double> mutation_list = {0.03, 0.1, 0.3, 1.0, 3.0};
//    std::vector<double> mutation_list = {0.1, 1.0};
    // 例としてグループサイズ m の候補（ここでは 2 人組、3 人組、5 人組、10 人組）
//    std::vector<int> group_size_list = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    std::vector<int> group_size_list = {22, 24, 26, 28, 30};

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <i_value>" << std::endl;
        return 1;
    }
    int i_value = std::stoi(argv[1]); // 例: 1～10 を想定

    double mu = 1.0;
//    if (std::stoi(argv[2]) == 1)
//        mu = 0.1;
//    if (std::stoi(argv[2]) == 2)
//        mu = 10.0;

    // run_simulation(10, 0.03, 2, 0);

    // trial 番号: i_value*10 から i_value*10+9 まで
    for (int offset = 0; offset <= 9; offset++) {
        int trial = i_value * 10 + offset;
        mt.seed(trial); // trial ごとに乱数シードを変更

        for (auto ng : groups_list)
        {
            for (auto m : group_size_list)
            {
                run_simulation(ng, mu, m, trial);
            }
        }
    }

    return 0;
}
