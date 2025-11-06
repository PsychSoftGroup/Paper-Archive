// Tiny Decoder-Only Transformer (single-head) inference, single-file C++17
// - 结构：Embedding + Sinusoidal PE + [Pre-LN → SelfAttn → Residual → Pre-LN → FFN → Residual] × N → LN_f → Linear
// - 注意力实现：整序列重算 + 因果约束（仅累加到 t）
// - 采样：temperature + 可选 top-k
// - 分词：空格切分（不强制小写），遇不到的词用 unk_id
// - 权重：由 export_for_c_infer.py 生成的 config.bin / vocab.txt / weights.bin
//
// 编译参数：g++ -O3 -march=native -ffast-math -funroll-loops -flto -std=c++17 tiny_decoder_infer.cpp -o tiny_infer
//
// 运行：直接修改 main() 内的硬编码参数后运行 ./tiny_infer
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>

struct Config {
    int32_t vocab_size = 0;
    int32_t d_model = 0;
    int32_t n_layers = 0;
    int32_t d_ff = 0;
    int32_t max_len = 0;
    int32_t eos_id = -1;
    int32_t unk_id = -1;
};

static inline std::vector<uint8_t> read_all(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { throw std::runtime_error("failed to open: " + path); }
    ifs.seekg(0, std::ios::end);
    size_t sz = (size_t)ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    ifs.read((char*)buf.data(), sz);
    return buf;
}

static inline void load_config_bin(const std::string& path, Config& cfg) {
    auto bytes = read_all(path);
    if (bytes.size() < 7 * sizeof(int32_t)) {
        throw std::runtime_error("config.bin too small");
    }
    const int32_t* p = reinterpret_cast<const int32_t*>(bytes.data());
    cfg.vocab_size = p[0];
    cfg.d_model    = p[1];
    cfg.n_layers   = p[2];
    cfg.d_ff       = p[3];
    cfg.max_len    = p[4];
    cfg.eos_id     = p[5];
    cfg.unk_id     = p[6];
}

static inline void load_vocab_txt(const std::string& path, std::vector<std::string>& id2tok, std::unordered_map<std::string,int>& tok2id) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("failed to open vocab: " + path);
    std::string line;
    int idx = 0;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        id2tok.push_back(line);
        tok2id.emplace(line, idx);
        idx++;
    }
}

struct Weights {
    // 持有所有模型参数的 float 存储区，各种权重实际都指向这个buffer中的一部分
    std::vector<float> storage;

    // --- 模型主权重 ---

    // 词嵌入权重矩阵：词表大小 × 隐藏维度。每个词对应一个 d_model 维的向量（即 embedding）。
    // 用于将 token id 转换为模型可处理的高维向量。
    // vocab_size = 32000, d_model = 512
    // 形状：[vocab_size, d_model]
    const float* we;

    // 每一层Transformer的权重，n_layers个，每层都是下面这个结构：
    struct Layer {
        // 第一层 LayerNorm 的缩放参数（gamma），长度为隐藏维度
        // 用于调整 self-attention 输入的每一维特征的幅度
        // 形状：[d_model]
        const float* ln1_gamma;

        // 第一层 LayerNorm 的偏移参数（beta），长度为隐藏维度
        // 用于调整 self-attention 输入的每一维特征的中心
        // 形状：[d_model]
        const float* ln1_beta;

        // 自注意力部分的 Query 权重矩阵。将输入特征变换为 Query 空间，用于计算注意力分数。
        // 形状：[d_model, d_model]，行主序，out维为行，in维为列
        const float* W_q;

        // 自注意力部分的 Key 权重矩阵。将输入特征变换为 Key 空间。
        // 形状：[d_model, d_model]
        const float* W_k;

        // 自注意力部分的 Value 权重矩阵。将输入特征变换为 Value 空间。
        // 形状：[d_model, d_model]
        const float* W_v;

        // 第二层 LayerNorm 的缩放参数（gamma），用于 FFN 输入
        // 形状：[d_model]
        const float* ln2_gamma;

        // 第二层 LayerNorm 的偏移参数（beta），用于 FFN 输入
        // 形状：[d_model]
        const float* ln2_beta;

        // FFN（前馈神经网络）第一层的权重矩阵，将输入从隐藏维度升到中间高维空间
        // 形状：[d_ff, d_model]
        const float* fc1_w;

        // FFN 第一层的偏置向量，长度中间层维度
        // 形状：[d_ff]
        const float* fc1_b;

        // FFN 第二层（输出层）的权重矩阵，将高维空间降回隐藏维度
        // 形状：[d_model, d_ff]
        const float* fc2_w;

        // FFN 第二层的偏置向量，长度为隐藏维度
        // 形状：[d_model]
        const float* fc2_b;
    };

    // 所有 Transformer 层（n_layers个）的参数集合
    std::vector<Layer> layers;

    // --- 输出部分相关权重 ---

    // 末端 LayerNorm 的缩放参数（gamma），用于输出隐藏状态的最后归一化
    // 形状：[d_model]
    const float* ln_f_gamma;

    // 末端 LayerNorm 的偏移参数（beta），用于输出隐藏状态的最后归一化
    // 形状：[d_model]
    const float* ln_f_beta;

    // 输出层的权重矩阵，将最后隐藏向量投影到词表所有词的打分空间
    // 形状：[vocab_size, d_model]
    const float* head_w;

    // 输出层的偏置向量，对每个词的打分做微调
    // 形状：[vocab_size]
    const float* head_b;
};

// 统计float 类型的数量
static inline size_t expect_weights_floats(const Config& c) {
    size_t n = 0;
    n += (size_t)c.vocab_size * c.d_model; // we
    for (int i=0;i<c.n_layers;i++) {
        n += c.d_model; // ln1_gamma
        n += c.d_model; // ln1_beta
        n += (size_t)c.d_model * c.d_model; // W_q
        n += (size_t)c.d_model * c.d_model; // W_k
        n += (size_t)c.d_model * c.d_model; // W_v
        n += c.d_model; // ln2_gamma
        n += c.d_model; // ln2_beta
        n += (size_t)c.d_ff * c.d_model; // fc1_w
        n += c.d_ff; // fc1_b
        n += (size_t)c.d_model * c.d_ff; // fc2_w
        n += c.d_model; // fc2_b
    }
    n += c.d_model; // ln_f_gamma
    n += c.d_model; // ln_f_beta
    n += (size_t)c.vocab_size * c.d_model; // head_w
    n += c.vocab_size; // head_b
    return n;
}

static inline void load_weights_bin(const std::string& path, const Config& cfg, Weights& W) {
    auto bytes = read_all(path);
    size_t need = expect_weights_floats(cfg);
    if (bytes.size() != need * sizeof(float)) {
        throw std::runtime_error("weights.bin size mismatch. expected floats=" + std::to_string(need) +
                                 " got bytes=" + std::to_string(bytes.size()));
    }
    W.storage.resize(need);
    std::memcpy(W.storage.data(), bytes.data(), bytes.size());
    size_t off = 0;
    auto take = [&](size_t cnt)->const float* {
        const float* p = W.storage.data() + off;
        off += cnt;
        return p;
    };
    W.we = take((size_t)cfg.vocab_size * cfg.d_model);

    W.layers.resize(cfg.n_layers);
    for (int i=0;i<cfg.n_layers;i++) {
        auto& L = W.layers[i];
        L.ln1_gamma = take(cfg.d_model);
        L.ln1_beta  = take(cfg.d_model);
        L.W_q       = take((size_t)cfg.d_model * cfg.d_model);
        L.W_k       = take((size_t)cfg.d_model * cfg.d_model);
        L.W_v       = take((size_t)cfg.d_model * cfg.d_model);
        L.ln2_gamma = take(cfg.d_model);
        L.ln2_beta  = take(cfg.d_model);
        L.fc1_w     = take((size_t)cfg.d_ff * cfg.d_model);
        L.fc1_b     = take(cfg.d_ff);
        L.fc2_w     = take((size_t)cfg.d_model * cfg.d_ff);
        L.fc2_b     = take(cfg.d_model);
    }
    W.ln_f_gamma = take(cfg.d_model);
    W.ln_f_beta  = take(cfg.d_model);
    W.head_w     = take((size_t)cfg.vocab_size * cfg.d_model);
    W.head_b     = take(cfg.vocab_size);
    assert(off == W.storage.size());
}

// ----------- 基础算子 -----------

static inline void layernorm_vec(const float* x, float* y, int D,
				 const float* gamma, const float* beta,
				 float eps=1e-5f) {
    // y = (x - mean) / sqrt(var+eps) * gamma + beta
    // 归一化，稳定每层输出分布
    float mean = 0.f;
    for (int i=0;i<D;i++) mean += x[i];
    mean /= (float)D;
    float var = 0.f;
    for (int i=0;i<D;i++){ float d=x[i]-mean; var += d*d; }
    var /= (float)D;
    float inv = 1.0f / std::sqrt(var + eps);
    // gamma调整尺度，beta调整中心位置。将向量微调到最适合当前任务和模型的状态
    for (int i=0;i<D;i++) {
        float n = (x[i]-mean)*inv;
        y[i] = n*gamma[i] + beta[i];
    }
}

static inline float gelu(float x){
    // PyTorch 默认 GELU（erf 版）
    return 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

static inline void linear_vec(const float* W /*[OUT,IN]*/, const float* b,
			      const float* x /*[IN]*/, float* y /*[OUT]*/,
			      int OUT, int IN) {
    // y = W * x + b，
    for (int o=0;o<OUT;o++){
        const float* wrow = W + (size_t)o*IN;
        float acc = b ? b[o] : 0.f;
        for (int i=0;i<IN;i++) acc += wrow[i]*x[i];
        y[o] = acc;
    }
}

static inline void add_inplace(float* a, const float* b, int D){
    for (int i=0;i<D;i++) a[i]+=b[i];
}

static inline void softmax_stable(float* x, int N){
    float mx = x[0];
    for (int i=1;i<N;i++) if (x[i]>mx) mx=x[i];
    // 防止爆float, 先求一个mx,分子分母同除以mx计算结果不变，并降低分子分母大小
    float sum=0.f;
    for (int i=0;i<N;i++){ x[i] = std::exp(x[i]-mx); sum+=x[i]; }
    float inv = 1.0f/sum;
    for (int i=0;i<N;i++) x[i]*=inv;
}

// ----------- 正弦位置编码（预计算） -----------
static inline std::vector<float> build_posenc(int max_len, int d_model){
    // max_len: 模型能支持的最大“输入序列长度”，包括提示词 + 已生成内容
    std::vector<float> pe((size_t)max_len * d_model, 0.f);
    for (int pos=0;pos<max_len;pos++){
      for (int i=0;i<d_model;i+=2){
  	  /*
	    高次幂运算可能会丢失精度，所以多一层exp和log运算
	    朴素方法，边界情况10000^{-1023/512}=1.02e-8对float仍在精度范围内
	    pe使用一维模拟二维：1. vector不支持显然的二维; 2. 内存连续性能好;
	   */
            float div = std::exp(-std::log(10000.0f) * (float)i / (float)d_model);
            pe[(size_t)pos*d_model + i] = std::sin(pos * div);
            if (i+1<d_model)
                pe[(size_t)pos*d_model + i+1] = std::cos(pos * div);
        }
    }
    return pe;
}

// ----------- 分词与反分词 -----------
static inline std::vector<std::string> split_ws(const std::string& s){
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string tok;
    while (iss >> tok) tokens.push_back(tok);
    return tokens;
}

// ----------- 自注意力（整序列、因果） -----------
// 输入 norm_out: [seq_len, model_dim]，输出 attn_out: [seq_len, model_dim]
// W_q/W_k/W_v: [model_dim, model_dim]（out, in）
static void self_attention(const std::vector<float>& norm_out, int seq_len, int model_dim,
                           const float* W_q, const float* W_k, const float* W_v,
                           std::vector<float>& attn_out)
{
    attn_out.assign((size_t)seq_len * model_dim, 0.f);
    std::vector<float> queries((size_t)seq_len * model_dim),
                       keys((size_t)seq_len * model_dim),
                       values((size_t)seq_len * model_dim),
                       tmp(model_dim);

    // 用训练好的Q/K/V权重，与输入向量做矩阵乘法，算出queries、keys、values
    for (int t = 0; t < seq_len; t++){
      linear_vec(W_q, /*b*/nullptr,
		 &norm_out[(size_t)t * model_dim],
		 &queries[(size_t)t * model_dim],
		 model_dim, model_dim);

      linear_vec(W_k, /*b*/nullptr,
		 &norm_out[(size_t)t * model_dim],
		 &keys[(size_t)t * model_dim],
		 model_dim, model_dim);

      linear_vec(W_v, /*b*/nullptr,
		 &norm_out[(size_t)t * model_dim],
		 &values[(size_t)t * model_dim],
		 model_dim, model_dim);
    }

    const float scale = 1.0f / std::sqrt((float)model_dim);
    // 对每个 t，计算到 0..t 的注意力
    std::vector<float> scores; scores.reserve(1024);
    for (int t = 0; t < seq_len; t++){
        /*
	  scores[u] = (queries[t] · keys[u]) / sqrt(model_dim)，u=0..t
	  计算当前词和历史词在特征空间的相似度/相关性
        */
        scores.assign(t + 1, 0.f);
        for (int u = 0; u <= t; u++){
            float dot = 0.f;
            // 计算点积，dot(queries[t], keys[u])
            for (int i = 0; i < model_dim; i++)
                dot += queries[t * model_dim + i] * keys[u * model_dim + i];
            scores[u] = dot * scale;
        }
        // softmax
        softmax_stable(scores.data(), (int)scores.size());

        // attn_out[t] = sum_u scores[u] * values[u]
        float* out_t = &attn_out[(size_t)t * model_dim];
        std::fill(out_t, out_t + model_dim, 0.f);
        for (int u = 0; u <= t; u++){
            float w = scores[u];
            for (int i = 0; i < model_dim; i++)
                out_t[i] += w * values[u * model_dim + i];
        }
    }
}

// ----------- 单步前向（整序列重算，取最后一步 logits） -----------
static void forward_fullsequence_logits_last(const Config& cfg, const Weights& W,
                                             const std::vector<float>& pe,
                                             const std::vector<int>& ids,
                                             std::vector<float>& logits_out)
{
    // 当前输入 prompt 序列的token长度
    const int seq_len   = (int)ids.size();
    // 向量维度 d_model （实际512）
    const int model_dim = cfg.d_model;
    // 词表大小，预测 vocab_size 个可能Token
    const int vocab_size= cfg.vocab_size;

    // hidden = Embedding + PE（原 X = Embedding + PE）
    // 需要对prompt的每个token转换成512维的向量，所以需要一个 seq_len * model_dim 的 vector
    std::vector<float> hidden((size_t)seq_len * model_dim);
    for (int t=0;t<seq_len;t++){
        int id = (ids[t] >= 0 && ids[t] < vocab_size) ? ids[t] : cfg.unk_id;
        /*
          找到第 id 个 token 的 embedding 向量起始地址
          W.we: 所有的 embedding 的首地址
          id*model_dim: 跳到第 id 个 embedding 的首元素(起点)
         */
        const float* emb = W.we + (size_t)id * model_dim;
        /*
          找到第 t 个位置编码向量的起始地址
          data()成员方法，返回这块 vector 的“首地址”指针
          t*model_dim: 跳到第 t 个位置的编码起点
         */
        const float* pe_t = pe.data() + (size_t)t * model_dim;

        // 对每一个维度的 i，把 embedding 和位置编码分别相加，得到输入向量
	for (int i = 0; i < model_dim; i++) {
	  hidden[(size_t)t * model_dim + i] = emb[i] + pe_t[i];
	}
    }

    /*
      norm_out: 存储每层的 LayerNorm归一化 输出
      attn_out: 存储每层的 自注意力 输出
      ffn_out:  存储每层的 前馈神经网络 输出
     */
    std::vector<float>
    norm_out((size_t)seq_len * model_dim),
    attn_out((size_t)seq_len * model_dim),
    ffn_out((size_t)seq_len * model_dim);

    for (int l=0;l<cfg.n_layers;l++){
        const auto& layer = W.layers[l];
        // Pre-LN → SelfAttn → 残差

	// LayerNorm归一化
        for (int t=0;t<seq_len;t++){
            layernorm_vec(&hidden[(size_t)t * model_dim],
                          &norm_out[(size_t)t * model_dim],
                          model_dim, layer.ln1_gamma, layer.ln1_beta);
        }

	// 自注意力
        self_attention(norm_out, seq_len, model_dim,
		       layer.W_q, layer.W_k, layer.W_v,
		       attn_out);
      	// 残差连接
        for (int i=0;i<seq_len*model_dim;i++) hidden[i] += attn_out[i];

        // Pre-LN → FFN → 残差
        for (int t=0;t<seq_len;t++){
            layernorm_vec(&hidden[(size_t)t * model_dim],
                          &norm_out[(size_t)t * model_dim],
                          model_dim, layer.ln2_gamma, layer.ln2_beta);
            // FFN: y = GELU(norm_out*fc1^T + b1) * fc2^T + b2
            // 先 h = fc1(norm_out)
            std::vector<float> h(cfg.d_ff);
            linear_vec(layer.fc1_w, layer.fc1_b, &norm_out[(size_t)t * model_dim], h.data(), cfg.d_ff, model_dim);

	    // 激活函数
            for (int i=0;i<cfg.d_ff;i++) h[i] = gelu(h[i]);

            // 再 y = fc2(h)
            linear_vec(layer.fc2_w, layer.fc2_b, h.data(), &ffn_out[(size_t)t * model_dim], model_dim, cfg.d_ff);
        }
        for (int i=0;i<seq_len*model_dim;i++) hidden[i] += ffn_out[i];
    }

    // 末端 LN + Linear → logits
    std::vector<float> y(model_dim);
    layernorm_vec(&hidden[(size_t)(seq_len-1) * model_dim], y.data(), model_dim, W.ln_f_gamma, W.ln_f_beta);
    logits_out.resize(vocab_size);
    linear_vec(W.head_w, W.head_b, y.data(), logits_out.data(), vocab_size, model_dim);
}

// ----------- 采样：temperature + 可选 top-k -----------
static int sample_next_id(std::mt19937_64& rng, std::vector<float>& logits, float temperature, int top_k) {
    const int V = (int)logits.size();
    /*
      温度缩放，在softmax采样之前，对logits得分列表进行缩放，调整概率分布的“尖锐程度”
      如果temperature<1.0，所有的logits被除以一个小于 1.0 的数，那么高分词概率越高，低分词概率越低;
      也就是说内容更加确定，更模板化。
      如果temperature>1.0，那么高分和低分差距就变小，生成内容更加随机
     */
    if (temperature != 1.0f) {
        for (int i=0;i<V;i++) logits[i] /= temperature;
    }

    // top-k 过滤（k<=0 表示关闭）
    if (top_k > 0 && top_k < V) {
        // 排序会改变下标，后续需要依靠原下标进行滤筛，所以暂存一下
        std::vector<int> idx(V);
        for (int i=0;i<V;i++) idx[i]=i;
	// 得到分数最高的前 k 个词，tresh是第k个分数，分数比他低的将被淘汰
        std::nth_element(idx.begin(), idx.begin()+top_k, idx.end(), [&](int a,int b){
            return logits[a] > logits[b];
        });
        float thresh = logits[idx[top_k]];
        for (int i=0;i<V;i++) if (logits[i] < thresh) logits[i] = -1e9f;
    }
    // softmax
    softmax_stable(logits.data(), V);

    // 离散采样（多项式）
    /*
      使生成内容更加多样，例如一个logits列表：
      "cat": 0.5
      "dog": 0.3
      "fish": 0.2
      可以防止文本总是产生cat
     */
    std::uniform_real_distribution<float> unif(0.f, 1.f);
    float r = unif(rng);
    float c = 0.f;
    for (int i=0;i<V;i++){
        c += logits[i];
        if (r <= c) return i;
    }
    return V-1; // 兜底（数值边界）
}

// ----------- 编码/解码 -----------
static inline std::vector<int> encode(const std::string& prompt,
                                      const std::unordered_map<std::string,int>& tok2id,
                                      int unk_id)
{
    /*
      prompt切分，"hello world!" -> toks = ["hello", "world!"]
      哈希找到对应id编号
     */
    auto toks = split_ws(prompt);
    std::vector<int> ids; ids.reserve(toks.size());
    for (auto& t : toks){
        auto it = tok2id.find(t);
        ids.push_back( (it==tok2id.end()) ? unk_id : it->second );
    }
    return ids;
}

static inline std::string decode_suffix(const std::vector<int>& all_ids, size_t skip,
                                        const std::vector<std::string>& id2tok)
{
    std::ostringstream oss;
    for (size_t i=skip;i<all_ids.size();i++){
        if (i>skip) oss << ' ';
        int id = all_ids[i];
        if (id>=0 && id<(int)id2tok.size()) oss << id2tok[id];
        else oss << "<UNK>";
    }
    return oss.str();
}

// ----------- 小工具：去掉续写末尾的某个 token（例如 <|endoftext|>） -----------
static inline std::string strip_trailing_token(const std::string& s, const std::string& token){
    if (token.empty()) return s;
    // 去掉末尾空格
    size_t end = s.size();
    while (end>0 && std::isspace((unsigned char)s[end-1])) --end;
    if (end < token.size()) return s;
    if (s.compare(end - token.size(), token.size(), token) == 0) {
        size_t pos = end - token.size();
        // 去掉 token 前可能的空格
        while (pos>0 && std::isspace((unsigned char)s[pos-1])) --pos;
        return s.substr(0, pos);
    }
    return s;
}

// ----------- 小工具：句首大写（按英文句子：开头、.?! 之后的第一个字母变大写） -----------
static inline std::string to_sentence_case(const std::string& s){
    std::string out = s;
    bool new_sentence = true;
    for (size_t i=0;i<out.size(); ++i){
        unsigned char c = static_cast<unsigned char>(out[i]);
        if (new_sentence){
            if (std::isalpha(c)){
                out[i] = (char)std::toupper(c);
                new_sentence = false;
            } else if (!std::isspace(c) && c!='\"' && c!='\'' && c!='(' && c!='[') {
                // 碰到其它非空白字符，也算进入句子，直到遇到终止符
                new_sentence = false;
            }
        }
        // 碰到句子终止符，开启下一句
        if (c=='.' || c=='!' || c=='?' || c=='\n'){
            new_sentence = true;
        }
    }
    return out;
}

// ----------- 主生成 -----------
static std::string generate(const Config& cfg, const Weights& W,
                            const std::vector<std::string>& id2tok,
                            const std::unordered_map<std::string,int>& tok2id,
                            const std::string& prompt,
                            int max_new_tokens, float temperature, int top_k)
{
    // 预计算位置编码
    auto pe = build_posenc(cfg.max_len, cfg.d_model);

    // 编码 prompt
    std::vector<int> ids = encode(prompt, tok2id, cfg.unk_id);
    size_t prompt_len = ids.size();
    if ((int)ids.size() >= cfg.max_len) {
        ids.resize(cfg.max_len-1);
    }

    std::mt19937_64 rng(42); // 随机数种子42,结果可复现
    // 自回归文本生成
    for (int step=0; step < max_new_tokens; ++step){
        if ((int)ids.size() >= cfg.max_len) break;

	// 下一个最可能出现的词打分列表
        std::vector<float> logits;
        forward_fullsequence_logits_last(cfg, W, pe, ids, logits);

	// 获取下一个 token 编号
        int next_id = sample_next_id(rng, logits, temperature, top_k);
        ids.push_back(next_id);
        if (next_id == cfg.eos_id) break;
    }
    return decode_suffix(ids, prompt_len, id2tok);
}

int main(){
    try {
        // 1) 硬编码参数
        const std::string run_dir   = "runs_ipynb/test";
        const std::string cfg_path  = run_dir + "/config.bin";
        const std::string vocab_path= run_dir + "/vocab.txt";
        const std::string wt_path   = run_dir + "/weights.bin";

        const std::string prompt = "There was a cat,";
        const int   max_new_tokens = 200;
        const float temperature    = 0.9f;
        const int   top_k          = 50;  // 0 表示关闭

        // 2) 加载工件
        Config cfg;
        load_config_bin(cfg_path, cfg);

        std::vector<std::string> id2tok;
        std::unordered_map<std::string,int> tok2id;
        load_vocab_txt(vocab_path, id2tok, tok2id);

        Weights W;
        load_weights_bin(wt_path, cfg, W);

        // 3) 生成续写（不含 prompt）
        std::string cont = generate(cfg, W, id2tok, tok2id, prompt, max_new_tokens, temperature, top_k);

        // 4) 输出优化：
        //    - 去掉末尾 <|endoftext|>
        //    - 对续写做句首大写（英文规范）
        std::string eos_tok = (cfg.eos_id >=0 && cfg.eos_id < (int)id2tok.size()) ? id2tok[cfg.eos_id] : std::string("<|endoftext|>");
        std::string cont_clean = strip_trailing_token(cont, eos_tok);
        std::string cont_pretty = to_sentence_case(cont_clean);

        // 5) 打印两行
        std::cout << "原本提示词：" << prompt << "\n\n";
        if (!cont_pretty.empty()) {
            std::cout << "完整内容：" << prompt << " " << cont_pretty << "\n";
        } else {
            std::cout << "完整内容：" << prompt << "\n";
        }

    } catch (const std::exception& e){
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
