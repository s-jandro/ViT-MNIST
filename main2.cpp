#define STB_IMAGE_IMPLEMENTATION

#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "stb_image.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ====================== UTILIDADES ======================
uint32_t swap_endian(uint32_t val)
{
    return ((val << 24) & 0xFF000000) | ((val << 8) & 0x00FF0000) |
           ((val >> 8) & 0x0000FF00) | ((val >> 24) & 0x000000FF);
}

vector<MatrixXf> load_images(const string &path, int num_images)
{
    ifstream file(path, ios::binary);
    if (!file.is_open())
    {
        cerr << "No se pudo abrir " << path << endl;
        exit(1);
    }
    uint32_t magic, n_imgs, rows, cols;
    file.read((char *)&magic, 4);
    file.read((char *)&n_imgs, 4);
    file.read((char *)&rows, 4);
    file.read((char *)&cols, 4);
    magic = swap_endian(magic);
    n_imgs = swap_endian(n_imgs);
    rows = swap_endian(rows);
    cols = swap_endian(cols);
    vector<MatrixXf> images;
    images.reserve(num_images);
    vector<unsigned char> buffer(rows * cols);
    for (int i = 0; i < num_images; ++i)
    {
        file.read((char *)buffer.data(), rows * cols);
        MatrixXf img(rows, cols);
        for (int p = 0; p < (int)(rows * cols); ++p)
            img(p / cols, p % cols) = buffer[p] / 255.0f;
        images.push_back(img);
    }
    return images;
}

vector<int> load_labels(const string &path, int num_labels)
{
    ifstream file(path, ios::binary);
    if (!file.is_open())
    {
        cerr << "No se pudo abrir " << path << endl;
        exit(1);
    }
    uint32_t magic, n;
    file.read((char *)&magic, 4);
    file.read((char *)&n, 4);
    vector<int> labels(num_labels);
    for (int i = 0; i < num_labels; ++i)
    {
        unsigned char c;
        file.read((char *)&c, 1);
        labels[i] = c;
    }
    return labels;
}

struct TransformerCache
{
    MatrixXf input_to_block;
    MatrixXf ln1_out;
    vector<MatrixXf> q_heads, k_heads, v_heads, scores_heads;
    MatrixXf attn_concat_out;
    MatrixXf ln2_out;
    MatrixXf ffn_inner;
    MatrixXf ffn_relu;
};

class VisionTransformer
{
public:
    int patch_size = 7;
    int num_patches = 16;
    int embed_dim = 64;
    int num_heads = 4;
    int depth = 3;
    int num_classes = 10;
    int head_dim;

    MatrixXf patch_embeddings;
    VectorXf cls_token;
    MatrixXf pos_embeddings;

    vector<MatrixXf> WQ, WK, WV, WO;
    vector<MatrixXf> ffn1, ffn2;
    vector<VectorXf> ln1_gamma, ln1_beta, ln2_gamma, ln2_beta;

    MatrixXf classifier_w;
    VectorXf classifier_b;

    MatrixXf d_patch_embeddings;
    VectorXf d_cls_token;
    MatrixXf d_pos_embeddings;
    vector<MatrixXf> d_WQ, d_WK, d_WV, d_WO;
    vector<MatrixXf> d_ffn1, d_ffn2;
    vector<VectorXf> d_ln1_gamma, d_ln1_beta, d_ln2_gamma, d_ln2_beta;
    MatrixXf d_classifier_w;
    VectorXf d_classifier_b;

    vector<TransformerCache> caches;
    MatrixXf patches_cache;

    mt19937 rng{42};

    VisionTransformer()
    {
        head_dim = embed_dim / num_heads;
        float std = sqrtf(2.0f / embed_dim);

        auto init_mat = [&](MatrixXf &m, int r, int c, float s)
        {
            m = MatrixXf::Zero(r, c);
            normal_distribution<float> dist(0.0f, s);
            for (int i = 0; i < m.size(); ++i)
                m(i) = dist(rng);
        };

        init_mat(patch_embeddings, patch_size * patch_size, embed_dim, std);
        cls_token = VectorXf::Zero(embed_dim);
        init_mat(pos_embeddings, 1 + num_patches, embed_dim, std);

        WQ.resize(depth);
        WK.resize(depth);
        WV.resize(depth);
        WO.resize(depth);
        ffn1.resize(depth);
        ffn2.resize(depth);
        ln1_gamma.resize(depth);
        ln1_beta.resize(depth);
        ln2_gamma.resize(depth);
        ln2_beta.resize(depth);
        caches.resize(depth);

        for (int i = 0; i < depth; ++i)
        {
            init_mat(WQ[i], embed_dim, embed_dim, std);
            init_mat(WK[i], embed_dim, embed_dim, std);
            init_mat(WV[i], embed_dim, embed_dim, std);
            init_mat(WO[i], embed_dim, embed_dim, std);
            init_mat(ffn1[i], embed_dim, embed_dim * 2, std);
            init_mat(ffn2[i], embed_dim * 2, embed_dim, std);

            ln1_gamma[i] = VectorXf::Ones(embed_dim);
            ln1_beta[i] = VectorXf::Zero(embed_dim);
            ln2_gamma[i] = VectorXf::Ones(embed_dim);
            ln2_beta[i] = VectorXf::Zero(embed_dim);
        }

        init_mat(classifier_w, embed_dim, num_classes, std);
        classifier_b = VectorXf::Zero(num_classes);

        resize_grads();
        reset_grads();
    }

    void resize_grads()
    {
        d_patch_embeddings.resizeLike(patch_embeddings);
        d_cls_token.resizeLike(cls_token);
        d_pos_embeddings.resizeLike(pos_embeddings);
        d_classifier_w.resizeLike(classifier_w);
        d_classifier_b.resizeLike(classifier_b);
        d_WQ.resize(depth);
        d_WK.resize(depth);
        d_WV.resize(depth);
        d_WO.resize(depth);
        d_ffn1.resize(depth);
        d_ffn2.resize(depth);
        d_ln1_gamma.resize(depth);
        d_ln1_beta.resize(depth);
        d_ln2_gamma.resize(depth);
        d_ln2_beta.resize(depth);
        for (int i = 0; i < depth; ++i)
        {
            d_WQ[i].resizeLike(WQ[i]);
            d_WK[i].resizeLike(WK[i]);
            d_WV[i].resizeLike(WV[i]);
            d_WO[i].resizeLike(WO[i]);
            d_ffn1[i].resizeLike(ffn1[i]);
            d_ffn2[i].resizeLike(ffn2[i]);
            d_ln1_gamma[i].resizeLike(ln1_gamma[i]);
            d_ln1_beta[i].resizeLike(ln1_beta[i]);
            d_ln2_gamma[i].resizeLike(ln2_gamma[i]);
            d_ln2_beta[i].resizeLike(ln2_beta[i]);
        }
    }

    void reset_grads()
    {
        d_patch_embeddings.setZero();
        d_cls_token.setZero();
        d_pos_embeddings.setZero();
        d_classifier_w.setZero();
        d_classifier_b.setZero();
        for (int i = 0; i < depth; ++i)
        {
            d_WQ[i].setZero();
            d_WK[i].setZero();
            d_WV[i].setZero();
            d_WO[i].setZero();
            d_ffn1[i].setZero();
            d_ffn2[i].setZero();
            d_ln1_gamma[i].setZero();
            d_ln1_beta[i].setZero();
            d_ln2_gamma[i].setZero();
            d_ln2_beta[i].setZero();
        }
    }

    VectorXf forward(const MatrixXf &img)
    {
        patches_cache = extract_patches(img);
        MatrixXf x = patches_cache * patch_embeddings;

        MatrixXf tokens(1 + num_patches, embed_dim);
        tokens.row(0) = cls_token.transpose();
        tokens.block(1, 0, num_patches, embed_dim) = x;
        tokens = tokens + pos_embeddings;

        for (int d = 0; d < depth; ++d)
        {
            caches[d].input_to_block = tokens;

            caches[d].ln1_out = layer_norm(tokens, ln1_gamma[d], ln1_beta[d]);
            MatrixXf attn_out = multi_head_attention(caches[d].ln1_out, d);
            tokens = tokens + attn_out;

            caches[d].ln2_out = layer_norm(tokens, ln2_gamma[d], ln2_beta[d]);
            caches[d].ffn_inner = caches[d].ln2_out * ffn1[d];
            caches[d].ffn_relu = caches[d].ffn_inner.array().max(0.0f);
            MatrixXf ffn_out = caches[d].ffn_relu * ffn2[d];
            tokens = tokens + ffn_out;
        }

        VectorXf cls_final = tokens.row(0).transpose();
        VectorXf logits = classifier_w.transpose() * cls_final + classifier_b;
        return logits;
    }

    void backward(const VectorXf &grad_logits)
    {
        MatrixXf final_tokens = caches[depth - 1].input_to_block +
                                (caches[depth - 1].attn_concat_out * WO[depth - 1]) +
                                (caches[depth - 1].ffn_relu * ffn2[depth - 1]);

        d_classifier_w += final_tokens.row(0).transpose() * grad_logits.transpose();
        d_classifier_b += grad_logits;

        MatrixXf d_tokens = MatrixXf::Zero(1 + num_patches, embed_dim);
        d_tokens.row(0) = classifier_w * grad_logits;

        for (int d = depth - 1; d >= 0; --d)
        {
            TransformerCache &cache = caches[d];

            // Backward MLP
            MatrixXf d_ffn_out = d_tokens;
            d_ffn2[d] += cache.ffn_relu.transpose() * d_ffn_out;
            MatrixXf d_ffn_relu = d_ffn_out * ffn2[d].transpose();
            MatrixXf d_ffn_inner = d_ffn_relu.array() * (cache.ffn_inner.array() > 0).cast<float>();

            d_ffn1[d] += cache.ln2_out.transpose() * d_ffn_inner;
            MatrixXf d_ln2_out = d_ffn_inner * ffn1[d].transpose();

            MatrixXf d_ln2_in = layer_norm_backward(d_ln2_out, cache.input_to_block + (cache.attn_concat_out * WO[d]),
                                                    ln2_gamma[d], d_ln2_gamma[d], d_ln2_beta[d]);
            d_tokens += d_ln2_in;

            // Backward Attention
            MatrixXf d_attn_proj = d_tokens;
            d_WO[d] += cache.attn_concat_out.transpose() * d_attn_proj;
            MatrixXf d_attn_concat = d_attn_proj * WO[d].transpose();

            MatrixXf d_ln1_out = MatrixXf::Zero(cache.ln1_out.rows(), embed_dim);
            int N = cache.ln1_out.rows();
            float scale = 1.0f / sqrtf((float)head_dim);

            for (int h = 0; h < num_heads; ++h)
            {
                MatrixXf d_head_out = d_attn_concat.block(0, h * head_dim, N, head_dim);
                MatrixXf q = cache.q_heads[h];
                MatrixXf k = cache.k_heads[h];
                MatrixXf v = cache.v_heads[h];
                MatrixXf s = cache.scores_heads[h];

                MatrixXf d_v = s.transpose() * d_head_out;
                MatrixXf d_s = d_head_out * v.transpose();

                MatrixXf d_logits(N, N);
                for (int r = 0; r < N; ++r)
                {
                    float sum_s_ds = s.row(r).dot(d_s.row(r));
                    d_logits.row(r) = s.row(r).array() * (d_s.row(r).array() - sum_s_ds);
                }
                d_logits *= scale;

                MatrixXf d_q = d_logits * k;
                MatrixXf d_k = d_logits.transpose() * q;

                d_WQ[d].block(0, h * head_dim, embed_dim, head_dim) += cache.ln1_out.transpose() * d_q;
                d_WK[d].block(0, h * head_dim, embed_dim, head_dim) += cache.ln1_out.transpose() * d_k;
                d_WV[d].block(0, h * head_dim, embed_dim, head_dim) += cache.ln1_out.transpose() * d_v;

                d_ln1_out += d_q * WQ[d].block(0, h * head_dim, embed_dim, head_dim).transpose();
                d_ln1_out += d_k * WK[d].block(0, h * head_dim, embed_dim, head_dim).transpose();
                d_ln1_out += d_v * WV[d].block(0, h * head_dim, embed_dim, head_dim).transpose();
            }

            MatrixXf d_ln1_in = layer_norm_backward(d_ln1_out, cache.input_to_block,
                                                    ln1_gamma[d], d_ln1_gamma[d], d_ln1_beta[d]);
            d_tokens += d_ln1_in;
        }

        d_pos_embeddings += d_tokens;
        MatrixXf d_tokens_patches = d_tokens.block(1, 0, num_patches, embed_dim);
        d_patch_embeddings += patches_cache.transpose() * d_tokens_patches;
        d_cls_token += d_tokens.row(0);
    }

    void update(float lr)
    {
        auto update_mat = [&](MatrixXf &w, const MatrixXf &dw)
        { w.noalias() -= lr * dw; };
        auto update_vec = [&](VectorXf &w, const VectorXf &dw)
        { w.noalias() -= lr * dw; };

        update_mat(patch_embeddings, d_patch_embeddings);
        update_vec(cls_token, d_cls_token);
        update_mat(pos_embeddings, d_pos_embeddings);
        update_mat(classifier_w, d_classifier_w);
        update_vec(classifier_b, d_classifier_b);

        for (int i = 0; i < depth; ++i)
        {
            update_mat(WQ[i], d_WQ[i]);
            update_mat(WK[i], d_WK[i]);
            update_mat(WV[i], d_WV[i]);
            update_mat(WO[i], d_WO[i]);
            update_mat(ffn1[i], d_ffn1[i]);
            update_mat(ffn2[i], d_ffn2[i]);
            update_vec(ln1_gamma[i], d_ln1_gamma[i]);
            update_vec(ln1_beta[i], d_ln1_beta[i]);
            update_vec(ln2_gamma[i], d_ln2_gamma[i]);
            update_vec(ln2_beta[i], d_ln2_beta[i]);
        }
    }

    // ================== GUARDAR Y CARGAR PESOS ==================
    void save_model(const string &filename)
    {
        ofstream file(filename, ios::binary);
        if (!file.is_open())
        {
            cerr << "Error guardando modelo" << endl;
            return;
        }

        auto write_mat = [&](const MatrixXf &m)
        {
            long rows = m.rows(), cols = m.cols();
            file.write((char *)&rows, sizeof(long));
            file.write((char *)&cols, sizeof(long));
            file.write((char *)m.data(), m.size() * sizeof(float));
        };
        auto write_vec = [&](const VectorXf &v)
        {
            long size = v.size();
            file.write((char *)&size, sizeof(long));
            file.write((char *)v.data(), v.size() * sizeof(float));
        };

        write_mat(patch_embeddings);
        write_vec(cls_token);
        write_mat(pos_embeddings);
        write_mat(classifier_w);
        write_vec(classifier_b);

        for (int i = 0; i < depth; ++i)
        {
            write_mat(WQ[i]);
            write_mat(WK[i]);
            write_mat(WV[i]);
            write_mat(WO[i]);
            write_mat(ffn1[i]);
            write_mat(ffn2[i]);
            write_vec(ln1_gamma[i]);
            write_vec(ln1_beta[i]);
            write_vec(ln2_gamma[i]);
            write_vec(ln2_beta[i]);
        }
        cout << "Modelo guardado en " << filename << endl;
    }

    void load_model(const string &filename)
    {
        ifstream file(filename, ios::binary);
        if (!file.is_open())
        {
            cerr << "No se encuentra el modelo guardado. Entrena primero." << endl;
            return;
        }

        auto read_mat = [&](MatrixXf &m)
        {
            long rows, cols;
            file.read((char *)&rows, sizeof(long));
            file.read((char *)&cols, sizeof(long));
            m.resize(rows, cols);
            file.read((char *)m.data(), m.size() * sizeof(float));
        };
        auto read_vec = [&](VectorXf &v)
        {
            long size;
            file.read((char *)&size, sizeof(long));
            v.resize(size);
            file.read((char *)v.data(), v.size() * sizeof(float));
        };

        read_mat(patch_embeddings);
        read_vec(cls_token);
        read_mat(pos_embeddings);
        read_mat(classifier_w);
        read_vec(classifier_b);

        for (int i = 0; i < depth; ++i)
        {
            read_mat(WQ[i]);
            read_mat(WK[i]);
            read_mat(WV[i]);
            read_mat(WO[i]);
            read_mat(ffn1[i]);
            read_mat(ffn2[i]);
            read_vec(ln1_gamma[i]);
            read_vec(ln1_beta[i]);
            read_vec(ln2_gamma[i]);
            read_vec(ln2_beta[i]);
        }
        cout << "Modelo cargado exitosamente." << endl;
    }

    // ================== PREDECIR IMAGEN PROPIA ==================
    void predict_custom_image(const string &path)
    {
        int w, h, channels;

        unsigned char *img_data = stbi_load(path.c_str(), &w, &h, &channels, 1);

        if (!img_data)
        {
            cerr << "Error: No se pudo cargar la imagen " << path << endl;
            return;
        }

        // Procesar imagen (Resize simple a 28x28 si no lo es)
        MatrixXf input(28, 28);

        // Factor de escala
        float scale_x = (float)w / 28.0f;
        float scale_y = (float)h / 28.0f;

        cout << "\n--- Prediccion para: " << path << " ---" << endl;
        cout << "Visualizacion (ASCII):" << endl;

        for (int i = 0; i < 28; ++i)
        {
            for (int j = 0; j < 28; ++j)
            {
                // Nearest Neighbor sampling
                int src_x = (int)(j * scale_x);
                int src_y = (int)(i * scale_y);
                int idx = src_y * w + src_x;

                float pixel = img_data[idx] / 255.0f;

                // Invertir colores (MNIST es blanco sobre negro)
                pixel = 1.0f - pixel;

                // Normalizar como en el entrenamiento
                input(i, j) = pixel - 0.1307f;

                // Dibujar ASCII pequeño para verificar
                if (pixel > 0.5)
                    cout << "#";
                else
                    cout << ".";
            }
            cout << endl;
        }
        stbi_image_free(img_data);

        // Forward
        VectorXf logits = forward(input);

        // Softmax para porcentajes
        float m = logits.maxCoeff();
        VectorXf probs = (logits.array() - m).exp();
        probs /= probs.sum();

        cout << "\nProbabilidades:" << endl;
        int pred;
        float confidence = probs.maxCoeff(&pred);

        for (int k = 0; k < 10; ++k)
        {
            printf("Digito %d: %.2f%%\n", k, probs(k) * 100.0f);
        }
        cout << ">> PREDICCION FINAL: " << pred << " (Confianza: " << confidence * 100 << "%)" << endl;
    }

private:
    MatrixXf extract_patches(const MatrixXf &img)
    {
        int patch_dim = patch_size * patch_size;
        MatrixXf patches(num_patches, patch_dim);
        int idx = 0;
        for (int i = 0; i < 28; i += patch_size)
            for (int j = 0; j < 28; j += patch_size)
                patches.row(idx++) = Map<const VectorXf>(img.block(i, j, patch_size, patch_size).data(), patch_dim);
        return patches;
    }

    // NORMALIZACION MANUAL (Sin Eigen broadcasting) para evitar errores
    MatrixXf layer_norm(const MatrixXf &x, const VectorXf &g, const VectorXf &b)
    {
        int N = x.rows();
        int D = x.cols();
        MatrixXf out(N, D);
        // Normalizacion "Batch-wise" sobre Features (D)
        for (int j = 0; j < D; ++j)
        {
            float mean = 0, var = 0;
            for (int i = 0; i < N; ++i)
                mean += x(i, j);
            mean /= N;
            for (int i = 0; i < N; ++i)
                var += (x(i, j) - mean) * (x(i, j) - mean);
            var /= N;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            float gamma = g(j);
            float beta = b(j);
            for (int i = 0; i < N; ++i)
            {
                out(i, j) = ((x(i, j) - mean) * inv) * gamma + beta;
            }
        }
        return out;
    }

    // BACKWARD NORMALIZACION MANUAL
    MatrixXf layer_norm_backward(const MatrixXf &dout, const MatrixXf &x, const VectorXf &g, VectorXf &dg, VectorXf &db)
    {
        int N = x.rows();
        int D = x.cols();
        MatrixXf dx(N, D);

        for (int j = 0; j < D; ++j)
        {
            // Recalcular stats
            float mean = 0, var = 0;
            for (int i = 0; i < N; ++i)
                mean += x(i, j);
            mean /= N;
            for (int i = 0; i < N; ++i)
                var += (x(i, j) - mean) * (x(i, j) - mean);
            var /= N;
            float inv_std = 1.0f / sqrtf(var + 1e-5f);

            float sum_dxhat = 0;
            float sum_dxhat_xhat = 0;
            float gamma = g(j);

            // Primer loop: acumular gradientes y sumas auxiliares
            for (int i = 0; i < N; ++i)
            {
                float x_norm = (x(i, j) - mean) * inv_std;
                float d_val = dout(i, j);

                dg(j) += d_val * x_norm;
                db(j) += d_val;

                float dxhat = d_val * gamma;
                sum_dxhat += dxhat;
                sum_dxhat_xhat += dxhat * x_norm;
            }

            // Segundo loop: calcular dx
            for (int i = 0; i < N; ++i)
            {
                float x_norm = (x(i, j) - mean) * inv_std;
                float dxhat = dout(i, j) * gamma;
                // Formula standard de Backprop para BN/LN
                float term = (N * dxhat) - sum_dxhat - (x_norm * sum_dxhat_xhat);
                dx(i, j) = term * (1.0f / N) * inv_std;
            }
        }
        return dx;
    }

    MatrixXf multi_head_attention(const MatrixXf &x, int layer)
    {
        int N = x.rows();
        caches[layer].q_heads.resize(num_heads);
        caches[layer].k_heads.resize(num_heads);
        caches[layer].v_heads.resize(num_heads);
        caches[layer].scores_heads.resize(num_heads);

        MatrixXf concat_out(N, embed_dim);
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < num_heads; ++h)
        {
            MatrixXf q = x * WQ[layer].block(0, h * head_dim, embed_dim, head_dim);
            MatrixXf k = x * WK[layer].block(0, h * head_dim, embed_dim, head_dim);
            MatrixXf v = x * WV[layer].block(0, h * head_dim, embed_dim, head_dim);
            caches[layer].q_heads[h] = q;
            caches[layer].k_heads[h] = k;
            caches[layer].v_heads[h] = v;

            MatrixXf scores = (q * k.transpose()) * scale;

            for (int r = 0; r < N; ++r)
            {
                float m = scores.row(r).maxCoeff();
                VectorXf row = (scores.row(r).array() - m).exp();
                scores.row(r) = row / row.sum();
            }
            caches[layer].scores_heads[h] = scores;

            concat_out.block(0, h * head_dim, N, head_dim) = scores * v;
        }
        caches[layer].attn_concat_out = concat_out;
        return concat_out * WO[layer];
    }
};

int main()
{
    // Instanciamos la red
    VisionTransformer vit;
    string model_filename = "vit_mnist_weights.bin";

    cout << "===========================================" << endl;
    cout << "   VISION TRANSFORMER (ViT) - C++ & EIGEN  " << endl;
    cout << "===========================================" << endl;
    cout << "1. Entrenar modelo (Train)" << endl;
    cout << "2. Probar imagen propia (Predict)" << endl;
    cout << "Selecciona una opcion: ";

    int option;
    cin >> option;

    // ------------------- MODO ENTRENAMIENTO -------------------
    if (option == 1)
    {
        cout << "\n¿Quieres continuar el entrenamiento anterior? (1: Si, 0: No - Desde cero): ";
        int resume;
        cin >> resume;

        if (resume == 1)
        {
            vit.load_model(model_filename);
        }

        cout << "Cargando MNIST..." << endl;
        auto train_imgs = load_images("train-images-idx3-ubyte", 60000);
        auto train_lbls = load_labels("train-labels-idx1-ubyte", 60000);
        auto test_imgs = load_images("t10k-images-idx3-ubyte", 10000);
        auto test_lbls = load_labels("t10k-labels-idx1-ubyte", 10000);

        // Normalización crítica
        for (auto &img : train_imgs)
            img.array() -= 0.1307f;
        for (auto &img : test_imgs)
            img.array() -= 0.1307f;

        cout << "Iniciando entrenamiento..." << endl;

        int epochs = 1;
        float lr = 0.0001f;
        int batch_size = 32;

        int N = train_imgs.size();
        vector<int> idx(N);
        iota(idx.begin(), idx.end(), 0);

        for (int ep = 0; ep < epochs; ++ep)
        {
            shuffle(idx.begin(), idx.end(), vit.rng);
            float loss_sum = 0.0f;
            int correct = 0;
            vit.reset_grads();

            for (int ii = 0; ii < N; ++ii)
            {
                int i = idx[ii];

                // 1. Forward
                VectorXf logits = vit.forward(train_imgs[i]);

                // 2. Loss & Softmax
                int target = train_lbls[i];
                float m = logits.maxCoeff();
                VectorXf exps = (logits.array() - m).exp();
                VectorXf prob = exps / exps.sum();

                int pred;
                prob.maxCoeff(&pred);
                if (pred == target)
                    correct++;
                loss_sum -= logf(prob(target) + 1e-8f);

                // 3. Backward
                VectorXf dlogits = prob;
                dlogits(target) -= 1.0f;
                vit.backward(dlogits);

                // 4. Update (Mini-batch)
                if ((ii + 1) % batch_size == 0)
                {
                    vit.update(lr / batch_size);
                    vit.reset_grads();
                }

                // Logging parcial
                if ((ii + 1) % 2000 == 0)
                {
                    cout << "Epoch " << ep + 1 << " [" << (ii + 1) << "/" << N
                         << "] Acc parcial: " << (100.0 * correct / (ii + 1)) << "%" << endl;
                }
            }

            cout << "--- Fin Epoch " << ep + 1 << " Loss: " << loss_sum / N
                 << " Acc: " << (100.0 * correct / N) << "% ---" << endl;

            // Decaer learning rate y GUARDAR MODELO
            lr *= 0.7f;
            vit.save_model(model_filename); // Guardado automático por epoch
        }

        // Validación final en Test Set
        cout << "\nValidando en TEST set (10,000 imagenes)..." << endl;
        int test_correct = 0;
        for (int i = 0; i < 10000; ++i)
        {
            VectorXf logits = vit.forward(test_imgs[i]);
            int pred;
            logits.maxCoeff(&pred);
            if (pred == test_lbls[i])
                test_correct++;
        }
        cout << "TEST ACCURACY FINAL: " << 100.0 * test_correct / 10000 << "%" << endl;
    }
    // ------------------- MODO PREDICCION -------------------
    else if (option == 2)
    {
        // Cargar pesos obligatoriamente
        vit.load_model(model_filename);

        while (true)
        {
            string img_path;
            cout << "\nIngresa la ruta de tu imagen (ej: numero.png) o escribre 'salir': ";
            cin >> img_path;

            if (img_path == "salir")
                break;

            // Llamada a la funcion que usa stb_image
            vit.predict_custom_image(img_path);
        }
    }

    return 0;
}
