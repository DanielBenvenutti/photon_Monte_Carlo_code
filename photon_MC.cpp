#include <cstddef>
#include <iostream>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>
#include <eigen3/Eigen/Dense>

struct Photon { //Estado do fóton
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    std::size_t zoneIndex = 0;
};

class Rng { //Gerador de números aleatórios a partir de distribuições específicas
private:
    std::mt19937_64 engine_; //objeto que gera números aleatórios inteiros de 64 bit
    std::uniform_real_distribution<double> uniform_{0.0, 1.0}; //objeto que recebe o número aleatório do engine_ e converte de acordo com a distr
    std::exponential_distribution<double> exponential_{1.0};

public:
    Rng(std::uint64_t baseSeed, std::size_t streamIndex){
        const std::uint64_t stream = static_cast<std::uint64_t>(streamIndex);
        std::seed_seq seed{ //Preparando a seed. Objetos da classe seed_seq operam com 32 bits, por isso estamos aproveitando todos os 64 bits de cada variável
            static_cast<std::uint32_t>(baseSeed), //variável de 32 bits definida pelos 32 bits inferiores de baseSeed
            static_cast<std::uint32_t>(baseSeed >> 32U), //transforma os 32 bits superiores de baseSeed para bits inferiores e converte estes para uma variável de 32 bits
            static_cast<std::uint32_t>(stream),
            static_cast<std::uint32_t>(stream >> 32U),
        };
        engine_.seed(seed); //Utiliza a seed preparada na construção do objeto da classe Rng para inicializar o gerador de números aleatórios
    }

    double uniformOpen01() {
        double u = 0.0;
        do {
            u = uniform_(engine_); //A classe std::uniform_real_distribution<double> está usando um operator() no lugar de uma função membro que retorna um double e recebe a seed
        } while (!(u > 0.0 && u < 1.0)); //Garante que tá no intervalo aberto (0,1), já que o objeto uniform_ retorna valores em [0,1)
        return u;
    }

    double opticalDepth() {
        return exponential_(engine_);
    }
};

Eigen::Vector3d randomUnitVector(Rng& rng) { //Amostra uma direção aleatória na esfera unitária
    const double z = 2.0 * rng.uniformOpen01() - 1.0;
    const double phi = 2.0 * M_PI * rng.uniformOpen01();
    const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));

    return Eigen::Vector3d(
        radial * std::cos(phi),
        radial * std::sin(phi),
        z
    );
};

void tangetFrame(const Eigen::Vector3d& normal, Eigen::Vector3d& tangent1, Eigen::Vector3d& tangent2) { //Criando uma base ortonormal local em torno do vetor normal, gerada pela base geral
    const Eigen::Vector3d n = normal.normalized();
    tangent1 = n.unitOrthogonal();
    tangent2 = n.cross(tangent1);
};

Eigen::Vector3d sampleCosineWeightedHemisphere(const Eigen::Vector3d& normalIntoMedium, Rng& rng) { //Amostra uma direção aleatória, enviesada para maior probabilidade em torno da normal (assumindo emissão difusa [Lambertiano] \rightarrow densidade angular proporcional ao cosseno)
    const Eigen::Vector3d normal = normalIntoMedium.normalized();
    Eigen::Vector3d tangent1;
    Eigen::Vector3d tangent2;
    tangetFrame(normal, tangent1, tangent2); //Criação da base ortonormal em torno do normalIntoMedium normalizado

    const double z = std::sqrt(rng.uniformOpen01()); //A ponderação para maior probabilidade em torno de normal vem do uso da raiz, além disso, fica claro a restrição a um único hemisfério
    const double phi = 2.0 * M_PI * rng.uniformOpen01();
    const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
    const Eigen::Vector3d localDirection( //Direção aleatória amostrada gerada na base local
        radial * std::cos(phi),
        radial * std::sin(phi),
        z
    );

    Eigen::Matrix3d localToWorld; //Criação da matriz de transformação de base local para global
    localToWorld.col(0) = tangent1;
    localToWorld.col(1) = tangent2;
    localToWorld.col(2) = normal;

    return (localToWorld * localDirection).normalized(); //Retorno da direção ponderada pela normal gerada pela base geral
};

Eigen::Vector3d sampleScatteredDirection(const Eigen::Vector3d& oldDirection, double cosineScatter, Rng& rng) { //Amostra uma direção aleatória após o espalhamento com base no \cos(\theta), obtido de alguma distribuição e passado como argumento de função [\theta é o ângulo de espalhamento]
    const Eigen::Vector3d oldDirectionNormalized = oldDirection.normalized();
    Eigen::Vector3d tangent1;
    Eigen::Vector3d tangent2;
    tangetFrame(oldDirectionNormalized, tangent1, tangent2); //Criação da base ortonormal em torno do oldDirection normalizado

    const double z = cosineScatter;
    const double phi = 2.0 * M_PI * rng.uniformOpen01();
    const double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
    const Eigen::Vector3d localDirection( //Direção aleatória amostrada gerada na base local
        radial * std::cos(phi),
        radial * std::sin(phi),
        z
    );

    Eigen::Matrix3d localToWorld; //Criação da matriz de transformação de base local para global
    localToWorld.col(0) = tangent1;
    localToWorld.col(1) = tangent2;
    localToWorld.col(2) = oldDirectionNormalized;

    return (localToWorld * localDirection).normalized(); //Retorno da direção ponderada pela normal gerada pela base geral
};


int main() {
    Rng rng(123, 0);

    double meanZ = 0.0;
    const int samples = 100000;

    for(int i = 0; i < samples; i++) {
        const Eigen::Vector3d d = randomUnitVector(rng);
        meanZ += d.z();

        if (std::abs(d.norm() - 1.0) > 1.0E-12) {
            std::cerr << "Direcao nao unitaria\n";
            return 1;
        }
    }

    meanZ /= samples;
    std::cout << "mean Z = " << meanZ << "\n";
    return 0;
}












