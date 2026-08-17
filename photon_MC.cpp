#include <cstddef>
#include <iostream>
#include <cstdint>
#include <random>
#include <eigen3/Eigen/Dense>

struct Photon {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    std::size_t zoneIndex = 0;
};

class Rng {
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
    };

    double uniformOpen01() {
        double u = 0.0;
        do {
            u = uniform_(engine_); //A classe std::uniform_real_distribution<double> está usando um operator() no lugar de uma função membro que retorna um double e recebe a seed
        } while (!(u > 0.0 && u < 1.0)); //Garante que tá no intervalo aberto (0,1), já que o objeto uniform_ retorna valores em [0,1)
        return u;
    };

    double opticalDepth() {
        return exponential_(engine_);
    };
};

int main() {
    Rng rng(123, 1);

    for (int i = 0; i < 5; i++){
        std::cout << rng.uniformOpen01() << " " << rng.opticalDepth() << "\n";
    }
}
