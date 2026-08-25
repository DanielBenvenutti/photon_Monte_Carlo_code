#include <cstddef>
#include <iostream>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>
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

class LegendrePhase { //Amostragem da distribuição do cosseno do ângulo de espalhamento no referencial laboratório reescrita como uma soma de polinômios de Legendre
private:
    std::vector<double> beta_;
    double rejectionBound_ = 1.0;

    void validateNonNegative() const {
        const int gridPoints = 100000;
        double minimum = std::numeric_limits<double>::infinity();
        double minimumX = 0.0;

        for (int i = 0; i <= gridPoints; i++) {
            const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(gridPoints);
            const double value = evaluateDistribution(x);

            if (value < minimum) {
                minimum = value;
                minimumX = x;
            }
        }

        if (minimum < -1.0E-9) {
            std::ostringstream message;
            message << "Os coeficientes beta não definem uma densidade angular de espalhamento"
                    << "não negativa: min D(x) = " << minimum << " em x = " << minimumX << ".";
            throw std::runtime_error(message.str());
        }
    }

public:
    explicit LegendrePhase(std::vector<double> beta)
        : beta_(std::move(beta)) {
        if (beta_.empty()) { //Condição de bom comportamento da distribuição angular (deve existir algum coeficiente)
            throw std::runtime_error("A lista de coeficientes beta não pode ser vazia.");
        }
        if (std::abs(beta_.front() - 1.0) > 1.0E-10) { //Condição de bom comportamento da distribuição angular (o primeiro coeficiente deve ser unitário)
            throw std::runtime_error("A normalização da função de fase exige beta[0] = 1.");
        }

        rejectionBound_ = std::accumulate(
            beta_.begin(), beta_.end(), 0.0,
            [](double sum, double value) { return sum + std::abs(value); } //Função lambda
        );

        if (!(rejectionBound_ > 0.0)) { //Condição de bom comportamento da distribuição angular (como já tem uma condição para não termos todos coeficientes nulo, aqui serve essencialmente para evitar NaN)
            throw std::runtime_error("Função de fase degenerada.");
        }
        validateNonNegative(); //Condição de bom comportamento da distribuição angular (garante numericamente que a distribuição é maior que zero para todo cosseno de \theta)
    }

    double sampleCosineFromDistribution(Rng& rng) const {
        const std::uint64_t maxAttempts = 100000000;

        for (std::uint64_t attempt = 0; attempt < maxAttempts; attempt++) {
            const double x = 2.0 * rng.uniformOpen01() - 1.0;
            double evaluationAuxiliary = evaluateDistribution(x);

            if (rng.uniformOpen01() < evaluationAuxiliary / rejectionBound_) { //Método de amostragem por rejeição, rejectionBound_ é um majorante
                return x;
            }
        }

        throw std::runtime_error("Amostragem da função de fase não convergiu, verifique o vetor dos coeficientes beta.");
    }

    double evaluateDistribution(double x) const {
        double sum = beta_.front();
        if (beta_.size() == 1) {
            return sum; //Parte constante da soma (primeiro termo)
        }

        double pPrevious = 1.0;
        double pCurrent = x;
        sum += beta_[1] * pCurrent; //Parte linear da soma (primeiro + segundo termos)

        for (std::size_t ell = 1; ell + 1 < beta_.size(); ++ell) {
            const double ellDouble = static_cast<double>(ell);
            const double pNext = ((2.0 * ellDouble + 1.0) * x * pCurrent - ellDouble * pPrevious) / (ellDouble + 1.0); //Relação de recorrência dos polinômios

            sum += beta_[ell + 1] * pNext; //Restante da soma até o (ell+1)-ésimo termo. "p" representa o polinômio de Legendre
            pPrevious = pCurrent;
            pCurrent = pNext;
        }

        return sum;
    }

    const std::vector<double>& beta() const {
        return beta_;
    }
};

struct Zone {
    double rInner = 0.0;
    double rOuter = 1.0;
    double sigmaT = 1.0; //Seção de choque macroscópica total da zona
    double omega = 0.0; //Probabilidade de espalhamento na zona. Ou seja, omega = sigmaS/sigmaT, onde sigmaS é a seção de choque macroscópica de espalhamento da zona
    double blackbodyIntensity = 0.0; //Associada a emissão de fótons na zona
};

enum class OuterAngularDistribution { //Controla como fótons da fonte externa entram no domínio
    Diffuse,
    Radial
};

struct Config { //Configuração inicial do problema (passada no main)
    double rhoInner = 0.0; //Reflexividade nas fronteiras globais
    double rhoOuter = 0.0;
    double outerSourceIntensity = 1.0; //Associada a emissão de fótons nas fronteiras
    double innterSourceIntensity = 0.0;
    OuterAngularDistribution outerAngular = OuterAngularDistribution::Diffuse;
    std::vector<double> beta = {1.0};
    std::vector<Zone> zones = {{1.0, 4.0, 1.0, 0.5, 0.2}};
    std::uint64_t histories = 10000000; //Número inicial de fótons simulados
    std::size_t batches = 32; //Semelhante as execuções no GNTIMC, ou seja, divide histories em n execuções, o que representaria n simulações Monte Carlo distintas e assim fornece uma distribuição de parâmetros de saída para uma análise estatística
    std::size_t threads = 0;
    std::uint64_t seed = 202607001;
    std::uint64_t maxEvents = 1000; //Número máximo permitido de passos Monte Carlo por história
};

enum class SourceKind { //Indica possíveis localizações/tipos das fontes criadas
    OuterBoundary,
    InnerBoundary,
    VolumeZone
};

struct SourceComponent { //Inicializa uma fonte com algumas de suas características
    SourceKind kind = SourceKind::OuterBoundary;
    std::size_t zoneIndex = 0;
    double strenght = 0.0;
    double cumulative = 0.0; //Usado na amostragem da fonte do fóton inicial
};

struct Model { //Estado físico completo para realizar a simulação com todas as informações necessárias, incluindo dados iniciais passados na estrutura Config
    std::vector<Zone> zones;
    double a = 0.0; //Raio global interno (a) e externo (b) do domínio que será obtido de zones.front() e zones.back()
    double b = 1.0;
    double rhoInner = 0.0; //Associado a reflexividade das fronteiras
    double rhoOuter = 0.0;
    double outerSourceIntensity = 1.0;
    double innerSourceIntensity = 0.0;
    OuterAngularDistribution outerAngular = OuterAngularDistribution::Diffuse;
    LegendrePhase phase;
    std::vector<SourceComponent> sources; //Vai armazenar as fontes existentes no problema
    double totalSourceStrength = 0.0;
    double geometryEpsilon = 1.0E-12; //Evita problemas geométricos para fótons que cruzam, refletem ou nascem em determinada superfície

    Model(
        std::vector<Zone> zonesIn,
        double rhoInnerIn,
        double rhoOuterIn,
        double outerSourceIntensityIn,
        double innerSourceIntensityIn,
        OuterAngularDistribution outerAngularIn,
        LegendrePhase phaseIn
    )
        : zones(std::move(zonesIn)),
          a(zones.front().rInner),
          b(zones.back().rOuter),
          rhoInner(rhoInnerIn),
          rhoOuter(rhoOuterIn),
          outerSourceIntensity(outerSourceIntensityIn),
          innerSourceIntensity(innerSourceIntensityIn),
          outerAngular(outerAngularIn),
          phase(std::move(phaseIn)) {}
};

struct BatchTally { //Contagem de dados relevantes por execução/batch
    std::uint64_t histories = 0;
    std::uint64_t outerIn = 0;
    std::uint64_t outerOut = 0;
    std::uint64_t innerIn = 0;
    std::uint64_t innerOut = 0;
    std::uint64_t absorbedMedium = 0;
    std::uint64_t terminatedOuter = 0;
    std::uint64_t terminatedInner = 0;
    std::uint64_t killedMaxEvents = 0;
    std::uint64_t killedInvalidGeometry = 0;
    std::uint64_t collisions = 0;
    std::uint64_t scatterings = 0;
    std::uint64_t outerHits = 0;
    std::uint64_t innerHits = 0;
    std::uint64_t interfaceCrossings = 0;

    BatchTally& operator+=(const BatchTally& other) { //Função/Sobrecarga de operador para somar as estatísticas de cada execução/batch
        histories += other.histories;
        outerIn += other.outerIn;
        outerOut += other.outerOut;
        innerIn += other.innerIn;
        innerOut += other.innerOut;
        absorbedMedium += other.absorbedMedium;
        terminatedOuter += other.terminatedOuter;
        terminatedInner += other.terminatedInner;
        killedMaxEvents += other.killedMaxEvents;
        killedInvalidGeometry += other.killedInvalidGeometry;
        collisions += other.collisions;
        scatterings += other.scatterings;
        outerHits += other.outerHits;
        innerHits += other.innerHits;
        interfaceCrossings += other.interfaceCrossings;
        return *this;
    }
};

void validateZones(const std::vector<Zone>& zones){
    if(zones.empty()) {
        throw std::runtime_error("O modelo precisa de pelo menos uma zona radial.");
    }

    const double tolerance = 1.0E-11;
    for (std::size_t i = 0; i < zones.size(); i++) {
        const Zone& zone = zones[i];

        if (zone.rInner < 0.0 || !(zone.rOuter > zone.rInner)) {
            throw std::runtime_error("Raios inválidos em uma zona radial.");
        }
        if (zone.sigmaT < 0.0) {
            throw std::runtime_error("Seção de choque total não pode ser negativa.");
        }
        if (zone.omega < 0.0 || zone.omega > 1.0) {
            throw std::runtime_error("Albedo é uma probabilidade e por isso deve estar contigo em [0,1].");
        }
        if (zone.blackbodyIntensity < 0.0) {
            throw std::runtime_error("Intensidade de corpo negro não pode ser negativa.");
        }

        if (i > 0) {
            const double scale = std::max({ //Obtendo uma escala das dimensões envolvidas para ajustar a tolerância nas checagens, evitando erros de limitação computacional
                1.0,
                std::abs(zones[i - 1].rOuter),
                std::abs(zone.rInner)
            });
            if (std::abs(zones[i - 1].rOuter - zone.rInner) > tolerance * scale) {
                throw std::runtime_error("As zonas radiais devem ser contíguas e ordenadas.");
            }
        }
    }
};

void buildSources(Model& model) { //Inicializando as fontes no modelo para posteriormente serem amostradas de acordo com sua força (cada fóton inicial vai nascer de uma destas fontes)
    model.sources.clear();
    model.totalSourceStrength = 0.0;

    const auto addSource = [&] (SourceKind kind,
        std::size_t zoneIndex,
        double strength
    ) {
        if (strength <= 0.0) {
            return;
        }

        model.totalSourceStrength += strength;
        model.sources.push_back({
            kind,
            zoneIndex,
            strength,
            model.totalSourceStrength
        });
    };

    addSource( //Adicionando fonte extremo externo
        SourceKind::OuterBoundary,
        model.zones.size() - 1, //Está na última zona
        model.b * model.b * model.outerSourceIntensity
    );

    if (model.a > 0.0) { //Adicionando fonte extremo interno (se existir)
        addSource(
            SourceKind::InnerBoundary,
            0, //Está na primeira zona
            model.a * model.a * model.innerSourceIntensity
        );
    }

    for (std::size_t i = 0; i < model.zones.size(); i++) {
        const Zone& zone = model.zones[i];
        const double sigmaA = zone.sigmaT * (1.0 - zone.omega);
        const double radialIntegral = (std::pow(zone.rOuter, 3) - std::pow(zone.rInner, 3)) / 3.0;
        const double strength = 4.0 * sigmaA * zone.blackbodyIntensity * radialIntegral;
        addSource( //Adicionando fonte para cada zona
            SourceKind::VolumeZone,
            i,
            strength);
    }

    if (!(model.totalSourceStrength > 0.0)) {
        throw std::runtime_error("A força total das fontes é zero. Ajuste as intensidades no main ou use Ib>0 em alguma zona.");
    }

    const double minimumWidth = std::accumulate(
        model.zones.begin(),
        model.zones.end(),
        std::numeric_limits<double>::infinity(),
        [](double current, const Zone& zone) {
            return std::min(current, zone.rOuter - zone.rInner);
        }
    );

    const double roundoffScale = 128.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, model.b); //Ajusta a tolerância geométrica para evitar problemas com fótons localizados nas fronteiras
    model.geometryEpsilon = std::max(roundoffScale, 1.0E-12 * minimumWidth);
    model.geometryEpsilon = std::min(model.geometryEpsilon, 1.0E-8 * minimumWidth);
};

double distanceToSphere( //A ideia é encontrar para quais deslocamentos, dada a direção (\vec{d}) do fóton, a posição (\vec{x}(s) = \vec{p} + s\vec{d}) deste irá intersectar a esfera testada. Basicamente |\vec{p} + s\vec{d}|^{2} = R^{2}
    const Eigen::Vector3d& position,
    const Eigen::Vector3d& direction,
    double radius,
    double epsilon
) {
    if (!(radius > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }

    const double pDotD = position.dot(direction);
    const double c = position.squaredNorm() - radius * radius;
    double discriminant = pDotD * pDotD - c; //Bhaskara

    const double scale = std::max({1.0, pDotD * pDotD, std::abs(c)});
    const double roundoffTolerance = 128.0 * std::numeric_limits<double>::epsilon() * scale; //Ajusta a tolerância geométrica para capturar corretamente fótons que tangenciam a esféra avaliada
    if (discriminant < 0.0 && discriminant > -roundoffTolerance) {
        discriminant = 0.0; //Fóton tangencial a superfície da esféra
    }
    if (discriminant < 0.0) {
        return std::numeric_limits<double>::infinity(); //Não existe deslocamento 's' tal que o fóton atravesse a esféra, ou seja, que satisfaça a igualdade |\vec{x}(s)|^{2} = R^{2}
    }

    const double root = std::sqrt(discriminant);
    const double first = -pDotD - root; //Primeira raiz
    const double second = -pDotD + root; //Segunda raiz;
    const double threshold = 8.0 * epsilon; //Evitar problemas geométricos se distâncias muito pequenas satisfazem a equação relevante, o que poderia induzir que um fóton atravessou novamente a mesma fronteira, e garantir um deslocamento positivo

    double result = std::numeric_limits<double>::infinity();
    if (first > threshold) { //Além da questão geométrica, as comparações com o threshold garantem que o deslocamento seja positivo. Caso seja negativo, o fóton teria que trocar de direção no deslocamento, o que é impossível
        result = first;
    }
    if (second > threshold) {
        result = std::min(result, second); //Se ambas raízes são positivas, o fóton eventualmente vai atingir os contornos da esféra em ambas as soluções. Obviamente, o deslocamento escolhido será o menor, já que o fóton vai atingir a superfície primeiro no respectivo deslocamento
    }
    return result;
}

const SourceComponent& selectSource(const Model& model, Rng& rng) {
    const double target = rng.uniformOpen01() * model.totalSourceStrength;

    const auto source = std::lower_bound(
        model.sources.begin(),
        model.sources.end(),
        target,
        [](const SourceComponent& component, double value) {
            return component.cumulative < value;
        }
    );

    if (source == model.sources.end()) {
        const double last = model.sources.back().cumulative;
        constexpr double tolerance = 1.0E-6;
        if (target <= last + tolerance) {
            return model.sources.back();
        }
        throw std::out_of_range("O valor amostrado excede o valor máximo da distribuição acumulada mais uma tolerância. Revise o código.");
    }
    return *source;
}

Photon launchPhoton(
    const Model& model,
    const SourceComponent& source,
    Rng& rng,
    BatchTally& tally
) {
    Photon photon;

    if (source.kind == SourceKind::OuterBoundary) {
        const Eigen::Vector3d radialOut = randomUnitVector(rng);
        photon.position = model.b * radialOut;
        if (model.outerAngular == OuterAngularDistribution::Diffuse) {
            photon.direction = sampleCosineWeightedHemisphere(-radialOut, rng);
        } else {
            photon.direction = -radialOut;
        }
        photon.zoneIndex = model.zones.size() - 1;
        ++tally.outerIn; //Contabiliza fótons entrando através da superfície extrema externa do domínio (In é o sentido radial do fóton)
        photon.position += model.geometryEpsilon * photon.direction;
        return photon;
    }

    if (source.kind == SourceKind::InnerBoundary) {
        const Eigen::Vector3d radialOut = randomUnitVector(rng);
        photon.position = model.a * radialOut;
        photon.direction = sampleCosineWeightedHemisphere(radialOut, rng);
        photon.zoneIndex = 0;
        ++tally.innerOut; //Contabiliza fótons entrando através da superfície extrema interna do domínio (Out é o sentido radial do fóton)
        photon.position += model.geometryEpsilon * photon.direction;
        return photon;
    }

    const Zone& zone = model.zones[source.zoneIndex];
    const double innerCubed = std::pow(zone.rInner, 3);
    const double outerCubed = std::pow(zone.rOuter, 3);
    const double radiusCubed = innerCubed + rng.uniformOpen01() * (outerCubed - innerCubed); //Raio aleatório dentro da casca, definindo uma superfície esférica na qual o fóton estará

    photon.position = std::cbrt(radiusCubed) * randomUnitVector(rng); //Posição aleatória na superfície esférica definida anteriormente
    photon.direction = randomUnitVector(rng); //Direção inicial do fóton na zona é isotrópica
    photon.zoneIndex = source.zoneIndex;
    return photon;
}

void simulateHistory(
    const Model& model,
    const Config& config,
    Rng& rng,
    BatchTally& tally
) {
    const SourceComponent& source = selectSource(model, rng);
    Photon photon = launchPhoton(model, source, rng, tally);
    double remainingOpticalDepth = rng.opticalDepth();

    for (std::uint64_t event = 0; event < config.maxEvents; ++event){
        const Zone& zone = model.zones[photon.zoneIndex];

        const double distanceToLower =
            (zone.rInner > 0.0)
            ? distanceToSphere(
                photon.position,
                photon.direction,
                zone.rInner,
                model.geometryEpsilon
            )
            : std::numeric_limits<double>::infinity();

        const double distanceToUpper = distanceToSphere(
            photon.position,
            photon.direction,
            zone.rOuter,
            model.geometryEpsilon
        );

        const bool hitsLowerInterface = distanceToLower < distanceToUpper; //O fóton atingirá primeiro a superfície associada ao menor dos deslocamentos, por isso é o que interessa
        const double distanceToInterface = hitsLowerInterface ? distanceToLower : distanceToUpper;

        if (!std::isfinite(distanceToInterface)) {
            ++tally.killedInvalidGeometry;
            return;
        }

        const double opticalDepthToInterface = zone.sigmaT * distanceToInterface; //Este produto (que representa o deslocamento máximo que o fóton pode ter até encontrar uma superfície) permite comparar diretamente com o valor amostrado da exponencial (que representa o deslocamento do fóton no passo)
        const double comparisonTolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, remainingOpticalDepth, opticalDepthToInterface});

        const bool collisionBeforeInterface = zone.sigmaT > 0.0 && remainingOpticalDepth < opticalDepthToInterface - comparisonTolerance; //Comparação relacionada ao comentário acima

        if (collisionBeforeInterface) {
            const double collisionDistance = remainingOpticalDepth / zone.sigmaT;
            photon.position += collisionDistance*photon.direction;
            ++tally.collisions;

            if (rng.uniformOpen01() < zone.omega) {
                ++tally.scatterings;
                const double cosineScatter = model.phase.sampleCosineFromDistribution(rng);
                photon.direction = sampleScatteredDirection(photon.direction, cosineScatter, rng);
                remainingOpticalDepth = rng.opticalDepth(); //Já amostra para a próxima colisão, já que remainingOpticalDepth foi inicializado fora do laço dos passos
                continue; //Para a iteração atual do for e pula para o próximo passo
            }

            ++tally.absorbedMedium;
            return; //Fim de história
        }

        photon.position += distanceToInterface * photon.direction; //O fóton irá parar na próxima superfície/interface, já que o meio terá outras características
        remainingOpticalDepth = std::max(0.0, remainingOpticalDepth - opticalDepthToInterface); //Já prepara para o próximo passo. Neste modelo, parte da profundidade ótica é "consumida" até encontrar a superfície e a restante é usada na próxima região. Equivale ao modelo usado no transporte de nêutrons

        if (hitsLowerInterface) {
            if (photon.zoneIndex > 0) {
                --photon.zoneIndex;
                ++tally.interfaceCrossings;
                photon.position += model.geometryEpsilon*photon.direction; //Evitando problemas geométricos que podem estar associados a limitação de computadores, como estar em uma posição que ainda caracterize a zona anterior por estas limitações. Note que, por construção, photon.direction aponta para o centro
                continue; //Para a iteração atual do for e pula para o próximo passo
            }

            if (model.a > 0.0) { //Fóton estava na zona válida mais interna do domínio e assim atingiu a superfície extrema interna deste
                ++tally.innerHits;
                ++tally.innerIn;

                if (rng.uniformOpen01() < model.rhoInner) { //O fóton pode refletir nas superfícies extremas (de acordo com os rhos)
                    const Eigen::Vector3d radialOut = photon.position.normalized();
                    photon.direction = sampleCosineWeightedHemisphere(radialOut, rng);
                    ++tally.innerOut;
                    photon.position += model.geometryEpsilon*photon.direction;
                    if (remainingOpticalDepth <= comparisonTolerance) { //remainingOpticalDepth é essencialmente nulo, considerando a tolerância, e assim amostramos uma nova profundidade
                        remainingOpticalDepth = rng.opticalDepth();
                    }
                    continue; //Para a iteração atual do for e pula para o próximo passo
                }

                ++tally.terminatedInner; //Fóton escapou pela superfície extrema interna
                return; //Fim de história
            }
        } else {
            if (photon.zoneIndex + 1 < model.zones.size()) {
                ++photon.zoneIndex;
                ++tally.interfaceCrossings;
                photon.position += model.geometryEpsilon*photon.direction;
                continue;
            }

            ++tally.outerHits;
            ++tally.outerOut;

            if (rng.uniformOpen01() < model.rhoOuter) { //O fóton pode refletir nas superfícies extremas (de acordo com os rhos)
                const Eigen::Vector3d radialOut = photon.position.normalized();
                photon.direction = sampleCosineWeightedHemisphere(-radialOut, rng);
                ++tally.outerIn;
                photon.position += model.geometryEpsilon*photon.direction;
                if (remainingOpticalDepth <= comparisonTolerance) {
                    remainingOpticalDepth = rng.opticalDepth();
                }
                continue;
            }

            ++tally.terminatedOuter; //Fóton escapou pela superfície extrema interna
            return; //Fim de história
        }
    }

    ++tally.killedMaxEvents; //Fóton atingiu o máximo de passos Monte Carlo permitidos para sua história
}

int main() {
    Config config;
    config.maxEvents = 1000;

    std::vector<Zone> zones = {{1.0, 4.0, 0.0, 0.0, 0.0}};
    LegendrePhase phase({1.0});
    Model model(
        zones,
        0.0, 0.0,
        1.0, 0.0,
        OuterAngularDistribution::Radial,
        phase
    );
    buildSources(model);

    BatchTally tally;
    Rng rng(123, 0);
    const std::uint64_t histories = 10000;

    for (std::uint64_t i = 0; i < histories; i++) {
        simulateHistory(model, config, rng, tally);
    }

    std::cout << "outerIn=" << tally.outerIn << "\n";
    std::cout << "innerIn=" << tally.innerIn << "\n";
    std::cout << "R=" << static_cast<double>(tally.outerOut) / tally.outerIn << "\n";
    std::cout << "T=" << static_cast<double>(tally.innerIn) / tally.outerIn << "\n";
    return 0;
}












