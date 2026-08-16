#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

Vec3 operator+(Vec3 a, const Vec3& b) {
    a += b;
    return a;
}

Vec3 operator*(double s, const Vec3& v) {
    return {s * v.x, s * v.y, s * v.z};
}

Vec3 operator/(const Vec3& v, double s) {
    return {v.x / s, v.y / s, v.z / s};
}

double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

double norm2(const Vec3& v) {
    return dot(v, v);
}

double norm(const Vec3& v) {
    return std::sqrt(norm2(v));
}

Vec3 normalized(const Vec3& v) {
    const double n = norm(v);
    if (!(n > 0.0)) {
        throw std::runtime_error("Tentativa de normalizar um vetor nulo.");
    }
    return v / n;
}

struct SplitMix64 {
    std::uint64_t state;

    explicit SplitMix64(std::uint64_t seed) : state(seed) {}

    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31U);
    }
};

class Rng {
public:
    explicit Rng(std::uint64_t seed) : sm_(seed), state0_(sm_.next()), state1_(sm_.next()) {
        if (state0_ == 0 && state1_ == 0) {
            state1_ = 1;
        }
    }

    std::uint64_t nextU64() {
        // xoroshiro128+
        const std::uint64_t s0 = state0_;
        std::uint64_t s1 = state1_;
        const std::uint64_t result = s0 + s1;
        s1 ^= s0;
        state0_ = rotl(s0, 55) ^ s1 ^ (s1 << 14);
        state1_ = rotl(s1, 36);
        return result;
    }

    double uniformOpen01() {
        // 53 bits, strictly inside (0,1).
        constexpr double inv = 1.0 / 9007199254740992.0; // 2^-53
        const std::uint64_t mantissa = nextU64() >> 11U;
        return (static_cast<double>(mantissa) + 0.5) * inv;
    }

    double opticalDepth() {
        return -std::log(uniformOpen01());
    }

private:
    static std::uint64_t rotl(const std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    SplitMix64 sm_;
    std::uint64_t state0_;
    std::uint64_t state1_;
};

Vec3 randomUnitVector(Rng& rng) {
    const double z = 2.0 * rng.uniformOpen01() - 1.0;
    const double phi = 2.0 * kPi * rng.uniformOpen01();
    const double rxy = std::sqrt(std::max(0.0, 1.0 - z * z));
    return {rxy * std::cos(phi), rxy * std::sin(phi), z};
}

void tangentFrame(const Vec3& n, Vec3& t1, Vec3& t2) {
    const Vec3 helper = (std::abs(n.z) < 0.9) ? Vec3{0.0, 0.0, 1.0} : Vec3{0.0, 1.0, 0.0};
    t1 = normalized(cross(helper, n));
    t2 = cross(n, t1);
}

Vec3 sampleCosineHemisphere(const Vec3& normalIntoMedium, Rng& rng) {
    const Vec3 n = normalized(normalIntoMedium);
    Vec3 t1, t2;
    tangentFrame(n, t1, t2);

    const double cosine = std::sqrt(rng.uniformOpen01());
    const double sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
    const double phi = 2.0 * kPi * rng.uniformOpen01();

    return normalized(cosine * n + sine * (std::cos(phi) * t1 + std::sin(phi) * t2));
}

Vec3 rotateDirection(const Vec3& oldDirection, double cosineScatter, Rng& rng) {
    const Vec3 w = normalized(oldDirection);
    Vec3 u, v;
    tangentFrame(w, u, v);

    const double sineScatter = std::sqrt(std::max(0.0, 1.0 - cosineScatter * cosineScatter));
    const double phi = 2.0 * kPi * rng.uniformOpen01();
    return normalized(cosineScatter * w +
                      sineScatter * (std::cos(phi) * u + std::sin(phi) * v));
}

class LegendrePhase {
public:
    explicit LegendrePhase(std::vector<double> beta) : beta_(std::move(beta)) {
        if (beta_.empty()) {
            throw std::runtime_error("A lista de coeficientes beta nao pode ser vazia.");
        }
        if (std::abs(beta_[0] - 1.0) > 1.0e-10) {
            throw std::runtime_error("A normalizacao da funcao de fase exige beta[0] = 1.");
        }

        rejectionBound_ = 0.0;
        for (double value : beta_) {
            rejectionBound_ += std::abs(value);
        }
        if (!(rejectionBound_ > 0.0)) {
            throw std::runtime_error("Funcao de fase degenerada.");
        }

        // Verificacao numerica de nao-negatividade da serie S(x)=sum beta_l P_l(x).
        double minValue = kInf;
        double minX = 0.0;
        constexpr int grid = 100000;
        for (int i = 0; i <= grid; ++i) {
            const double x = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(grid);
            const double value = series(x);
            if (value < minValue) {
                minValue = value;
                minX = x;
            }
        }
        if (minValue < -1.0e-9) {
            std::ostringstream oss;
            oss << "Os coeficientes beta nao definem uma densidade de espalhamento nao-negativa: "
                << "min S(x) = " << minValue << " em x = " << minX << ".";
            throw std::runtime_error(oss.str());
        }
    }

    double sampleCosine(Rng& rng) const {
        // O alvo marginal e f(x)=0.5*S(x), x in [-1,1]. A proposta uniforme
        // tem densidade 0.5. Como |P_l(x)|<=1, sum|beta_l| e um majorante rigoroso.
        for (std::uint64_t attempt = 0; attempt < 100000000ULL; ++attempt) {
            const double x = 2.0 * rng.uniformOpen01() - 1.0;
            double s = series(x);
            if (s < 0.0 && s > -1.0e-12) {
                s = 0.0;
            }
            if (rng.uniformOpen01() * rejectionBound_ <= s) {
                return x;
            }
        }
        throw std::runtime_error("Amostragem da funcao de fase nao convergiu; verifique beta_l.");
    }

    double series(double x) const {
        double sum = beta_[0];
        if (beta_.size() == 1) {
            return sum;
        }

        double pLm1 = 1.0;
        double pL = x;
        sum += beta_[1] * pL;

        for (std::size_t ell = 1; ell + 1 < beta_.size(); ++ell) {
            const double pLp1 = ((2.0 * static_cast<double>(ell) + 1.0) * x * pL -
                                 static_cast<double>(ell) * pLm1) /
                                (static_cast<double>(ell) + 1.0);
            sum += beta_[ell + 1] * pLp1;
            pLm1 = pL;
            pL = pLp1;
        }
        return sum;
    }

    const std::vector<double>& beta() const { return beta_; }

private:
    std::vector<double> beta_;
    double rejectionBound_ = 1.0;
};

struct Zone {
    double rInner = 0.0;
    double rOuter = 1.0;
    double sigmaT = 1.0;
    double omega = 0.0;
    double blackbodyIntensity = 0.0;
};

enum class OuterAngularDistribution {
    Diffuse,
    Radial
};

struct Config {
    double a = 1.0;
    double b = 4.0;
    double sigmaT = 1.0;
    double omega = 0.5;
    double blackbodyIntensity = 0.0;

    double rhoInner = 0.0;
    double rhoOuter = 0.0;

    // Intensidades prescritas, constantes no hemisferio, entrando no meio.
    // Podem representar epsilon_i I_bi ou uma iluminacao externa difusa.
    double outerSourceIntensity = 1.0;
    double innerSourceIntensity = 0.0;
    OuterAngularDistribution outerAngular = OuterAngularDistribution::Diffuse;

    std::vector<double> beta = {1.0};
    std::vector<Zone> zones;
    std::string zoneFile;

    std::uint64_t histories = 1000000ULL;
    std::size_t batches = 32;
    std::size_t threads = 0;
    std::uint64_t seed = 20260701ULL;
    std::uint64_t maxEvents = 1000000ULL;

    std::string outputCsv;
};

enum class SourceKind {
    OuterBoundary,
    InnerBoundary,
    VolumeZone
};

struct SourceComponent {
    SourceKind kind = SourceKind::OuterBoundary;
    std::size_t zoneIndex = 0;
    double strength = 0.0;
    double cumulative = 0.0;
};

struct Model {
    std::vector<Zone> zones;
    double a = 0.0;
    double b = 1.0;
    double rhoInner = 0.0;
    double rhoOuter = 0.0;
    double outerSourceIntensity = 0.0;
    double innerSourceIntensity = 0.0;
    OuterAngularDistribution outerAngular = OuterAngularDistribution::Diffuse;
    LegendrePhase phase;
    std::vector<SourceComponent> sources;
    double totalSourceStrength = 0.0;
    double geometryEpsilon = 1.0e-12;

    Model(std::vector<Zone> zonesIn,
          double rhoInnerIn,
          double rhoOuterIn,
          double outerSourceIntensityIn,
          double innerSourceIntensityIn,
          OuterAngularDistribution outerAngularIn,
          LegendrePhase phaseIn)
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

struct BatchTally {
    std::uint64_t histories = 0;

    double outerIn = 0.0;   // -q^-(b), integrado em area (potencia reduzida)
    double outerOut = 0.0;  //  q^+(b), integrado em area
    double innerIn = 0.0;   // -q^-(a), integrado em area
    double innerOut = 0.0;  //  q^+(a), integrado em area

    double absorbedMedium = 0.0;
    double terminatedOuter = 0.0;
    double terminatedInner = 0.0;
    double killedMaxEvents = 0.0;

    std::uint64_t collisions = 0;
    std::uint64_t scatterings = 0;
    std::uint64_t outerHits = 0;
    std::uint64_t innerHits = 0;
    std::uint64_t interfaceCrossings = 0;

    BatchTally& operator+=(const BatchTally& o) {
        histories += o.histories;
        outerIn += o.outerIn;
        outerOut += o.outerOut;
        innerIn += o.innerIn;
        innerOut += o.innerOut;
        absorbedMedium += o.absorbedMedium;
        terminatedOuter += o.terminatedOuter;
        terminatedInner += o.terminatedInner;
        killedMaxEvents += o.killedMaxEvents;
        collisions += o.collisions;
        scatterings += o.scatterings;
        outerHits += o.outerHits;
        innerHits += o.innerHits;
        interfaceCrossings += o.interfaceCrossings;
        return *this;
    }
};

struct Photon {
    Vec3 position;
    Vec3 direction;
    std::size_t zoneIndex = 0;
};

std::string trim(std::string s) {
    const auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

std::vector<double> parseDoubleList(const std::string& text) {
    std::vector<double> values;
    std::string token;
    std::stringstream ss(text);
    while (std::getline(ss, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
    }
    if (values.empty()) {
        throw std::runtime_error("Lista numerica vazia: " + text);
    }
    return values;
}

std::vector<Zone> readZonesCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Nao foi possivel abrir o arquivo de zonas: " + path);
    }

    std::vector<Zone> zones;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(in, line)) {
        ++lineNumber;
        line = trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }

        for (char& c : line) {
            if (c == ';' || c == '\t') {
                c = ',';
            }
        }

        std::vector<double> v;
        try {
            std::string token;
            std::stringstream ss(line);
            while (std::getline(ss, token, ',')) {
                token = trim(token);
                if (token.empty()) {
                    throw std::runtime_error("coluna vazia");
                }
                std::size_t parsed = 0;
                const double value = std::stod(token, &parsed);
                if (!trim(token.substr(parsed)).empty()) {
                    throw std::runtime_error("conteudo nao numerico");
                }
                v.push_back(value);
            }
        } catch (const std::exception&) {
            // Permite uma linha de cabecalho antes dos dados.
            if (zones.empty()) {
                continue;
            }
            std::ostringstream oss;
            oss << "Linha " << lineNumber << " invalida no arquivo de zonas.";
            throw std::runtime_error(oss.str());
        }

        if (v.size() != 5) {
            std::ostringstream oss;
            oss << "Linha " << lineNumber << " do arquivo de zonas deve ter 5 colunas: "
                << "r_inner,r_outer,sigma_t,omega,Ib.";
            throw std::runtime_error(oss.str());
        }
        zones.push_back({v[0], v[1], v[2], v[3], v[4]});
    }

    if (zones.empty()) {
        throw std::runtime_error("O arquivo de zonas nao contem nenhuma zona valida.");
    }
    return zones;
}

void validateZones(const std::vector<Zone>& zones) {
    if (zones.empty()) {
        throw std::runtime_error("O modelo precisa de pelo menos uma zona radial.");
    }
    constexpr double tol = 1.0e-11;
    for (std::size_t i = 0; i < zones.size(); ++i) {
        const Zone& z = zones[i];
        if (z.rInner < 0.0 || !(z.rOuter > z.rInner)) {
            throw std::runtime_error("Raios invalidos em uma zona radial.");
        }
        if (z.sigmaT < 0.0) {
            throw std::runtime_error("sigma_t deve ser nao-negativo.");
        }
        if (z.omega < 0.0 || z.omega > 1.0) {
            throw std::runtime_error("omega deve pertencer a [0,1].");
        }
        if (z.blackbodyIntensity < 0.0) {
            throw std::runtime_error("Ib deve ser nao-negativo.");
        }
        if (i > 0) {
            const double scale = std::max({1.0, std::abs(zones[i - 1].rOuter), std::abs(z.rInner)});
            if (std::abs(zones[i - 1].rOuter - z.rInner) > tol * scale) {
                throw std::runtime_error("As zonas radiais devem ser contiguas e ordenadas.");
            }
        }
    }
    if (!(zones.back().rOuter > 0.0)) {
        throw std::runtime_error("O raio externo b deve ser positivo.");
    }
}

void buildSources(Model& model) {
    model.sources.clear();
    model.totalSourceStrength = 0.0;

    auto add = [&](SourceKind kind, std::size_t zoneIndex, double strength) {
        if (strength <= 0.0) {
            return;
        }
        model.totalSourceStrength += strength;
        model.sources.push_back({kind, zoneIndex, strength, model.totalSourceStrength});
    };

    // Para intensidade hemisferica constante I, a potencia fisica e 4*pi^2*r^2*I.
    // Retiramos o fator comum 4*pi^2; resta r^2*I.
    add(SourceKind::OuterBoundary, model.zones.size() - 1,
        model.b * model.b * model.outerSourceIntensity);

    if (model.a > 0.0) {
        add(SourceKind::InnerBoundary, 0,
            model.a * model.a * model.innerSourceIntensity);
    }

    // Emissao volumetrica isotropica: 4*pi*kappa*Ib por volume. Depois de retirar
    // 4*pi^2, a intensidade total da zona e 4*int r^2*kappa*Ib dr.
    for (std::size_t i = 0; i < model.zones.size(); ++i) {
        const Zone& z = model.zones[i];
        const double sigmaA = z.sigmaT * (1.0 - z.omega);
        const double radialIntegral = (std::pow(z.rOuter, 3) - std::pow(z.rInner, 3)) / 3.0;
        const double strength = 4.0 * sigmaA * z.blackbodyIntensity * radialIntegral;
        add(SourceKind::VolumeZone, i, strength);
    }

    if (!(model.totalSourceStrength > 0.0)) {
        throw std::runtime_error(
            "A potencia total das fontes e zero. Defina --outer-I, --inner-I ou Ib>0 em alguma zona.");
    }

    const double minWidth = std::accumulate(
        model.zones.begin(), model.zones.end(), kInf,
        [](double value, const Zone& z) { return std::min(value, z.rOuter - z.rInner); });
    const double roundoffScale = 128.0 * std::numeric_limits<double>::epsilon() *
                                 std::max(1.0, model.b);
    model.geometryEpsilon = std::max(roundoffScale, 1.0e-12 * minWidth);
    model.geometryEpsilon = std::min(model.geometryEpsilon, 1.0e-8 * minWidth);
}

double distanceToSphere(const Vec3& p, const Vec3& d, double radius, double epsilon) {
    if (!(radius > 0.0)) {
        return kInf;
    }

    const double pd = dot(p, d);
    const double c = norm2(p) - radius * radius;
    double discriminant = pd * pd - c;
    const double scale = std::max({1.0, pd * pd, std::abs(c)});
    if (discriminant < 0.0 && discriminant > -128.0 * std::numeric_limits<double>::epsilon() * scale) {
        discriminant = 0.0;
    }
    if (discriminant < 0.0) {
        return kInf;
    }

    const double root = std::sqrt(discriminant);
    const double s1 = -pd - root;
    const double s2 = -pd + root;
    double answer = kInf;
    const double threshold = 8.0 * epsilon;
    if (s1 > threshold) {
        answer = s1;
    }
    if (s2 > threshold) {
        answer = std::min(answer, s2);
    }
    return answer;
}

const SourceComponent& selectSource(const Model& model, Rng& rng) {
    const double target = rng.uniformOpen01() * model.totalSourceStrength;
    const auto it = std::lower_bound(
        model.sources.begin(), model.sources.end(), target,
        [](const SourceComponent& component, double value) {
            return component.cumulative < value;
        });
    return (it == model.sources.end()) ? model.sources.back() : *it;
}

Photon launchPhoton(const Model& model,
                    const SourceComponent& source,
                    Rng& rng,
                    double weight,
                    BatchTally& tally) {
    Photon photon;

    if (source.kind == SourceKind::OuterBoundary) {
        const Vec3 n = randomUnitVector(rng); // normal radial para fora
        photon.position = model.b * n;
        photon.direction = (model.outerAngular == OuterAngularDistribution::Diffuse)
                               ? sampleCosineHemisphere(-1.0 * n, rng)
                               : -1.0 * n;
        photon.zoneIndex = model.zones.size() - 1;
        tally.outerIn += weight;
        photon.position += model.geometryEpsilon * photon.direction;
        return photon;
    }

    if (source.kind == SourceKind::InnerBoundary) {
        const Vec3 n = randomUnitVector(rng); // radial para fora, que aponta para o meio
        photon.position = model.a * n;
        photon.direction = sampleCosineHemisphere(n, rng);
        photon.zoneIndex = 0;
        tally.innerOut += weight;
        photon.position += model.geometryEpsilon * photon.direction;
        return photon;
    }

    const Zone& z = model.zones[source.zoneIndex];
    const double r3 = std::pow(z.rInner, 3) +
                      rng.uniformOpen01() * (std::pow(z.rOuter, 3) - std::pow(z.rInner, 3));
    const double r = std::cbrt(r3);
    photon.position = r * randomUnitVector(rng);
    photon.direction = randomUnitVector(rng);
    photon.zoneIndex = source.zoneIndex;
    return photon;
}

void simulateHistory(const Model& model,
                     const Config& config,
                     Rng& rng,
                     double weight,
                     BatchTally& tally) {
    const SourceComponent& source = selectSource(model, rng);
    Photon photon = launchPhoton(model, source, rng, weight, tally);
    double remainingOpticalDepth = rng.opticalDepth();

    for (std::uint64_t event = 0; event < config.maxEvents; ++event) {
        const Zone& zone = model.zones[photon.zoneIndex];

        const double sLower = (zone.rInner > 0.0)
                                  ? distanceToSphere(photon.position, photon.direction,
                                                     zone.rInner, model.geometryEpsilon)
                                  : kInf;
        const double sUpper = distanceToSphere(photon.position, photon.direction,
                                               zone.rOuter, model.geometryEpsilon);

        const bool hitLower = sLower < sUpper;
        const double distanceToInterface = hitLower ? sLower : sUpper;
        if (!std::isfinite(distanceToInterface)) {
            tally.killedMaxEvents += weight;
            return;
        }

        const double opticalDepthToInterface = zone.sigmaT * distanceToInterface;
        const double comparisonTol = 64.0 * std::numeric_limits<double>::epsilon() *
                                     std::max({1.0, remainingOpticalDepth,
                                               opticalDepthToInterface});

        if (zone.sigmaT > 0.0 &&
            remainingOpticalDepth < opticalDepthToInterface - comparisonTol) {
            const double collisionDistance = remainingOpticalDepth / zone.sigmaT;
            photon.position += collisionDistance * photon.direction;
            ++tally.collisions;

            if (rng.uniformOpen01() < zone.omega) {
                ++tally.scatterings;
                const double cosineScatter = model.phase.sampleCosine(rng);
                photon.direction = rotateDirection(photon.direction, cosineScatter, rng);
                remainingOpticalDepth = rng.opticalDepth();
                continue;
            }

            tally.absorbedMedium += weight;
            return;
        }

        photon.position += distanceToInterface * photon.direction;
        remainingOpticalDepth = std::max(0.0, remainingOpticalDepth - opticalDepthToInterface);

        if (hitLower) {
            if (photon.zoneIndex > 0) {
                --photon.zoneIndex;
                ++tally.interfaceCrossings;
                photon.position += model.geometryEpsilon * photon.direction;
                continue;
            }

            // Fronteira interna fisica, somente para esfera oca (a>0).
            if (model.a > 0.0) {
                ++tally.innerHits;
                tally.innerIn += weight;

                if (rng.uniformOpen01() < model.rhoInner) {
                    const Vec3 radialOut = normalized(photon.position);
                    photon.direction = sampleCosineHemisphere(radialOut, rng);
                    tally.innerOut += weight;
                    photon.position += model.geometryEpsilon * photon.direction;
                    if (remainingOpticalDepth <= comparisonTol) {
                        remainingOpticalDepth = rng.opticalDepth();
                    }
                    continue;
                }

                tally.terminatedInner += weight;
                return;
            }
        } else {
            if (photon.zoneIndex + 1 < model.zones.size()) {
                ++photon.zoneIndex;
                ++tally.interfaceCrossings;
                photon.position += model.geometryEpsilon * photon.direction;
                continue;
            }

            // Fronteira externa fisica.
            ++tally.outerHits;
            tally.outerOut += weight;

            if (rng.uniformOpen01() < model.rhoOuter) {
                const Vec3 radialOut = normalized(photon.position);
                photon.direction = sampleCosineHemisphere(-1.0 * radialOut, rng);
                tally.outerIn += weight;
                photon.position += model.geometryEpsilon * photon.direction;
                if (remainingOpticalDepth <= comparisonTol) {
                    remainingOpticalDepth = rng.opticalDepth();
                }
                continue;
            }

            tally.terminatedOuter += weight;
            return;
        }
    }

    tally.killedMaxEvents += weight;
}

std::uint64_t seedForBatch(std::uint64_t baseSeed, std::size_t batchIndex) {
    SplitMix64 sm(baseSeed + 0x9E3779B97F4A7C15ULL * (static_cast<std::uint64_t>(batchIndex) + 1ULL));
    return sm.next();
}

BatchTally runBatch(const Model& model,
                    const Config& config,
                    std::size_t batchIndex,
                    std::uint64_t historiesInBatch,
                    double packetWeight) {
    BatchTally tally;
    tally.histories = historiesInBatch;
    Rng rng(seedForBatch(config.seed, batchIndex));
    for (std::uint64_t i = 0; i < historiesInBatch; ++i) {
        simulateHistory(model, config, rng, packetWeight, tally);
    }
    return tally;
}

struct Estimate {
    double value = std::numeric_limits<double>::quiet_NaN();
    double standardError = std::numeric_limits<double>::quiet_NaN();
    std::size_t validBatches = 0;
};

template <class Metric>
Estimate estimateFromBatches(const std::vector<BatchTally>& batches,
                             double globalValue,
                             Metric metric) {
    std::vector<double> values;
    values.reserve(batches.size());
    for (const BatchTally& batch : batches) {
        const double value = metric(batch);
        if (std::isfinite(value)) {
            values.push_back(value);
        }
    }

    Estimate result;
    result.value = globalValue;
    result.validBatches = values.size();
    if (values.size() < 2) {
        return result;
    }

    const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                        static_cast<double>(values.size());
    double sumSquares = 0.0;
    for (double value : values) {
        const double delta = value - mean;
        sumSquares += delta * delta;
    }
    const double sampleVariance = sumSquares / static_cast<double>(values.size() - 1);
    result.standardError = std::sqrt(sampleVariance / static_cast<double>(values.size()));
    return result;
}

std::string betaToString(const std::vector<double>& beta) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < beta.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << std::setprecision(17) << beta[i];
    }
    return oss.str();
}

void printHelp(const char* executable) {
    std::cout
        << "Monte Carlo fisico para transporte radiativo em esfera/casca esferica\n\n"
        << "Uso:\n  " << executable << " [opcoes]\n\n"
        << "Geometria e meio homogeneo (ignorados quando --zones e usado):\n"
        << "  --a VAL                 raio interno; a=0 representa esfera solida [1]\n"
        << "  --b VAL                 raio externo [4]\n"
        << "  --sigma-t VAL           coeficiente de extincao [1]\n"
        << "  --omega VAL             albedo de espalhamento simples [0.5]\n"
        << "  --Ib VAL                intensidade de corpo negro no volume [0]\n"
        << "  --zones ARQUIVO.csv     zonas: r_inner,r_outer,sigma_t,omega,Ib\n\n"
        << "Espalhamento e fronteiras:\n"
        << "  --beta LISTA            beta_l separados por virgula; beta_0=1 [1]\n"
        << "                          isotropico: 1; linear: 1,a_bar\n"
        << "  --rho-inner VAL         refletividade difusa em r=a [0]\n"
        << "  --rho-outer VAL         refletividade difusa em r=b [0]\n"
        << "  --outer-I VAL           intensidade difusa prescrita entrando em r=b [1]\n"
        << "  --inner-I VAL           intensidade difusa prescrita entrando em r=a [0]\n"
        << "  --outer-angular MODE    diffuse ou radial [diffuse]\n\n"
        << "Monte Carlo:\n"
        << "  --histories N           numero total de historias [1000000]\n"
        << "  --batches N             lotes independentes para erro-padrao [32]\n"
        << "  --threads N             threads; 0 usa hardware disponivel [0]\n"
        << "  --seed N                semente inteira [20260701]\n"
        << "  --max-events N          limite de eventos por historia [1000000]\n"
        << "  --output ARQUIVO.csv    grava um resumo em CSV\n"
        << "  --help                   mostra esta ajuda\n\n"
        << "Para reproduzir a equacao normalizada dos slides, use sigma_t=1.\n";
}

Config parseArguments(int argc, char** argv) {
    Config config;

    auto requireValue = [&](int& i, const std::string& option) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error("Falta valor para " + option);
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else if (arg == "--a") {
            config.a = std::stod(requireValue(i, arg));
        } else if (arg == "--b") {
            config.b = std::stod(requireValue(i, arg));
        } else if (arg == "--sigma-t") {
            config.sigmaT = std::stod(requireValue(i, arg));
        } else if (arg == "--omega") {
            config.omega = std::stod(requireValue(i, arg));
        } else if (arg == "--Ib") {
            config.blackbodyIntensity = std::stod(requireValue(i, arg));
        } else if (arg == "--zones") {
            config.zoneFile = requireValue(i, arg);
        } else if (arg == "--beta") {
            config.beta = parseDoubleList(requireValue(i, arg));
        } else if (arg == "--rho-inner") {
            config.rhoInner = std::stod(requireValue(i, arg));
        } else if (arg == "--rho-outer") {
            config.rhoOuter = std::stod(requireValue(i, arg));
        } else if (arg == "--outer-I") {
            config.outerSourceIntensity = std::stod(requireValue(i, arg));
        } else if (arg == "--inner-I") {
            config.innerSourceIntensity = std::stod(requireValue(i, arg));
        } else if (arg == "--outer-angular") {
            const std::string mode = requireValue(i, arg);
            if (mode == "diffuse") {
                config.outerAngular = OuterAngularDistribution::Diffuse;
            } else if (mode == "radial") {
                config.outerAngular = OuterAngularDistribution::Radial;
            } else {
                throw std::runtime_error("--outer-angular aceita diffuse ou radial.");
            }
        } else if (arg == "--histories") {
            config.histories = std::stoull(requireValue(i, arg));
        } else if (arg == "--batches") {
            config.batches = static_cast<std::size_t>(std::stoull(requireValue(i, arg)));
        } else if (arg == "--threads") {
            config.threads = static_cast<std::size_t>(std::stoull(requireValue(i, arg)));
        } else if (arg == "--seed") {
            config.seed = std::stoull(requireValue(i, arg));
        } else if (arg == "--max-events") {
            config.maxEvents = std::stoull(requireValue(i, arg));
        } else if (arg == "--output") {
            config.outputCsv = requireValue(i, arg);
        } else {
            throw std::runtime_error("Opcao desconhecida: " + arg);
        }
    }

    return config;
}

Model makeModel(Config& config) {
    if (!config.zoneFile.empty()) {
        config.zones = readZonesCsv(config.zoneFile);
        config.a = config.zones.front().rInner;
        config.b = config.zones.back().rOuter;
    } else {
        config.zones = {{config.a, config.b, config.sigmaT,
                         config.omega, config.blackbodyIntensity}};
    }

    validateZones(config.zones);

    if (config.rhoInner < 0.0 || config.rhoInner > 1.0 ||
        config.rhoOuter < 0.0 || config.rhoOuter > 1.0) {
        throw std::runtime_error("As refletividades rho devem pertencer a [0,1].");
    }
    if (config.outerSourceIntensity < 0.0 || config.innerSourceIntensity < 0.0) {
        throw std::runtime_error("As intensidades de fonte de fronteira devem ser nao-negativas.");
    }
    if (config.histories == 0) {
        throw std::runtime_error("--histories deve ser positivo.");
    }
    if (config.batches == 0 || config.batches > config.histories) {
        throw std::runtime_error("--batches deve estar entre 1 e o numero de historias.");
    }
    if (config.maxEvents == 0) {
        throw std::runtime_error("--max-events deve ser positivo.");
    }

    LegendrePhase phase(config.beta);
    Model model(config.zones, config.rhoInner, config.rhoOuter,
                config.outerSourceIntensity, config.innerSourceIntensity,
                config.outerAngular, std::move(phase));
    buildSources(model);
    return model;
}

void writeCsv(const std::string& path,
              const Config& config,
              const Model& model,
              const BatchTally& total,
              const Estimate& reflectivity,
              const Estimate& transmissivity,
              double qMinusB,
              double qPlusB,
              double qMinusA,
              double qPlusA) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Nao foi possivel criar o arquivo CSV: " + path);
    }

    out << "histories,batches,threads,seed,a,b,zones,beta,rho_inner,rho_outer,"
           "outer_I,inner_I,total_source,qminus_b,qplus_b,qminus_a,qplus_a,"
           "reflectivity,reflectivity_se,transmissivity,transmissivity_se,"
           "outer_in,outer_out,inner_in,inner_out,absorbed_medium,terminated_outer,"
           "terminated_inner,killed_max_events,collisions,scatterings,outer_hits,inner_hits\n";

    out << config.histories << ','
        << config.batches << ','
        << config.threads << ','
        << config.seed << ','
        << std::setprecision(17) << model.a << ','
        << model.b << ','
        << model.zones.size() << ','
        << '"' << betaToString(model.phase.beta()) << '"' << ','
        << model.rhoInner << ','
        << model.rhoOuter << ','
        << model.outerSourceIntensity << ','
        << model.innerSourceIntensity << ','
        << model.totalSourceStrength << ','
        << qMinusB << ','
        << qPlusB << ','
        << qMinusA << ','
        << qPlusA << ','
        << reflectivity.value << ','
        << reflectivity.standardError << ','
        << transmissivity.value << ','
        << transmissivity.standardError << ','
        << total.outerIn << ','
        << total.outerOut << ','
        << total.innerIn << ','
        << total.innerOut << ','
        << total.absorbedMedium << ','
        << total.terminatedOuter << ','
        << total.terminatedInner << ','
        << total.killedMaxEvents << ','
        << total.collisions << ','
        << total.scatterings << ','
        << total.outerHits << ','
        << total.innerHits << '\n';
}

void printEstimate(const std::string& name, const Estimate& estimate) {
    std::cout << "  " << std::left << std::setw(28) << name << std::right
              << std::scientific << std::setprecision(8) << estimate.value;
    if (std::isfinite(estimate.standardError)) {
        std::cout << "  SE=" << estimate.standardError
                  << "  IC95%=[" << estimate.value - 1.96 * estimate.standardError
                  << ", " << estimate.value + 1.96 * estimate.standardError << ']';
    } else {
        std::cout << "  SE=indisponivel";
    }
    std::cout << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        Config config = parseArguments(argc, argv);
        Model model = makeModel(config);

        if (config.threads == 0) {
            config.threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        }
        config.threads = std::min(config.threads, config.batches);

        bool noVolumeAbsorption = true;
        for (const Zone& z : model.zones) {
            if (z.sigmaT * (1.0 - z.omega) > 0.0) {
                noVolumeAbsorption = false;
                break;
            }
        }
        if (noVolumeAbsorption && model.rhoOuter >= 1.0 &&
            (model.a == 0.0 || model.rhoInner >= 1.0)) {
            std::cerr << "AVISO: meio sem absorcao e fronteiras perfeitamente refletoras; "
                         "historias podem nao terminar.\n";
        }
        if (model.a == 0.0 && (model.rhoInner > 0.0 || model.innerSourceIntensity > 0.0)) {
            std::cerr << "AVISO: para a=0 nao existe fronteira interna; rho_inner e inner_I sao ignorados.\n";
        }
        if (model.outerAngular == OuterAngularDistribution::Radial) {
            std::cerr << "AVISO: fonte radial e um caso de feixe colimado; a condicao de contorno "
                         "dos slides corresponde a fonte difusa.\n";
        }

        const double packetWeight = model.totalSourceStrength /
                                    static_cast<double>(config.histories);

        std::vector<std::uint64_t> historiesPerBatch(config.batches,
                                                     config.histories / config.batches);
        for (std::size_t i = 0; i < config.histories % config.batches; ++i) {
            ++historiesPerBatch[i];
        }

        std::vector<BatchTally> batches(config.batches);
        std::atomic<std::size_t> nextBatch{0};
        std::vector<std::thread> workers;
        workers.reserve(config.threads);

        for (std::size_t t = 0; t < config.threads; ++t) {
            workers.emplace_back([&]() {
                while (true) {
                    const std::size_t index = nextBatch.fetch_add(1);
                    if (index >= config.batches) {
                        break;
                    }
                    batches[index] = runBatch(model, config, index,
                                              historiesPerBatch[index], packetWeight);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }

        BatchTally total;
        for (const BatchTally& batch : batches) {
            total += batch;
        }

        const double reflectivityValue = (total.outerIn > 0.0)
                                             ? total.outerOut / total.outerIn
                                             : std::numeric_limits<double>::quiet_NaN();
        const double transmissivityValue = (model.a > 0.0 && total.outerIn > 0.0)
                                               ? total.innerIn / total.outerIn
                                               : std::numeric_limits<double>::quiet_NaN();

        const Estimate reflectivity = estimateFromBatches(
            batches, reflectivityValue,
            [](const BatchTally& b) {
                return (b.outerIn > 0.0)
                           ? b.outerOut / b.outerIn
                           : std::numeric_limits<double>::quiet_NaN();
            });
        const Estimate transmissivity = estimateFromBatches(
            batches, transmissivityValue,
            [a = model.a](const BatchTally& b) {
                return (a > 0.0 && b.outerIn > 0.0)
                           ? b.innerIn / b.outerIn
                           : std::numeric_limits<double>::quiet_NaN();
            });

        // P = 8*pi^2*r^2*|q|; os pesos foram definidos com o fator 4*pi^2 removido.
        // Logo q = +/- P_reduzida/(2*r^2).
        const double qMinusB = -total.outerIn / (2.0 * model.b * model.b);
        const double qPlusB = total.outerOut / (2.0 * model.b * model.b);
        const double qMinusA = (model.a > 0.0)
                                   ? -total.innerIn / (2.0 * model.a * model.a)
                                   : std::numeric_limits<double>::quiet_NaN();
        const double qPlusA = (model.a > 0.0)
                                  ? total.innerOut / (2.0 * model.a * model.a)
                                  : std::numeric_limits<double>::quiet_NaN();

        const double terminalEnergy = total.absorbedMedium + total.terminatedOuter +
                                      total.terminatedInner + total.killedMaxEvents;

        std::cout << std::setprecision(8);
        std::cout << "\n=== Monte Carlo de transporte radiativo esferico ===\n";
        std::cout << "Geometria: " << ((model.a > 0.0) ? "casca oca" : "esfera solida")
                  << ", a=" << model.a << ", b=" << model.b
                  << ", zonas=" << model.zones.size() << '\n';
        std::cout << "Historias=" << config.histories
                  << ", lotes=" << config.batches
                  << ", threads=" << config.threads
                  << ", seed=" << config.seed << '\n';
        std::cout << "beta={" << betaToString(model.phase.beta()) << "}\n";
        std::cout << "Potencia total reduzida das fontes=" << std::scientific
                  << model.totalSourceStrength << "\n\n";

        std::cout << "Fluxos reduzidos (na convencao q dos slides):\n";
        std::cout << "  q^-(b) = " << qMinusB << '\n';
        std::cout << "  q^+(b) = " << qPlusB << '\n';
        if (model.a > 0.0) {
            std::cout << "  q^-(a) = " << qMinusA << '\n';
            std::cout << "  q^+(a) = " << qPlusA << '\n';
        }

        std::cout << "\nRazoes de fluxo (fluxos brutos, incluindo reflexoes repetidas):\n";
        printEstimate("R = -q^+(b)/q^-(b)", reflectivity);
        if (model.a > 0.0) {
            printEstimate("T = a^2 q^-(a)/(b^2 q^-(b))", transmissivity);
        } else {
            std::cout << "  T nao e definido para a=0 pela formula apresentada.\n";
        }

        std::cout << "\nPotencias reduzidas acumuladas:\n";
        std::cout << "  entrada total em b         = " << total.outerIn << '\n';
        std::cout << "  saida total em b           = " << total.outerOut << '\n';
        if (model.a > 0.0) {
            std::cout << "  chegada total em a         = " << total.innerIn << '\n';
            std::cout << "  saida total de a           = " << total.innerOut << '\n';
        }
        std::cout << "  absorvida no meio          = " << total.absorbedMedium << '\n';
        std::cout << "  terminal em b              = " << total.terminatedOuter << '\n';
        std::cout << "  terminal em a              = " << total.terminatedInner << '\n';
        std::cout << "  truncada por max-events    = " << total.killedMaxEvents << '\n';
        std::cout << "  balanco terminal/fonte     = "
                  << terminalEnergy / model.totalSourceStrength << '\n';

        std::cout << "\nDiagnostico por historia:\n";
        std::cout << "  colisoes medias      = "
                  << static_cast<double>(total.collisions) / static_cast<double>(total.histories)
                  << '\n';
        std::cout << "  espalhamentos medios = "
                  << static_cast<double>(total.scatterings) / static_cast<double>(total.histories)
                  << '\n';
        std::cout << "  hits externos medios = "
                  << static_cast<double>(total.outerHits) / static_cast<double>(total.histories)
                  << '\n';
        if (model.a > 0.0) {
            std::cout << "  hits internos medios = "
                      << static_cast<double>(total.innerHits) / static_cast<double>(total.histories)
                      << '\n';
        }

        if (!config.outputCsv.empty()) {
            writeCsv(config.outputCsv, config, model, total, reflectivity, transmissivity,
                     qMinusB, qPlusB, qMinusA, qPlusA);
            std::cout << "\nResumo gravado em: " << config.outputCsv << '\n';
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "ERRO: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
