#pragma once

#include "intersections.h"

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 sampleDiffuse(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3& wi) { // write out to wi

    wi = calculateRandomDirectionInHemisphere(nor, rng);
    return m.color;
}

__host__ __device__
glm::vec3 sampleDiffuseTrans(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3& wi,
    float& absDot, float& pdf) {
    // sample wi in the opposite hemisphere of wo, everything else same as diffuse_refl
    wi = calculateRandomDirectionInHemisphere(nor, rng);
    wi.z = -wi.z;
    absDot = glm::abs(glm::dot(nor, wi));
    pdf = absDot * INV_PI;
    return m.color * INV_PI;
}

__host__ __device__
glm::vec3 sampleSpecularRefl(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {

    wi = glm::reflect(wo, nor);
    return m.specular.color;
}

__host__ __device__
glm::vec3 sampleSpecularTrans(const Material& m, glm::vec3 nor,
    glm::vec3 wo, glm::vec3& wi) {

    // Default to hard-coded glass IOR if not provided
    float ior = m.indexOfRefraction;
    ior = ior ? 1.55f : ior;

    // figure out which surface to enter/exit
    // eta of air always assumed to be 1
    bool entering = glm::clamp(glm::dot(wo, nor), -1.f, 1.f) < 0.0f;
    float etaI = entering ? 1.0f : ior; // incident index
    float etaT = entering ? ior : 1.0f; // transmitted index
    float eta = etaI / etaT;

    // flip normal
    nor = entering ? nor : -nor;
    wi = glm::refract(wo, nor, eta);

    // handle total internal refl --> basically specular BRDF
    if (glm::length(wi) < EPSILON) {
        wi = glm::reflect(wo, nor);
        return glm::vec3(0.0f);
    }
    return m.specular.color;
}

__host__ __device__
float fresnelDielectricEval(const Material& m, float cosThetaI) {
    float etaI = 1.f;
    float etaT = m.indexOfRefraction;
    etaT = etaT < EPSILON ? 1.55f : etaT;
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);

    // see pbrt FrDielectric()
    // Potentially swap indices of refraction
    if (cosThetaI > 0.f) {
        float temp = etaI;
        etaI = etaT;
        etaT = temp;
    }
    cosThetaI = glm::abs(cosThetaI);

    // Computer cosThetaT using Snell's law
    float sinThetaI = glm::sqrt(glm::max(0.f,
        1.f - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;

    // Handle total internal reflection
    if (sinThetaT >= 0.999f) {
        return 1.f;
    }

    // Compute Fresnel reflectance using light polarization eqns, see PBRT 8.2.1
    float cosThetaT = glm::sqrt(glm::max(0.f,
        1.f - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
        ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
        ((etaI * cosThetaI) + (etaT * cosThetaT));

    return (Rparl * Rparl + Rperp * Rperp) * 0.5f; // coefficient
}

__host__ __device__
glm::vec3 sampleGlass(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3 wo, glm::vec3& wi,
    float& absDot, float& pdf) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    bool random = u01(rng);

    float fresnel = fresnelDielectricEval(m, glm::dot(wo, nor));
    glm::vec3 bsdf(0.f);
    if (random < fresnel) {
        // Have to double contribution b/c we only sample
        // reflection BxDF half the time
        bsdf = sampleSpecularRefl(m, nor, wo, wi);
        return bsdf;
    }
    else {
        bsdf = sampleSpecularTrans(m, nor, wo, wi);
        return bsdf; // 1-fr b/c all conditions sum up to 1
    }
}

__host__ __device__
glm::vec3 samplePlastic(const Material& m, glm::vec3 nor,
    thrust::default_random_engine& rng, glm::vec3 wo, glm::vec3& wi,
    float& absDot, float& pdf) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    bool random = u01(rng);
    glm::vec3 bsdf(0.f);

    float fresnel = fresnelDielectricEval(m, glm::dot(wo, nor));;
    //if (random < fresnel) {
    //    // Have to double contribution b/c we only sample
    //    // reflection BxDF half the time
    //    bsdf = sampleSpecularRefl(m, nor, wo, wi);
    //    return bsdf;
    //}
    //else {
    //    bsdf = sampleDiffuse(m, nor, rng, wi);
    //    return bsdf; // 1-fr b/c all conditions sum up to 1
    //}

    if (random < 0.5f) {
        // diffuse
        bsdf = sampleDiffuse(m, nor, rng, wi);// glm::normalize(calculateRandomDirectionInHemisphere(nor, rng));
        absDot = glm::abs(glm::dot(nor, wi));
        pdf = absDot * INV_PI;
        bsdf = m.color * INV_PI;
        bsdf *= (1.f - fresnel);
    }
    else {
        // spec refl
        bsdf = sampleSpecularRefl(m, nor, wo, wi);// glm::reflect(wo, nor);
        absDot = glm::abs(glm::dot(nor, wi));
        pdf = 1.0f;
        if (absDot > 0.0f) {
            bsdf = m.specular.color / absDot;
        }
        bsdf *= fresnel;
    }
    return bsdf * 2.0f;
}

/**
 * Simple ray scattering with diffuse and perfect specular support.
 */
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    glm::vec3 newDirection;
    if (m.hasReflective) {
        newDirection = glm::reflect(pathSegment.ray.direction, normal);
    }
    else {
        newDirection = calculateRandomDirectionInHemisphere(normal, rng);
    }

    pathSegment.ray.direction = newDirection;
    pathSegment.ray.origin = intersect + (newDirection * 0.0001f);
}

__host__ __device__
void scatterRayMoreMats(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    if (pathSegment.remainingBounces == 0) return;

    glm::vec3 wo = pathSegment.ray.direction;
    glm::vec3 wi(0.0f);
    glm::vec3 bsdf(0.0f);
    float absDot = 1.f, pdf = 1.f;
    bool isDiffuse = false;
    bool manualCalc = false;

    // Based on material type
    if (m.hasReflective && m.hasRefractive) {
        bsdf = sampleGlass(m, normal, rng, wo, wi, absDot, pdf);
    }
    // Had to manually calculate absDot and pdf to get correct results, otherwise not working properly
    else if (m.hasReflective && glm::length(m.color) > EPSILON) {
        bsdf = samplePlastic(m, normal, rng, wo, wi, absDot, pdf);
        manualCalc = true;
    }
    else if (m.hasReflective) {
        bsdf = sampleSpecularRefl(m, normal, wo, wi);
    }
    else if (m.hasRefractive) {
        bsdf = sampleSpecularTrans(m, normal, wo, wi);
    }
    else { // default to lambert diffuse
        bsdf = sampleDiffuse(m, normal, rng, wi);
        isDiffuse = true;
    }

    pathSegment.ray.direction = glm::normalize(wi);
    pathSegment.ray.origin = intersect + (glm::normalize(wi) * 0.0001f);
    pathSegment.throughput = manualCalc ? (bsdf * absDot / pdf) : bsdf;
}
