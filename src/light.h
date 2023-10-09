#pragma once

#include "lightStruct.h"
#include "sample.h"

__device__ glm::vec3 L(const Light & light, const glm::vec3 & p, const glm::vec3 & n, const glm::vec2 & uv, 
	const glm::vec3 & w /* Outgoing ray from light */
) {
	return light.scale * light.color;
	//return light.color;
}

__device__ LightLiSample sampleLi(const Light& light, const Triangle & tri, const ShadeableIntersection& intersection, const glm::vec2 & u) {
	auto ss = sampleTriangle(tri, u);
	if (ss.pdf == 0 || glm::length2(ss.position - intersection.intersectionPoint) < EPSILON) {
		return {};
	}

	glm::vec3 wi = glm::normalize(ss.position - intersection.intersectionPoint);
	if (glm::dot(wi, intersection.surfaceNormal) < 0  
		|| glm::dot(-wi, ss.normal) < 0
		) 
	{
			return {};
	}
	float pdf = ss.pdf * glm::length2(ss.position - intersection.intersectionPoint) / glm::dot(ss.normal, -wi);
	//float pdf = ss.pdf / glm::dot(ss.intersection.surfaceNormal, -wi);
	//auto Le = L(light, ss.intersection.intersectionPoint, ss.intersection.surfaceNormal, ss.intersection.uv, -wi);
	auto Le = light.scale * light.color;
	return {Le, wi, pdf, ss.position};
}