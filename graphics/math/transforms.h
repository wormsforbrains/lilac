/**
 * ============================================================================
 * GRAPHICS-SPECIFIC MATHEMATICS
 * Transforms, projections, NDC conversions, barycentric coordinates.
 * Depends on: math/core.h
 * ============================================================================
 */

#pragma once

#include "../math/core.h"
#include <cmath>

namespace graphics {

// ============================================================================
// TRANSFORMATION MATRICES
// ============================================================================

/// @brief Creates a translation matrix.
[[nodiscard]] inline math::Mat4x4 mat_translate(math::Vec3 offset) {
  auto m = math::mat_identity();
  math::mat_set(m, 0, 3, offset.x);
  math::mat_set(m, 1, 3, offset.y);
  math::mat_set(m, 2, 3, offset.z);
  return m;
}

/// @brief Creates a scale matrix.
[[nodiscard]] inline math::Mat4x4 mat_scale(math::Vec3 scale) {
  auto m = math::mat_identity();
  math::mat_set(m, 0, 0, scale.x);
  math::mat_set(m, 1, 1, scale.y);
  math::mat_set(m, 2, 2, scale.z);
  return m;
}

/// @brief Creates a rotation matrix around the X axis (right vector).
[[nodiscard]] inline math::Mat4x4 mat_rotate_x(float radians) {
  auto m = math::mat_identity();
  float c = std::cos(radians);
  float s = std::sin(radians);
  math::mat_set(m, 1, 1, c);
  math::mat_set(m, 1, 2, -s);
  math::mat_set(m, 2, 1, s);
  math::mat_set(m, 2, 2, c);
  return m;
}

/// @brief Creates a rotation matrix around the Y axis (up vector).
[[nodiscard]] inline math::Mat4x4 mat_rotate_y(float radians) {
  auto m = math::mat_identity();
  float c = std::cos(radians);
  float s = std::sin(radians);
  math::mat_set(m, 0, 0, c);
  math::mat_set(m, 0, 2, s);
  math::mat_set(m, 2, 0, -s);
  math::mat_set(m, 2, 2, c);
  return m;
}

/// @brief Creates a rotation matrix around the Z axis (forward vector).
[[nodiscard]] inline math::Mat4x4 mat_rotate_z(float radians) {
  auto m = math::mat_identity();
  float c = std::cos(radians);
  float s = std::sin(radians);
  math::mat_set(m, 0, 0, c);
  math::mat_set(m, 0, 1, -s);
  math::mat_set(m, 1, 0, s);
  math::mat_set(m, 1, 1, c);
  return m;
}

/// @brief Creates a perspective projection matrix (OpenGL-style, right-handed).
/// @param fov_y_radians Vertical field of view in radians
/// @param aspect Aspect ratio (width / height)
/// @param near_z Distance to near plane (must be > 0)
/// @param far_z Distance to far plane (must be > near_z)
[[nodiscard]] inline math::Mat4x4 mat_perspective(float fov_y_radians, float aspect, float near_z, float far_z) {
  math::Mat4x4 m = math::mat_zero();
  float f = 1.0f / std::tan(fov_y_radians * 0.5f);
  float denom = near_z - far_z;

  math::mat_set(m, 0, 0, f / aspect);
  math::mat_set(m, 1, 1, f);
  math::mat_set(m, 2, 2, (far_z + near_z) / denom);
  math::mat_set(m, 2, 3, (2.0f * far_z * near_z) / denom);
  math::mat_set(m, 3, 2, -1.0f);
  return m;
}

/// @brief Creates an orthographic projection matrix.
[[nodiscard]] inline math::Mat4x4 mat_ortho(float left, float right, float bottom, float top, float near_z, float far_z) {
  math::Mat4x4 m = math::mat_zero();
  math::mat_set(m, 0, 0, 2.0f / (right - left));
  math::mat_set(m, 1, 1, 2.0f / (top - bottom));
  math::mat_set(m, 2, 2, -2.0f / (far_z - near_z));
  math::mat_set(m, 0, 3, -(right + left) / (right - left));
  math::mat_set(m, 1, 3, -(top + bottom) / (top - bottom));
  math::mat_set(m, 2, 3, -(far_z + near_z) / (far_z - near_z));
  math::mat_set(m, 3, 3, 1.0f);
  return m;
}

/// @brief Creates a look-at matrix (view matrix).
/// @param eye Position of the camera
/// @param target Point the camera is looking at
/// @param up Approximate up direction (will be orthonormalized)
[[nodiscard]] inline math::Mat4x4 mat_look_at(math::Vec3 eye, math::Vec3 target, math::Vec3 up) {
  math::Vec3 forward = math::normalize(eye - target);
  math::Vec3 right = math::normalize(math::cross(up, forward));
  math::Vec3 new_up = math::cross(forward, right);

  auto m = math::mat_identity();

  math::mat_set(m, 0, 0, right.x);
  math::mat_set(m, 1, 0, right.y);
  math::mat_set(m, 2, 0, right.z);

  math::mat_set(m, 0, 1, new_up.x);
  math::mat_set(m, 1, 1, new_up.y);
  math::mat_set(m, 2, 1, new_up.z);

  math::mat_set(m, 0, 2, forward.x);
  math::mat_set(m, 1, 2, forward.y);
  math::mat_set(m, 2, 2, forward.z);

  math::mat_set(m, 0, 3, -math::dot(right, eye));
  math::mat_set(m, 1, 3, -math::dot(new_up, eye));
  math::mat_set(m, 2, 3, -math::dot(forward, eye));

  return m;
}

// ============================================================================
// COORDINATE SPACE CONVERSIONS
// ============================================================================

/// @brief Perspective divide: converts homogeneous clip space to NDC.
[[nodiscard]] inline math::Vec3 perspective_divide(math::Vec4 clip_space) {
  float w_inv = 1.0f / clip_space.w;
  return {clip_space.x * w_inv, clip_space.y * w_inv, clip_space.z * w_inv};
}

/// @brief Converts from NDC ([-1, 1] in X,Y) to screen space [0, width) x [0, height).
[[nodiscard]] inline math::Vec3 ndc_to_screen(math::Vec3 ndc, int viewport_width, int viewport_height) {
  float screen_x = (ndc.x + 1.0f) * 0.5f * viewport_width;
  float screen_y = (1.0f - ndc.y) * 0.5f * viewport_height;
  return {screen_x, screen_y, ndc.z};
}

/// @brief Converts from screen space back to NDC.
[[nodiscard]] inline math::Vec3 screen_to_ndc(math::Vec3 screen, int viewport_width, int viewport_height) {
  float ndc_x = (screen.x / viewport_width) * 2.0f - 1.0f;
  float ndc_y = 1.0f - (screen.y / viewport_height) * 2.0f;
  return {ndc_x, ndc_y, screen.z};
}

// ============================================================================
// BARYCENTRIC COORDINATES (for triangle rasterization)
// ============================================================================

/// @brief Computes barycentric coordinates of point P with respect to triangle ABC.
/// @return (u, v, w) where P = u*A + v*B + w*C and u + v + w = 1
[[nodiscard]] inline math::Vec3 barycentric_coords(math::Vec2 p, math::Vec2 a, math::Vec2 b, math::Vec2 c) {
  math::Vec2 v0 = c - a;
  math::Vec2 v1 = b - a;
  math::Vec2 v2 = p - a;

  float dot00 = math::dot(v0, v0);
  float dot01 = math::dot(v0, v1);
  float dot02 = math::dot(v0, v2);
  float dot11 = math::dot(v1, v1);
  float dot12 = math::dot(v1, v2);

  float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
  float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
  float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
  float w = 1.0f - u - v;

  return {u, v, w};
}

/// @brief Tests if point P is inside triangle ABC using barycentric coordinates.
[[nodiscard]] inline bool is_point_in_triangle(math::Vec2 p, math::Vec2 a, math::Vec2 b, math::Vec2 c) {
  auto bary = barycentric_coords(p, a, b, c);
  return bary.x >= 0.0f && bary.y >= 0.0f && bary.z >= 0.0f;
}

// ============================================================================
// TRIANGLE PROPERTIES
// ============================================================================

/// @brief Computes the signed area of a triangle in screen space (2D).
[[nodiscard]] inline float triangle_signed_area(math::Vec2 a, math::Vec2 b, math::Vec2 c) {
  return ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) * 0.5f;
}

/// @brief Determines if triangle vertices are in counter-clockwise order.
[[nodiscard]] inline bool is_front_facing(math::Vec2 a, math::Vec2 b, math::Vec2 c) {
  return triangle_signed_area(a, b, c) > 0.0f;
}

// ============================================================================
// PERSPECTIVE-CORRECT INTERPOLATION
// ============================================================================

/// @brief Perspective-correct interpolation of a Vec2 attribute across a triangle.
[[nodiscard]] inline math::Vec2 interpolate_perspective_correct(
    math::Vec2 attr0, math::Vec2 attr1, math::Vec2 attr2,
    float w0_inv, float w1_inv, float w2_inv,
    math::Vec3 bary) {
  float denom = bary.x * w0_inv + bary.y * w1_inv + bary.z * w2_inv;
  denom = 1.0f / denom;
  return (attr0 * (bary.x * w0_inv) +
          attr1 * (bary.y * w1_inv) +
          attr2 * (bary.z * w2_inv)) * denom;
}

/// @brief Perspective-correct interpolation of a scalar attribute.
[[nodiscard]] inline float interpolate_perspective_correct(
    float attr0, float attr1, float attr2,
    float w0_inv, float w1_inv, float w2_inv,
    math::Vec3 bary) {
  float denom = bary.x * w0_inv + bary.y * w1_inv + bary.z * w2_inv;
  denom = 1.0f / denom;
  return (attr0 * (bary.x * w0_inv) +
          attr1 * (bary.y * w1_inv) +
          attr2 * (bary.z * w2_inv)) * denom;
}

// ============================================================================
// VIEWPORT HELPERS
// ============================================================================

struct Viewport {
  int x = 0;
  int y = 0;
  int width = 800;
  int height = 600;
};

/// @brief Tests if a pixel is inside the viewport.
[[nodiscard]] inline bool is_in_viewport(int pixel_x, int pixel_y, const Viewport& viewport) {
  return pixel_x >= viewport.x && pixel_x < (viewport.x + viewport.width) &&
         pixel_y >= viewport.y && pixel_y < (viewport.y + viewport.height);
}

} // namespace graphics
