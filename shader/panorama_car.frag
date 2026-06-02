#version 300 es
precision highp float;

uniform vec2  uResolution;
uniform vec2  uViewportOffset;
uniform vec4  uControl;
uniform vec2  uMoilSize;
uniform vec2  uMoilCenter;
uniform float uCpRatio;

uniform float p0, p1, p2, p3, p4, p5;

// ===============================
// YUY2 PACKED INPUT
// RGBA = Y0 U Y1 V
// texture size = width/2 x height
// ===============================
uniform sampler2D uYUY2Tex;

uniform vec2  uFlip;
uniform vec2  uMarginScale;
uniform vec2  uMarginOffset;
uniform vec4  uMarginROIBounds;

out vec4 fragColor;

#define PI 3.14159265358979323846

// ======================================================
// Lens polynomial
// ======================================================
float calculateRho(float alpha) {
    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a3 * alpha;
    float a5 = a4 * alpha;
    float a6 = a5 * alpha;

    return (p0 * a6 + p1 * a5 + p2 * a4 + p3 * a3 + p4 * a2 + p5 * alpha) * uCpRatio;
}

// ======================================================
// YUY2 decode (NO bicubic)
// ======================================================
vec3 sampleYUY2(vec2 uv)
{
    vec2 texSize = vec2(textureSize(uYUY2Tex, 0));

    vec2 pixel = uv * uMoilSize;

    float pairX = floor(pixel.x * 0.5);

    vec2 texCoord = vec2(
        (pairX + 0.5) / texSize.x,
        (floor(pixel.y) + 0.5) / texSize.y
    );

    vec4 yuyv = texture(uYUY2Tex, texCoord);

    float isEven = 1.0 - mod(floor(pixel.x), 2.0);

    float Y = mix(yuyv.b, yuyv.r, isEven);

    float U = yuyv.g - 0.5;
    float V = yuyv.a - 0.5;

    vec3 rgb = vec3(
        Y + 1.402 * V,
        Y - 0.344136 * U - 0.714136 * V,
        Y + 1.772 * U
    );

    return clamp(rgb, 0.0, 1.0);
}

// ======================================================
// Chromatic aberration (3 fetches)
// ======================================================
vec4 sampleWithAberration(vec2 uv)
{
    const float strength = 0.0025;
    vec2 dir = uv - 0.5;

    vec2 uvR = clamp(uv + dir * strength, 0.0, 1.0);
    vec2 uvB = clamp(uv - dir * strength, 0.0, 1.0);

    vec3 cR = sampleYUY2(uvR);
    vec3 c  = sampleYUY2(uv);
    vec3 cB = sampleYUY2(uvB);

    return vec4(cR.r, c.g, cB.b, 1.0);
}

// ======================================================
// Boundary fade
// ======================================================
float boundaryAlpha(vec2 texCoord) {
    vec2 edge = min(texCoord, 1.0 - texCoord);
    float fadePixels = 3.0 / min(uMoilSize.x, uMoilSize.y);

    return smoothstep(0.0, fadePixels, min(edge.x, edge.y));
}

// ======================================================
// Vignette
// ======================================================
float vignette(vec2 uvScreen) {
    vec2 d = uvScreen - 0.5;

    return clamp(pow(1.0 - dot(d, d) * 1.8, 3.0), 0.0, 1.0);
}

// ======================================================
// MAIN
// ======================================================
void main()
{
    vec2 localFragCoord = gl_FragCoord.xy - uViewportOffset;

    vec2 fragCoord = vec2(
        localFragCoord.x,
        uResolution.y - localFragCoord.y
    );

    // flip
    if (uFlip.x > 0.5)
        fragCoord.x = uResolution.x - fragCoord.x;

    if (uFlip.y > 0.5)
        fragCoord.y = uResolution.y - fragCoord.y;

    // margin transform
    fragCoord = fragCoord * uMarginScale + uMarginOffset;

    // ROI crop (letterbox)
    if (fragCoord.x < uMarginROIBounds.x ||
        fragCoord.x > uMarginROIBounds.z ||
        fragCoord.y < uMarginROIBounds.y ||
        fragCoord.y > uMarginROIBounds.w)
    {
        fragColor = vec4(0.0);
        return;
    }

    // ==================================================
    // Fisheye parameters (your original logic unchanged)
    // ==================================================
    float alphaMax = uControl.w * (PI / 180.0);
    float iC_alpha = uControl.x * (PI / 180.0);
    float iC_beta  = -uControl.y * (PI / 180.0);

    float kx = sin(iC_alpha) * cos(iC_beta);
    float ky = sin(iC_alpha) * sin(iC_beta);
    float kz = cos(iC_alpha);

    float localY_post = uResolution.y - fragCoord.y;

    float target_alpha =
        iC_alpha + ((localY_post / uResolution.y) * alphaMax);

    float Vx = sin(target_alpha) * cos(iC_beta);
    float Vy = sin(target_alpha) * sin(iC_beta);
    float Vz = cos(target_alpha);

    float kxa_x = ky * Vz - kz * Vy;
    float kxa_y = kz * Vx - kx * Vz;
    float kxa_z = kx * Vy - ky * Vx;

    float k_a = kx * Vx + ky * Vy + kz * Vz;

    float ing_beta = (fragCoord.x / uResolution.x) * 2.0 * PI;

    float cB = cos(ing_beta);
    float sB = sin(ing_beta);

    float v_rot_x = cB * Vx + kxa_x * sB + kx * k_a * (1.0 - cB);
    float v_rot_y = cB * Vy + kxa_y * sB + ky * k_a * (1.0 - cB);
    float v_rot_z = cB * Vz + kxa_z * sB + kz * k_a * (1.0 - cB);

    float f_alpha = atan(sqrt(v_rot_x * v_rot_x + v_rot_y * v_rot_y), v_rot_z);
    float f_beta  = (PI / 2.0) - atan(v_rot_y, v_rot_x);

    float rho = calculateRho(f_alpha);

    float sourceX = uMoilCenter.x - rho * cos(f_beta);
    float sourceY = uMoilCenter.y - rho * sin(f_beta);

    vec2 texCoord = vec2(sourceX, sourceY) / uMoilSize;

    if (texCoord.x < 0.0 || texCoord.x > 1.0 ||
        texCoord.y < 0.0 || texCoord.y > 1.0)
    {
        fragColor = vec4(0.0);
        return;
    }

    // ==================================================
    // Sample
    // ==================================================
    vec3 col = sampleWithAberration(texCoord).rgb;

    col *= boundaryAlpha(texCoord);

    vec2 uvScreen = fragCoord / uResolution;

    col = mix(col, col * vignette(uvScreen), 0.35);

    col = (col - 0.5) * 1.05 + 0.5;

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));

    col = mix(vec3(lum), col, 1.08);

    fragColor = vec4(col, 1.0);
}