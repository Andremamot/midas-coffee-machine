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
// YUY2 PACKED TEXTURE
// RGBA = Y0 U Y1 V
// ===============================
uniform sampler2D uYUY2Tex;

uniform vec2  uFlip;
uniform vec2  uMarginScale;
uniform vec2  uMarginOffset;

out vec4 fragColor;

#define PI 3.14159265358979323846

// ======================================================
// Lens curve
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
// YUY2 SAMPLE (no bicubic, packed decode)
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

    float isEven = step(0.5, 1.0 - mod(floor(pixel.x), 2.0));

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
// Aberration (fixed cost 3 fetches)
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
// Edge fade
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

    if (uFlip.x > 0.5)
        fragCoord.x = uResolution.x - fragCoord.x;

    if (uFlip.y > 0.5)
        fragCoord.y = uResolution.y - fragCoord.y;

    fragCoord = fragCoord * uMarginScale + uMarginOffset;

    const float PCT_W = 1.27;
    const float PCT_H = 1.27;
    const float FOCAL_ZOOM = 410.0;

    float dcx = uResolution.x * 0.5;
    float dcy = uResolution.y * 0.5;

    float mAlphaOffset = uControl.x * (PI / 180.0);
    float mBetaOffset  = (uControl.y + 180.0) * (PI / 180.0);
    float zoom_val     = uControl.z;

    float widthCosB      = PCT_W * cos(mBetaOffset);
    float widthSinB      = PCT_W * sin(mBetaOffset);

    float heightCosASinB = PCT_H * cos(mAlphaOffset) * sin(mBetaOffset);
    float heightCosACosB = PCT_H * cos(mAlphaOffset) * cos(mBetaOffset);
    float heightSinA     = PCT_H * sin(mAlphaOffset);

    float flZoomSinASinB = FOCAL_ZOOM * zoom_val * sin(mAlphaOffset) * sin(mBetaOffset);
    float flZoomSinACosB = FOCAL_ZOOM * zoom_val * sin(mAlphaOffset) * cos(mBetaOffset);
    float flZoomCosA     = FOCAL_ZOOM * zoom_val * cos(mAlphaOffset);

    float tempX =
        (fragCoord.x - dcx) * widthCosB -
        (fragCoord.y - dcy) * heightCosASinB +
        flZoomSinASinB;

    float tempY =
        (fragCoord.x - dcx) * widthSinB +
        (fragCoord.y - dcy) * heightCosACosB -
        flZoomSinACosB;

    float tempZ =
        (fragCoord.y - dcy) * heightSinA +
        flZoomCosA;

    float alpha = atan(sqrt(tempX * tempX + tempY * tempY), tempZ);

    float beta;
    if (tempX != 0.0)
        beta = atan(tempY, tempX);
    else
        beta = (tempY >= 0.0) ? (PI / 2.0) : (-PI / 2.0);

    float rho = calculateRho(alpha);

    float sourceX = uMoilCenter.x - rho * cos(beta);
    float sourceY = uMoilCenter.y - rho * sin(beta);

    vec2 texCoord = vec2(sourceX, sourceY) / uMoilSize;

    if (texCoord.x < 0.0 || texCoord.x > 1.0 ||
        texCoord.y < 0.0 || texCoord.y > 1.0)
    {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 col = sampleWithAberration(texCoord).rgb;

    col *= boundaryAlpha(texCoord);

    vec2 uvScreen = fragCoord / uResolution;

    col = mix(col, col * vignette(uvScreen), 0.35);

    col = (col - 0.5) * 1.05 + 0.5;

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));

    col = mix(vec3(lum), col, 1.08);

    fragColor = vec4(col, 1.0);
}