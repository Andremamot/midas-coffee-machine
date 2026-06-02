#version 300 es
precision highp float;

uniform vec2  uResolution;
uniform vec2  uViewportOffset;
uniform vec4  uControl;
uniform vec2  uMoilSize;
uniform vec2  uMoilCenter;
uniform float uCpRatio;

uniform float p0, p1, p2, p3, p4, p5;

// ======================================================
// YUY2 packed texture
// RGBA = Y0 U Y1 V
//
// texture size:
// width  = image_width / 2
// height = image_height
// ======================================================
uniform sampler2D uYUY2Tex;

// GPU post-transform
uniform vec2  uFlip;
uniform vec2  uMarginScale;
uniform vec2  uMarginOffset;

out vec4 fragColor;

#define PI 3.14159265358979323846

// ======================================================
// Lens Polynomial
// ======================================================
float calculateRho(float alpha)
{
    float a2 = alpha * alpha;
    float a3 = a2 * alpha;
    float a4 = a3 * alpha;
    float a5 = a4 * alpha;
    float a6 = a5 * alpha;

    return (
        p0 * a6 +
        p1 * a5 +
        p2 * a4 +
        p3 * a3 +
        p4 * a2 +
        p5 * alpha
    ) * uCpRatio;
}

// ======================================================
// YUY2 Sampling
//
// RGBA = Y0 U Y1 V
//
// even pixel -> Y0
// odd  pixel -> Y1
// ======================================================
vec4 sampleYUY2(vec2 uv)
{
    vec2 texSize = vec2(textureSize(uYUY2Tex, 0));

    // image pixel coordinate
    vec2 pixel = uv * uMoilSize;

    // pair index (2 pixels per texel)
    float pairX = floor(pixel.x * 0.5);

    // sample coordinate in packed texture
    vec2 texCoord = vec2(
        (pairX + 0.5) / texSize.x,
        (floor(pixel.y) + 0.5) / texSize.y
    );

    vec4 yuyv = texture(uYUY2Tex, texCoord);

    float Y;

    // even/odd pixel select
    if (mod(floor(pixel.x), 2.0) < 0.5)
        Y = yuyv.r;
    else
        Y = yuyv.b;

    float U = yuyv.g - 0.5;
    float V = yuyv.a - 0.5;

    // BT.601 Full Range
    vec3 rgb = clamp(vec3(
        Y + 1.402 * V,
        Y - 0.344136 * U - 0.714136 * V,
        Y + 1.772 * U
    ), 0.0, 1.0);

    return vec4(rgb, 1.0);
}

// ======================================================
// Chromatic Aberration
// ======================================================
vec4 sampleWithAberration(vec2 uv)
{
    const float strength = 0.0025;

    vec2 dir = uv - 0.5;

    vec2 uvR = clamp(uv + dir * strength, 0.0, 1.0);
    vec2 uvB = clamp(uv - dir * strength, 0.0, 1.0);

    return vec4(
        sampleYUY2(uvR).r,
        sampleYUY2(uv).g,
        sampleYUY2(uvB).b,
        1.0
    );
}

// ======================================================
// Edge Fade
// ======================================================
float boundaryAlpha(vec2 texCoord)
{
    vec2 edge = min(texCoord, 1.0 - texCoord);

    float fadePixels =
        3.0 / min(uMoilSize.x, uMoilSize.y);

    return smoothstep(
        0.0,
        fadePixels,
        min(edge.x, edge.y)
    );
}

// ======================================================
// Vignette
// ======================================================
float vignette(vec2 uvScreen)
{
    vec2 d = uvScreen - 0.5;

    return clamp(
        pow(1.0 - dot(d, d) * 1.8, 3.0),
        0.0,
        1.0
    );
}

// ======================================================
// Main
// ======================================================
void main()
{
    vec2 localFragCoord =
        gl_FragCoord.xy - uViewportOffset;

    vec2 fragCoord = vec2(
        localFragCoord.x,
        uResolution.y - localFragCoord.y
    );

    // ==================================================
    // GPU post-transform
    // ==================================================
    if (uFlip.x > 0.5)
        fragCoord.x = uResolution.x - fragCoord.x;

    if (uFlip.y > 0.5)
        fragCoord.y = uResolution.y - fragCoord.y;

    fragCoord =
        fragCoord * uMarginScale + uMarginOffset;

    // ==================================================
    // Fisheye constants
    // ==================================================
    const float PCT_W = 1.27;
    const float PCT_H = 1.27;
    const float FOCAL_ZOOM = 410.0;

    float dcx = uResolution.x * 0.5;
    float dcy = uResolution.y * 0.5;

    float mAlphaOffset =
        uControl.x * (PI / 180.0);

    float mBetaOffset =
        uControl.y * (PI / 180.0);

    float zoom_val =
        uControl.z;

    float widthCosB =
        PCT_W * cos(mBetaOffset);

    float widthSinB =
        PCT_W * sin(mBetaOffset);

    float heightCosASinB =
        PCT_H *
        cos(mAlphaOffset) *
        sin(mBetaOffset);

    float flZoomSinASinB =
        FOCAL_ZOOM *
        zoom_val *
        sin(mAlphaOffset) *
        sin(mBetaOffset);

    float flZoomSinACosB =
        FOCAL_ZOOM *
        zoom_val *
        sin(mAlphaOffset) *
        cos(mBetaOffset);

    // ==================================================
    // 3D projection
    // ==================================================
    float tempX =
        -(
            (fragCoord.x - dcx) * widthCosB +
            (fragCoord.y - dcy) * heightCosASinB +
            flZoomSinASinB
        );

    float tempY =
        -(
            (fragCoord.y - dcy) *
            (PCT_H * cos(mAlphaOffset))
            -
            (FOCAL_ZOOM *
             zoom_val *
             sin(mAlphaOffset))
        );

    float tempZ =
        -(
            (fragCoord.x - dcx) * widthSinB
            -
            (fragCoord.y - dcy) *
            (
                PCT_H *
                sin(mAlphaOffset) *
                cos(mBetaOffset)
            )
            -
            flZoomSinACosB
        );

    float alpha =
        atan(
            sqrt(tempX * tempX + tempY * tempY),
            tempZ
        );

    float beta = 0.0;

    if (tempX != 0.0)
    {
        beta = atan(tempY, tempX);
    }
    else
    {
        beta =
            (tempY >= 0.0)
            ? (PI / 2.0)
            : (-PI / 2.0);
    }

    // ==================================================
    // Lens mapping
    // ==================================================
    float rho = calculateRho(alpha);

    float sourceX =
        uMoilCenter.x - rho * cos(beta);

    float sourceY =
        uMoilCenter.y - rho * sin(beta);

    vec2 texCoord =
        vec2(sourceX, sourceY) / uMoilSize;

    // ==================================================
    // Outside image
    // ==================================================
    if (
        texCoord.x < 0.0 ||
        texCoord.x > 1.0 ||
        texCoord.y < 0.0 ||
        texCoord.y > 1.0
    )
    {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // ==================================================
    // Sample image
    // ==================================================
    vec4 col = sampleWithAberration(texCoord);

    // edge fade
    col.rgb *= boundaryAlpha(texCoord);

    // vignette
    vec2 uvScreen = fragCoord / uResolution;

    col.rgb = mix(
        col.rgb,
        col.rgb * vignette(uvScreen),
        0.35
    );

    // slight contrast
    col.rgb =
        (col.rgb - 0.5) * 1.05 + 0.5;

    // saturation
    float lum = dot(
        col.rgb,
        vec3(0.2126, 0.7152, 0.0722)
    );

    col.rgb = mix(
        vec3(lum),
        col.rgb,
        1.08
    );

    fragColor = vec4(col.rgb, 1.0);
}