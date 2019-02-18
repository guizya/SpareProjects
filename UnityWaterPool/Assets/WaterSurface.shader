Shader "Custom/WaterSurface" {
	Properties{
		_Color("Color", Color) = (1,1,1,1)
		_MainTex("Albedo (RGB)", 2D) = "white" {}
		_Glossiness("Smoothness", Range(0,1)) = 0.5
		_Metallic("Metallic", Range(0,1)) = 0.0
		_Cube("Cubemap", CUBE) = "" {}
	}
		SubShader{
		Tags{ "RenderType" = "Opaque" }
		LOD 200

		CGPROGRAM
		// Physically based Standard lighting model, and enable shadows on all light types
#pragma surface surf Standard fullforwardshadows

		// Use shader model 3.0 target, to get nicer looking lighting
#pragma target 3.0

		sampler2D _MainTex;
	samplerCUBE _Cube;
	uniform sampler2D TextureLeft;
	uniform sampler2D TextureRight;
	uniform sampler2D TextureUp;
	uniform sampler2D TextureBottom;
	
	struct Input {
		float2 uv_MainTex;
		float3 worldRefl;
		float3 worldPos;
		float3 worldNormal;
		float3 viewDir;
	};

	half _Glossiness;
	half _Metallic;
	fixed4 _Color;

	// Add instancing support for this shader. You need to check 'Enable Instancing' on materials that use the shader.
	// See https://docs.unity3d.com/Manual/GPUInstancing.html for more information about instancing.
	// #pragma instancing_options assumeuniformscaling
	UNITY_INSTANCING_BUFFER_START(Props)
		// put more per-instance properties here
		UNITY_INSTANCING_BUFFER_END(Props)

		void surf(Input IN, inout SurfaceOutputStandard o) {
		// Albedo comes from a texture tinted by color
		fixed4 c = tex2D(_MainTex, IN.uv_MainTex) * _Color;
		float ratio = 0.2;
		o.Albedo = c.rgb * ratio + texCUBE(_Cube, IN.worldRefl).rgb * (1-ratio);

		fixed3 color;
		fixed3 pos;
		fixed2 uv;
		float weightRefl = 0;
		if(IN.worldRefl.x > 0)
		{
			pos = IN.worldPos + (4.0f - IN.worldPos.x) / IN.worldRefl.x * IN.worldRefl;
			if (pos.y < 1.5f && pos.y > 0 && pos.z < 4.0f && pos.z > -4.0f)
			{
				uv.x = (pos.z + 4.0f) / (8.0f);
				uv.y = (1.5f - pos.y) / (8.0f);
				weightRefl = tex2D(TextureRight, uv) - 0.3f;
			}
		}
		else if (IN.worldRefl.x < 0) 
		{
			pos = IN.worldPos - (4.0f + IN.worldPos.x) / IN.worldRefl.x * IN.worldRefl;
			if (pos.y < 1.5f && pos.y > 0 && pos.z < 4.0f && pos.z > -4.0f)
			{
				uv.x = (pos.z + 4.0f) / (8.0f);
				uv.y = (1.5f - pos.y) / (8.0f);
				weightRefl = tex2D(TextureLeft, uv);
			}
		}
		
		if (IN.worldRefl.z > 0)
		{
			pos = IN.worldPos + (4.0f - IN.worldPos.z) / IN.worldRefl.z * IN.worldRefl;
			if (pos.y < 1.5f && pos.y > 0 && pos.x < 4.0f && pos.x > -4.0f)
			{
				uv.x = (pos.x + 4.0f) / (8.0f);
				uv.y = (1.5f - pos.y) / (8.0f);
				weightRefl = tex2D(TextureUp, uv);
			}
		}
		
		// for refraction
		float RefractN = 1.1f;
		float weightRefr = 0;
		float3 dir = IN.viewDir;
		float3 normal = IN.worldNormal;
		normalize(dir); normalize(normal);
		float cosTheta = dir.x * normal.x + dir.y * normal.y + dir.z * normal.z;
		if (cosTheta > 0.2) 
		{
			float3 horizon = -IN.viewDir + IN.worldNormal * cosTheta;
			normalize(horizon);

			float sinTheta = sqrt(1 - cosTheta * cosTheta);
			float sinTheta1 = sinTheta / RefractN;

			float cosTheta1 = sqrt(1 - sinTheta1 * sinTheta1);
			float3 direction = -IN.worldNormal * cosTheta1 + horizon * sinTheta1;

			if (direction.x > 0)
			{
				pos = IN.worldPos + (4.0f - IN.worldPos.x) / direction.x * direction;
				if (pos.y < 1.5f && pos.y > 0 && pos.z < 4.0f && pos.z > -4.0f)
				{
					uv.x = (pos.z + 4.0f) / (8.0f);
					uv.y = (1.5f - pos.y) / (8.0f);
					weightRefr = tex2D(TextureRight, uv) - 0.3f;
				}
			}
			else if (direction.x < 0)
			{
				pos = IN.worldPos - (4.0f + IN.worldPos.x) / direction.x * direction;
				if (pos.y < 1.5f && pos.y > 0 && pos.z < 4.0f && pos.z > -4.0f)
				{
					uv.x = (pos.z + 4.0f) / (8.0f);
					uv.y = (1.5f - pos.y) / (8.0f);
					weightRefr = tex2D(TextureLeft, uv);
				}
			}

			if (direction.z > 0)
			{
				pos = IN.worldPos + (4.0f - IN.worldPos.z) / direction.z * direction;
				if (pos.y < 1.5f && pos.y > 0 && pos.x < 4.0f && pos.x > -4.0f)
				{
					uv.x = (pos.x + 4.0f) / (8.0f);
					uv.y = (1.5f - pos.y) / (8.0f);
					weightRefr = tex2D(TextureUp, uv);
				}
			}

			if (direction.y < 0) 
			{
				pos = IN.worldPos - IN.worldPos.y / direction.y * direction;
				if (pos.z < 4.0f && pos.z > -4.0f && pos.x < 4.0f && pos.x > -4.0f)
				{
					uv.x = (pos.x + 4.0f) / (8.0f);
					uv.y = (pos.z + 4.0f) / (8.0f);
					weightRefr = tex2D(TextureBottom, uv);
				}
			}
		}

		o.Albedo *= 1.0f + weightRefr + weightRefl;
		//o.Albedo.x = weightRefr; o.Albedo.y = weightRefr; o.Albedo.z = weightRefr;
		

		// Metallic and smoothness come from slider variables
		o.Metallic = _Metallic;
		o.Smoothness = _Glossiness;
		o.Alpha = c.a;
	}
	ENDCG
	}
		FallBack "Diffuse"
}
