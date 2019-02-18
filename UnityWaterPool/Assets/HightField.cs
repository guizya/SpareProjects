using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HightField : MonoBehaviour {
    public Vector3[] newVertices;
    public Vector3[] tempVertices;
    public List<Vector3> newNormals;
    public Vector2[] newUV;
    public int[] newTriangles;
    public float[] tempU;
    public Texture2D debugPlanetexture;
    public Texture2D textureLeft;
    public Texture2D textureRight;
    public Texture2D textureUp;
    public Texture2D textureDown;
    public Texture2D textureBottom;
    public Quaternion rotation = Quaternion.Euler(30, -30, 0);
    public Vector3 lightDirection;
    public float RefractN;

    public float[,] phamtonMapUp;
    public float[,] phamtonMapDown;
    public float[,] phamtonMapLeft;
    public float[,] phamtonMapRight;
    public float[,] phamtonMapBottom;
    public int phamtonWidth;
    public int lightWidth;
    public Vector3[] wall1;
    public Vector3[] wall2;
    public Vector3[] wall3;
    public Vector3[] wall4;

    [HideInInspector]
    public int slices;
    public float yHeight;
    public float size;
    public float force;
    public int count;

    // Use this for initialization
    void Start() {
        Mesh mesh = GetComponent<MeshFilter>().mesh;

        mesh.Clear();

        count = 0;
        slices = 256;
        yHeight = 1.0f;
        size = 8.0f;
        force = 0.5f;
        int slices_1 = slices - 1;
        float multiplier = 1.0f / slices_1;
        newVertices = new Vector3[slices * slices];
        tempU = new float[slices * slices];
        newNormals = new List<Vector3>(slices * slices);
        newUV = new Vector2[slices * slices];
        for (int i = 0; i < slices; i++)
        {
            for (int j = 0; j < slices; j++)
            {
                int index = i * slices + j;
                newVertices[index].x = i * size * multiplier - size / 2.0f;
                newVertices[index].y = yHeight;
                newVertices[index].z = j * size * multiplier - size / 2.0f;

                newUV[index].x = i * multiplier;
                newUV[index].y = j * multiplier;
                tempU[index] = 0;
            }
        }
        newTriangles = new int[slices_1 * slices_1 * 2 * 3];
        for (int i = 0; i < slices_1; i++)
        {
            for (int j = 0; j < slices_1; j++)
            {
                int index = (i * slices_1 + j) * 2;
                newTriangles[index * 3] = i * slices + j;
                newTriangles[index * 3 + 1] = i * slices + (j + 1);
                newTriangles[index * 3 + 2] = (i + 1) * slices + j;

                newTriangles[index * 3 + 3] = (i + 1) * slices + j;
                newTriangles[index * 3 + 4] = i * slices + (j + 1);
                newTriangles[index * 3 + 5] = (i + 1) * slices + (j + 1);
            }
        }

        //newVertices[slices * slices / 2 + slices / 2].y = force;
        applyForce(slices / 2, slices / 2);
        tempVertices = new Vector3[slices * slices];

        // Do some calculations...
        mesh.vertices = newVertices;
        mesh.uv = newUV;
        mesh.triangles = newTriangles;
        mesh.RecalculateNormals();

        phamtonWidth = 64;
        phamtonMapUp = new float[phamtonWidth, phamtonWidth];
        phamtonMapDown = new float[phamtonWidth, phamtonWidth];
        phamtonMapLeft = new float[phamtonWidth, phamtonWidth];
        phamtonMapRight = new float[phamtonWidth, phamtonWidth];
        phamtonMapBottom = new float[phamtonWidth, phamtonWidth];
        lightWidth = 128;

        wall1 = new Vector3[] { new Vector3(-4, 1.5f, -4), new Vector3(-4, 0, -4), new Vector3(-4, 0, 4), new Vector3(-4, 1.5f, 4) };
        wall2 = new Vector3[] { new Vector3(4, 1.5f, -4), new Vector3(4, 0, -4), new Vector3(4, 0, 4), new Vector3(4, 1.5f, 4) };
        wall3 = new Vector3[] { new Vector3(-4, 1.5f, -4), new Vector3(-4, 0, -4), new Vector3(4, 0, -4), new Vector3(4, 1.5f, -4) };
        wall4 = new Vector3[] { new Vector3(-4, 1.5f, 4), new Vector3(-4, 0, 4), new Vector3(4, 0, 4), new Vector3(4, 1.5f, 4) };

        debugPlanetexture = new Texture2D(phamtonWidth, phamtonWidth);
        textureLeft = new Texture2D(phamtonWidth, phamtonWidth);
        textureRight = new Texture2D(phamtonWidth, phamtonWidth);
        textureUp = new Texture2D(phamtonWidth, phamtonWidth);
        textureDown = new Texture2D(phamtonWidth, phamtonWidth);
        textureBottom = new Texture2D(phamtonWidth, phamtonWidth);
        lightDirection = new Vector3(0, 0, 1.0f);
        lightDirection = rotation * lightDirection;
        lightDirection.Normalize();
        //Debug.Log(lightDirection.x + " " + lightDirection.y + " " + lightDirection.z);

        RefractN = 1.1f;
    }

    void applyForce(int i, int j)
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;

        float h = size / (slices - 1);
        int num = (int)(force / h);
        for (int m = 0; m <= num; m++)
        {
            for (int n = 0; n <= num; n++)
            {
                float value = force * force - (m * h * m * h) - (n * h * n * h);

                value = value < 0 ? 0 : Mathf.Sqrt(value);
                if (i + m < slices && j + n < slices) newVertices[(i + m) * slices + (j + n)].y += value;
                if (m != 0 && i - m >= 0 && j + n < slices) newVertices[(i - m) * slices + (j + n)].y += value;
                if (n != 0 && i + m < slices && j - n >= 0) newVertices[(i + m) * slices + (j - n)].y += value;
                if (m != 0 && n != 0 && i - m >= 0 && j - n >= 0) newVertices[(i - m) * slices + (j - n)].y += value;
            }
        }
        mesh.vertices = newVertices;
        mesh.RecalculateNormals();
    }

    // Update is called once per frame
    void Update() {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.GetNormals(newNormals);

        mousEvent();
        //randomEvent();
        //averageNormals();
        reflection();
    }

    void FixedUpdate()
    {
        //Debug.Log("Mouse is over GameObject. ");
        for (int i = 0; i < slices; i++)
        {
            for (int j = 0; j < slices; j++)
            {
                int index = i * slices + j;
                int idx_00 = i == 0 ? index : ((i - 1) * slices + j);
                int idx_10 = i == (slices - 1) ? index : ((i + 1) * slices + j);
                int idx_01 = j == 0 ? index : (i * slices + j - 1);
                int idx_11 = j == (slices - 1) ? index : (i * slices + j + 1);

                float val_00 = newVertices[idx_00].y;
                float val_10 = newVertices[idx_10].y;
                float val_01 = newVertices[idx_01].y;
                float val_11 = newVertices[idx_11].y;
                float val = newVertices[index].y;

                float temp = (val_00 + val_01 + val_10 + val_11 - 4 * val) / 4.0f;
                tempU[index] += temp;
                tempU[index] *= 0.99f;
                //tempVertices[index].y = temp;
            }
        }

        for (int i = 0; i < slices; i++)
        {
            for (int j = 0; j < slices; j++)
            {
                int index = i * slices + j;
                //        newVertices[index].y += tempVertices[index].y;
                newVertices[index].y += tempU[index];
            }
        }

        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = newVertices;
        mesh.RecalculateNormals();
    }

    void averageNormals()
    {
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        List<Vector3> normals = new List<Vector3>();
        mesh.GetNormals(normals);

        for (int i = 0; i < slices; i++)
        {
            for (int j = 0; j < slices; j++)
            {
                int index = i * slices + j;

                if (i == 0 || j == 0 || i == slices - 1 || j == slices - 1)
                {
                    newNormals[index] = normals[index];
                    continue;
                }

                newNormals[index] = normals[index];
                newNormals[index] += normals[(i - 1) * slices + j];
                newNormals[index] += normals[(i + 1) * slices + j];

                newNormals[index] += normals[i * slices + j - 1];
                newNormals[index] += normals[i * slices + j + 1];

                newNormals[index] /= 5.0f;
            }
        }
        mesh.SetNormals(newNormals);
    }

    void mousEvent()
    {
        if (!Input.GetMouseButtonDown(0)) return;

        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        //Debug.Log("Ray " + ray.ToString());
        if (ray.direction.y == 0) return;

        float k = (yHeight - ray.origin.y) / ray.direction.y;
        if (k <= 0) return;


        float x = ray.origin.x + k * ray.direction.x;
        float y = ray.origin.z + k * ray.direction.z;
        //Debug.Log("x " + x + " y " + y + " k " + k);

        int i = (int)(x / size * slices) + slices / 2;
        int j = (int)(y / size * slices) + slices / 2;

        if (i < 0 || i >= slices || j < 0 || j >= slices) return;

        //Debug.Log("Mouse is over GameObject. " + i + " " + j + " " + x + " " + y);
        //newVertices[i * slices + j].y = force;
        applyForce(i, j);
    }

    void randomEvent()
    {
        float f = Random.Range(0, force);
        int i = Random.Range(0, slices - 1);
        int j = Random.Range(0, slices - 1);
        newVertices[i * slices + j].y += f;
    }

    void getLitPosition(int ii, int jj, float wx, float wy, ref Vector3 vvv)
    {
        vvv.Set(0, 0, 0);

        vvv.x += (1 - wx) * (1 - wy) * newVertices[ii * slices + jj].x;
        vvv.y += (1 - wx) * (1 - wy) * newVertices[ii * slices + jj].y;
        vvv.z += (1 - wx) * (1 - wy) * newVertices[ii * slices + jj].z;

        //Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += wx * (1 - wy) * newVertices[(ii + 1) * slices + jj].x;
        vvv.y += wx * (1 - wy) * newVertices[(ii + 1) * slices + jj].y;
        vvv.z += wx * (1 - wy) * newVertices[(ii + 1) * slices + jj].z;
        //Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += (1 - wx) * wy * newVertices[ii * slices + jj + 1].x;
        vvv.y += (1 - wx) * wy * newVertices[ii * slices + jj + 1].y;
        vvv.z += (1 - wx) * wy * newVertices[ii * slices + jj + 1].z;
        //Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += wx * wy * newVertices[(ii + 1) * slices + jj + 1].x;
        vvv.y += wx * wy * newVertices[(ii + 1) * slices + jj + 1].y;
        vvv.z += wx * wy * newVertices[(ii + 1) * slices + jj + 1].z;
        //Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);
    }

    void getLitNormal(int ii, int jj, float wx, float wy, ref Vector3 vvv)
    {
        vvv.Set(0, 0, 0);

        //if (ii > 254 || jj > 254) Debug.Log(ii + " HHH " + jj + " " + newNormals.Count);

        vvv.x += (1 - wx) * (1 - wy) * newNormals[ii * slices + jj].x;
        vvv.y += (1 - wx) * (1 - wy) * newNormals[ii * slices + jj].y;
        vvv.z += (1 - wx) * (1 - wy) * newNormals[ii * slices + jj].z;

        ////Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += wx * (1 - wy) * newNormals[(ii + 1) * slices + jj].x;
        vvv.y += wx * (1 - wy) * newNormals[(ii + 1) * slices + jj].y;
        vvv.z += wx * (1 - wy) * newNormals[(ii + 1) * slices + jj].z;
        ////Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += (1 - wx) * wy * newNormals[ii * slices + jj + 1].x;
        vvv.y += (1 - wx) * wy * newNormals[ii * slices + jj + 1].y;
        vvv.z += (1 - wx) * wy * newNormals[ii * slices + jj + 1].z;
        ////Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);

        vvv.x += wx * wy * newNormals[(ii + 1) * slices + jj + 1].x;
        vvv.y += wx * wy * newNormals[(ii + 1) * slices + jj + 1].y;
        vvv.z += wx * wy * newNormals[(ii + 1) * slices + jj + 1].z;
        ////Debug.Log("x " + vvv.x + " y " + vvv.y + " z " + vvv.z);
    }

    bool RayIntersectsTriangle(Vector3 position, Vector3 direction, Vector3 v0, Vector3 v1, Vector3 v2, ref Vector3 intersect)
    {
        const float EPSILON = 0.0000001f;
        Vector3 edge1, edge2, h, s, q;
        float a, f, u, v;

        edge1 = v1 - v0;
        edge2 = v2 - v0;

        h = Vector3.Cross(direction, edge2);
        a = Vector3.Dot(h, edge1);
        if (a > -EPSILON && a < EPSILON)
            return false;

        f = 1 / a;
        s = position - v0;
        u = f * (Vector3.Dot(s, h));
        if (u < 0.0 || u > 1.0)
            return false;

        q = Vector3.Cross(s, edge1);
        v = f * Vector3.Dot(direction, q);
        if (v < 0.0 || u + v > 1.0)
            return false;

        float t = f * Vector3.Dot(edge2, q);
        if (t > EPSILON) // ray intersection
        {
            intersect = position + direction * t;
            return true;
        }

        return false;
    }

    bool occluded(Vector3 position, Vector3 direction)
    {
        //Vector3 temp = new Vector3();
        //return RayIntersectsTriangle(position, direction, wall1[0], wall1[1], wall1[2], ref temp) || RayIntersectsTriangle(position, direction, wall1[2], wall1[2], wall1[3], ref temp) ||
        //       RayIntersectsTriangle(position, direction, wall2[0], wall2[1], wall2[2], ref temp) || RayIntersectsTriangle(position, direction, wall2[2], wall2[2], wall2[3], ref temp) ||
        //       RayIntersectsTriangle(position, direction, wall3[0], wall3[1], wall3[2], ref temp) || RayIntersectsTriangle(position, direction, wall3[2], wall3[2], wall3[3], ref temp) ||
        //       RayIntersectsTriangle(position, direction, wall4[0], wall4[1], wall4[2], ref temp) || RayIntersectsTriangle(position, direction, wall4[2], wall4[2], wall4[3], ref temp);

        float temp = 4.0f - position.x;
        bool flag1 = (1.5f - position.y) / temp >= Mathf.Abs(lightDirection.y / lightDirection.x);
        temp = position.z + 4.0f;
        bool flag2 = (1.5f - position.y) / temp > Mathf.Abs(lightDirection.y / lightDirection.z);

        return flag1 || flag2;
    }

    void reflection()
    {
        System.Array.Clear(phamtonMapRight, 0, phamtonWidth * phamtonWidth);
        System.Array.Clear(phamtonMapLeft, 0, phamtonWidth * phamtonWidth);
        System.Array.Clear(phamtonMapDown, 0, phamtonWidth * phamtonWidth);
        System.Array.Clear(phamtonMapUp, 0, phamtonWidth * phamtonWidth);
        System.Array.Clear(phamtonMapBottom, 0, phamtonWidth * phamtonWidth);


        float multiplier = 1.0f / (lightWidth - 1);
        float multiplier_1 = 1.0f / (slices - 1);
        float multiplier_2 = multiplier / multiplier_1;
        Vector3 position = new Vector3();
        Vector3 direction = new Vector3();
        Vector3 normal = new Vector3();
        Vector3 light = new Vector3(2, 6, -4);
        //int i = lightWidth - 30; int j = lightWidth - 20;
        for (int i = 1; i < lightWidth - 1; i++)
        {
            for (int j = 1; j < lightWidth - 1; j++)
            {
                // first get the lit position
                float x = i * multiplier_2;
                float y = j * multiplier_2;

                int ii = (int)x; int jj = (int)y;
                float wx = x - ii;
                float wy = y - jj;

                //Debug.Log("i " + i + " j " + j + " ii " + ii + " jj " + jj + " wx " + wx + " wy " + wy);
                getLitPosition(ii, jj, wx, wy, ref position);
                //Debug.Log("position " + position.ToString() + " " + position.x);

                // second get the lit direction
                direction = light - position;

                // check if it is occluded by mesh
                bool lit = !occluded(position, direction);
                if (!lit) continue;

                getLitNormal(ii, jj, wx, wy, ref normal);
                direction = Vector3.Reflect(lightDirection, normal);
                litOnWalls(position, direction, 0.1f);

                lit = refract(lightDirection, normal, ref direction);
                if (!lit) continue;
                refractOnWalls(position, direction, 0.1f);
            }
        }
        {
            for (int i = 0; i < debugPlanetexture.width; i++)
            {
                for (int j = 0; j < debugPlanetexture.height; j++)
                {
                    debugPlanetexture.SetPixel(i, j, new Color(phamtonMapBottom[i, j], phamtonMapBottom[i, j], phamtonMapBottom[i, j]));
                }
            }

            debugPlanetexture.Apply();

            for (int i = 0; i < phamtonWidth; i++)
            {
                for (int j = 0; j < phamtonWidth; j++)
                {
                    textureLeft.SetPixel(i, j, new Color(phamtonMapLeft[i, j], phamtonMapLeft[i, j], phamtonMapLeft[i, j]));
                    textureRight.SetPixel(i, j, new Color(phamtonMapRight[i, j], phamtonMapRight[i, j], phamtonMapRight[i, j]));
                    textureUp.SetPixel(i, j, new Color(phamtonMapUp[i, j], phamtonMapUp[i, j], phamtonMapUp[i, j]));
                    textureDown.SetPixel(i, j, new Color(phamtonMapDown[i, j], phamtonMapDown[i, j], phamtonMapDown[i, j]));
                    textureBottom.SetPixel(i, j, new Color(phamtonMapBottom[i, j], phamtonMapBottom[i, j], phamtonMapBottom[i, j]));
                }
            }

            textureLeft.Apply();
            textureRight.Apply();
            textureUp.Apply();
            textureDown.Apply();
            textureBottom.Apply();

            Shader.SetGlobalTexture("DebugPlaneTexture", debugPlanetexture);
            Shader.SetGlobalTexture("TextureLeft", textureLeft);
            Shader.SetGlobalTexture("TextureRight", textureRight);
            Shader.SetGlobalTexture("TextureUp", textureUp);
            Shader.SetGlobalTexture("TextureDown", textureDown);
            Shader.SetGlobalTexture("TextureBottom", textureBottom);
            Shader.SetGlobalInt("PhantomWidth", phamtonWidth);
        }
    }

    void litOnWalls(Vector3 position, Vector3 direction, float weight)
    {
        float h = 8.0f / (phamtonWidth - 1);
        h = 1.0f / h;
        //the right wall
        if (direction.x > 0)
        {
            Vector3 temp = position + (4.0f - position.x) / direction.x * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.z < 4.0f && temp.z > -4.0f)
            {
                float iDepth = temp.z + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[(slices - 1) * slices + ii].y < temp.y)
                {

                    phamtonMapRight[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapRight[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapRight[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapRight[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapRight[ii, jj] = phamtonMapRight[ii, jj] > 1.0f ? 1.0f : phamtonMapRight[ii, jj];
                    phamtonMapRight[ii + 1, jj] = phamtonMapRight[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapRight[ii + 1, jj];
                    phamtonMapRight[ii, jj + 1] = phamtonMapRight[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapRight[ii, jj + 1];
                    phamtonMapRight[ii + 1, jj + 1] = phamtonMapRight[ii + 1, jj + 1] > 1.0f ? 1.0f : phamtonMapRight[ii + 1, jj + 1];
                    return;
                }
            }
        }

        // the left wall
        if (direction.x < 0)
        {
            Vector3 temp = position - (4.0f + position.x) / direction.x * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.z < 4.0f && temp.z > -4.0f)
            {
                float iDepth = temp.z + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[ii].y < temp.y)
                {

                    phamtonMapLeft[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapLeft[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapLeft[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapLeft[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapLeft[ii, jj] = phamtonMapLeft[ii, jj] > 1.0f ? 1.0f : phamtonMapLeft[ii, jj];
                    phamtonMapLeft[ii + 1, jj] = phamtonMapLeft[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapLeft[ii + 1, jj];
                    phamtonMapLeft[ii, jj + 1] = phamtonMapLeft[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapLeft[ii, jj + 1];
                    phamtonMapLeft[ii + 1, jj + 1] = phamtonMapLeft[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapLeft[ii + 1, jj + 1];
                    if (ii >= phamtonWidth || jj >= phamtonWidth || ii < 0 || jj < 0)
                    {
                        Debug.Log("Out of Bound! " + ii + " " + jj);
                    }
                    return;
                }
            }
        }

        // the bottom wall
        if (direction.z < 0)
        {
            Vector3 temp = position - (4.0f + position.z) / direction.z * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.x < 4.0f && temp.x > -4.0f)
            {
                float iDepth = temp.x + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[ii * slices].y < temp.y)
                {
                    phamtonMapDown[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapDown[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapDown[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapDown[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapDown[ii, jj] = phamtonMapDown[ii, jj] > 1.0f ? 1.0f : phamtonMapDown[ii, jj];
                    phamtonMapDown[ii + 1, jj] = phamtonMapDown[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapDown[ii + 1, jj];
                    phamtonMapDown[ii, jj + 1] = phamtonMapDown[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapDown[ii, jj + 1];
                    phamtonMapDown[ii + 1, jj + 1] = phamtonMapDown[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapDown[ii + 1, jj + 1];
                    return;
                }
            }
        }

        // the top wall
        if (direction.z > 0)
        {
            Vector3 temp = position + (4.0f - position.z) / direction.z * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.x < 4.0f && temp.x > -4.0f)
            {
                float iDepth = temp.x + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[ii * slices + slices - 1].y < temp.y)
                {
                    phamtonMapUp[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapUp[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapUp[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapUp[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapUp[ii, jj] = phamtonMapUp[ii, jj] > 1.0f ? 1.0f : phamtonMapUp[ii, jj];
                    phamtonMapUp[ii + 1, jj] = phamtonMapUp[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapUp[ii + 1, jj];
                    phamtonMapUp[ii, jj + 1] = phamtonMapUp[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapUp[ii, jj + 1];
                    phamtonMapUp[ii + 1, jj + 1] = phamtonMapUp[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapUp[ii + 1, jj + 1];
                    return;
                }
            }
        }
    }

    bool refract(Vector3 light, Vector3 normal, ref Vector3 direction)
    {
        float cosTheta = Vector3.Dot(-light, normal);
        if (cosTheta <= 0.2) return false;

        Vector3 horizon = light + normal * cosTheta;
        Vector3.Normalize(horizon);

        float sinTheta = Mathf.Sqrt(1 - cosTheta * cosTheta);
        float sinTheta1 = sinTheta / RefractN;

        float cosTheta1 = Mathf.Sqrt(1 - sinTheta1 * sinTheta1);
        direction = -normal * cosTheta1 + horizon * sinTheta1;

        return true;
    }

    void refractOnWalls(Vector3 position, Vector3 direction, float weight)
    {
        float h = 8.0f / (phamtonWidth - 1);
        h = 1.0f / h;
        //the right wall
        if (direction.x > 0)
        {
            Vector3 temp = position + (4.0f - position.x) / direction.x * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.z < 4.0f && temp.z > -4.0f)
            {
                float iDepth = temp.z + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[(slices - 1) * slices + ii].y > temp.y)
                {

                    phamtonMapRight[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapRight[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapRight[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapRight[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapRight[ii, jj] = phamtonMapRight[ii, jj] > 1.0f ? 1.0f : phamtonMapRight[ii, jj];
                    phamtonMapRight[ii + 1, jj] = phamtonMapRight[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapRight[ii + 1, jj];
                    phamtonMapRight[ii, jj + 1] = phamtonMapRight[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapRight[ii, jj + 1];
                    phamtonMapRight[ii + 1, jj + 1] = phamtonMapRight[ii + 1, jj + 1] > 1.0f ? 1.0f : phamtonMapRight[ii + 1, jj + 1];
                    return;
                }
            }
        }

        // the left wall
        if (direction.x < 0)
        {
            Vector3 temp = position - (4.0f + position.x) / direction.x * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.z < 4.0f && temp.z > -4.0f)
            {
                float iDepth = temp.z + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[ii].y > temp.y)
                {

                    phamtonMapLeft[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapLeft[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapLeft[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapLeft[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapLeft[ii, jj] = phamtonMapLeft[ii, jj] > 1.0f ? 1.0f : phamtonMapLeft[ii, jj];
                    phamtonMapLeft[ii + 1, jj] = phamtonMapLeft[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapLeft[ii + 1, jj];
                    phamtonMapLeft[ii, jj + 1] = phamtonMapLeft[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapLeft[ii, jj + 1];
                    phamtonMapLeft[ii + 1, jj + 1] = phamtonMapLeft[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapLeft[ii + 1, jj + 1];
                    if (ii >= phamtonWidth || jj >= phamtonWidth || ii < 0 || jj < 0)
                    {
                        Debug.Log("Out of Bound! " + ii + " " + jj);
                    }
                    return;
                }
            }
        }

        // the top wall
        if (direction.z > 0)
        {
            Vector3 temp = position + (4.0f - position.z) / direction.z * direction;
            if (temp.y < 1.5f && temp.y > 0 && temp.x < 4.0f && temp.x > -4.0f)
            {
                float iDepth = temp.x + 4.0f;
                float jDepth = 1.5f - temp.y;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                if (newVertices[ii * slices + slices - 1].y > temp.y)
                {
                    phamtonMapUp[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                    phamtonMapUp[ii + 1, jj] += wx * (1.0f - wy) * weight;
                    phamtonMapUp[ii, jj + 1] += (1.0f - wx) * wy * weight;
                    phamtonMapUp[ii + 1, jj + 1] += wx * wy * weight;
                    phamtonMapUp[ii, jj] = phamtonMapUp[ii, jj] > 1.0f ? 1.0f : phamtonMapUp[ii, jj];
                    phamtonMapUp[ii + 1, jj] = phamtonMapUp[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapUp[ii + 1, jj];
                    phamtonMapUp[ii, jj + 1] = phamtonMapUp[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapUp[ii, jj + 1];
                    phamtonMapUp[ii + 1, jj + 1] = phamtonMapUp[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapUp[ii + 1, jj + 1];
                    return;
                }
            }
        }


        // the bottom wall
        if (direction.y < 0)
        {
            Vector3 temp = position - position.y / direction.y * direction;
            if (temp.z < 4.0f && temp.z > -4.0f && temp.x < 4.0f && temp.x > -4.0f)
            {
                float iDepth = temp.x + 4.0f;
                float jDepth = temp.z + 4.0f;
                int ii = (int)(iDepth * h);
                int jj = (int)(jDepth * h);
                float wx = iDepth * h - ii;
                float wy = jDepth * h - jj;

                phamtonMapBottom[ii, jj] += (1.0f - wx) * (1.0f - wy) * weight;
                phamtonMapBottom[ii + 1, jj] += wx * (1.0f - wy) * weight;
                phamtonMapBottom[ii, jj + 1] += (1.0f - wx) * wy * weight;
                phamtonMapBottom[ii + 1, jj + 1] += wx * wy * weight;
                phamtonMapBottom[ii, jj] = phamtonMapBottom[ii, jj] > 1.0f ? 1.0f : phamtonMapBottom[ii, jj];
                phamtonMapBottom[ii + 1, jj] = phamtonMapBottom[ii + 1, jj] > 1.0f ? 1.0f : phamtonMapBottom[ii + 1, jj];
                phamtonMapBottom[ii, jj + 1] = phamtonMapBottom[ii, jj + 1] > 1.0f ? 1.0f : phamtonMapBottom[ii, jj + 1];
                phamtonMapBottom[ii + 1, jj + 1] = phamtonMapBottom[ii + 1, jj + 1] > 1.0 ? 1.0f : phamtonMapBottom[ii + 1, jj + 1];
                return;
            }
        }
    }
}
