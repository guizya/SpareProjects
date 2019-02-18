using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class UpWallScript : MonoBehaviour {

	// Use this for initialization
	void Start () {
        Mesh mesh = GetComponent<MeshFilter>().mesh;

        Debug.Log("Triangle " + mesh.triangles.Length);
        for (int i = 0; i < mesh.triangles.Length; i++)
        {
            Debug.Log("i " + mesh.triangles[i] + " vertex " + mesh.vertices[mesh.triangles[i]].ToString());
        }

        Debug.Log(mesh.uv.Length);
        //mesh.uv[7].x = -0.125f;
        //mesh.uv[7].y = 0.25f;
        //mesh.uv[6].x = 1.125f;
        //mesh.uv[6].y = 0.25f;
        //mesh.uv[10].x = 1.125f;
        //mesh.uv[10].y = 0;
        //mesh.uv[11].x = -0.125f;
        //mesh.uv[11].y = 0;

        //mesh.uv[7].x = 1.125f;
        //mesh.uv[7].y = 0.25f;
        //mesh.uv[6].x = -.125f;
        //mesh.uv[6].y = 0.25f;
        //mesh.uv[10].x = -.125f;
        //mesh.uv[10].y = 0;
        //mesh.uv[11].x = 1.125f;
        //mesh.uv[11].y = 0;

        //mesh.uv[0].x = 1.125f;
        //mesh.uv[0].y = 0.25f;
        //mesh.uv[1].x = -.125f;
        //mesh.uv[1].y = 0.25f;
        //mesh.uv[2].x = -.125f;
        //mesh.uv[2].y = 0;
        //mesh.uv[3].x = 1.125f;
        //mesh.uv[3].y = 0;
    }
	
	// Update is called once per frame
	void Update () {

    }
}
