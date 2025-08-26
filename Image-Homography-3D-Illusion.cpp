#include <opencv2/opencv.hpp>
#include "MatrixInverse.h"
#include "vec.h"
#include "mat.h"

IplImage* dst = nullptr;
IplImage* src = nullptr;
int W = 500;										// 결과 이미지 가로 크기
int H = 500;										// 결과 이미지 세로 크기

vec3 pos[8] = {										// 육면체를 구성하는 8개의 꼭지점 3차원 좌표
		vec3(-0.5, -0.5,  0.5),
		vec3(-0.5,  0.5,  0.5),
		vec3(0.5,  0.5,  0.5),
		vec3(0.5, -0.5,  0.5),
		vec3(-0.5, -0.5, -0.5),
		vec3(-0.5,  0.5, -0.5),
		vec3(0.5,  0.5, -0.5),
		vec3(0.5, -0.5, -0.5) };



struct rect											// 사각형 한 면
{
	int ind[4];										// 꼭지점의 인덱스
	vec3 pos[4];									// 꼭지점의 화면 방향으로의 3차원 위치
	vec3 nor;										// 법선(normal) 벡터 방향 (= 면이 향하는 방향)
};

rect setRect(int a, int b, int c, int d)			// 사각형 정보를 채워주는 함수(바로 아래 cube정의에 사용)
{
	rect r;
	r.ind[0] = a;
	r.ind[1] = b;
	r.ind[2] = c;
	r.ind[3] = d;
	return r;
}

rect cube[6] = { setRect(1, 0, 3, 2),				// 사각형 6개를 정의해 육면체를 구성
				 setRect(2, 3, 7, 6),
				 setRect(3, 0, 4, 7),
				 setRect(6, 5, 1, 2),
				 setRect(6, 7, 4, 5),
				 setRect(5, 4, 0, 1) };

vec3 epos = vec3(1.5, 1.5, 1.5);					// 카메라(시점의 3차원) 위치
mat4 ModelMat;										// 모델에 변형을 주는 변형 행렬
mat4 ViewMat;										// 카메라 시점을 맞춰주는 변형 행렬
mat4 ProjMat;										// 화면상 위치로 투영해주는 변형 행렬

void init()											// 초기화
{	
	ModelMat = mat4(1.0f);
	ViewMat = LookAt(epos, vec3(0, 0, 0), vec3(0, 1, 0));  
													// 카메라 위치(epos)에서 (0,0,0)을 바라보는 카메라 설정			
	ProjMat = Perspective(45, W / (float)H, 0.1, 100);	
													// 45도의 시야각을 가진 투영 변환 (가시거리 0.1~100)
}

void rotateModel(float rx, float ry, float rz)		// 육면체 모델에 회전을 적용하는 함수
{
	ModelMat = RotateX(rx) * RotateY(ry) * RotateZ(rz) * ModelMat;
}

vec3 convert3Dto2D(vec3 in)							// 3차원 좌표를 화면에 투영된 2차원+깊이값(z) 좌표로 변환
{
	vec4 p = ProjMat * ViewMat * ModelMat * vec4(in);
	p.x /= p.w;
	p.y /= p.w;
	p.z /= p.w;
	p.x = (p.x + 1) / 2.0f * W;
	p.y = (-p.y + 1) / 2.0f * H;
	return vec3(p.x, p.y, p.z);
}

void updatePosAndNormal(rect* r, vec3 p[])			// 육면체의 회전에 따른 각 면의 3차원 좌표 및 법선 벡터 방향 업데이트
{
	for (int i = 0; i < 4; i++)
		r->pos[i] = convert3Dto2D(p[r->ind[i]]);
	vec3 a = normalize(r->pos[0] - r->pos[1]);
	vec3 b = normalize(r->pos[2] - r->pos[1]);
	r->nor = cross(a, b);
}

void drawImage()									// 그림을 그린다 (각 면의 테두리를 직선으로 그림)
{
	int h = src->height;
	int w = src->width;

	//변환시킬 원본 이미지의 네 꼭짓점
	CvPoint p[4] = { cvPoint(0,0),cvPoint(w - 1,0), cvPoint(w - 1,h - 1), cvPoint(0,h - 1) };

	cvSet(dst, cvScalar(0, 0, 0));
	for (int i = 0; i < 6; i++)
	{
		updatePosAndNormal(&cube[i], pos);
		if (cube[i].nor.z < 0) continue;			// 보이지 않는 사각형을 제외, 보이는 사각형만 그린다	
		
		//원본 이미지를 넣을 사각형의 네 꼭짓점
		vec3 q[4] = { cube[i].pos[0],cube[i].pos[1], cube[i].pos[2], cube[i].pos[3] };

		//선형 방정식을 정리한 행렬
		float A[8][8] = { {-p[0].x, -p[0].y, -1, 0, 0, 0, q[0].x * p[0].x, q[0].x * p[0].y}, 
							{0, 0, 0, -p[0].x, -p[0].y, -1, q[0].y * p[0].x, q[0].y * p[0].y},
							{-p[1].x, -p[1].y, -1, 0, 0, 0, q[1].x * p[1].x, q[1].x * p[1].y},
							{0, 0, 0, -p[1].x, -p[1].y, -1, q[1].y * p[1].x, q[1].y * p[1].y},
							{-p[2].x, -p[2].y, -1, 0, 0, 0, q[2].x * p[2].x, q[2].x * p[2].y},
							{0, 0, 0, -p[2].x, -p[2].y, -1, q[2].y * p[2].x, q[2].y * p[2].y},
							{-p[3].x, -p[3].y, -1, 0, 0, 0, q[3].x * p[3].x, q[3].x * p[3].y},
							{0, 0, 0, -p[3].x, -p[3].y, -1, q[3].y * p[3].x, q[3].y * p[3].y} };
		float b[8] = { -q[0].x,-q[0].y,-q[1].x,-q[1].y, -q[2].x,-q[2].y, -q[3].x,-q[3].y};

		float invA[8][8] = { 0 };	//A의 역행렬

		InverseMatrixGJ8(A, invA);

		float hg[8] = { 0 };	//h33을 제외한 homography 행렬
		for (int j = 0; j < 8; j++) {
			for (int k = 0; k < 8; k++) {
				hg[j] += invA[j][k] * b[k];
			}
		}

		float HG[3][3] = { {hg[0],hg[1],hg[2]},
						{hg[3],hg[4],hg[5]},
						{hg[6],hg[7],1.0f} };	//homography 행렬

		float invH[3][3] = { 0 };
		InverseMatrixGJ3(HG, invH);

		//homography 변환을 역변화으로 실행
		for (int y2 = 0; y2 < 500; y2++) {
			for (int x2 = 0; x2 < 500; x2++) {
				float w1 = (invH[2][0] * x2 + invH[2][1] * y2 + invH[2][2]);
				float x1 = (invH[0][0] * x2 + invH[0][1] * y2 + invH[0][2]) / w1;
				float y1 = (invH[1][0] * x2 + invH[1][1] * y2 + invH[1][2]) / w1;

				if (x1 < 0 || x1 > w-1 || y1 < 0 || y1 > h-1)continue;
				CvScalar f = cvGet2D(src, y1, x1);
				cvSet2D(dst, y2, x2, f);
			}
		}

		for (int j = 0; j < 4; j++)
		{
			vec3 p1 = cube[i].pos[j];
			vec3 p2 = cube[i].pos[(j + 1) % 4];

			cvLine(dst, cvPoint(p1.x, p1.y), cvPoint(p2.x, p2.y), cvScalar(255, 255, 255), 3);
		}
	}
		
	cvShowImage("3D view", dst);
}

void myMouse(int event, int x, int y, int flags, void*)
{
	static CvPoint prev = cvPoint(0, 0);
	if (event == CV_EVENT_LBUTTONDOWN)
		prev = cvPoint(x, y);
	if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) == CV_EVENT_FLAG_LBUTTON)
	{
		int dx = x - prev.x;
		int dy = y - prev.y;
		rotateModel(dy, dx, -dy);					// 마우스 조작에 따라 모델을 회전함
		drawImage();
		prev = cvPoint(x, y);
	}
}

int main()
{
	dst = cvCreateImage(cvSize(W, H), 8, 3);

	printf("Input File Path: ");
	char file[100];
	scanf("%s", file);

	src = cvLoadImage(file);
	init();

	while (true)
	{
		rotateModel(0, 1, 0);
		drawImage();
		cvSetMouseCallback("3D view", myMouse);
		int key = cvWaitKey(1);
		if (key == ' ') key = cvWaitKey();
	}

	return 0;
}