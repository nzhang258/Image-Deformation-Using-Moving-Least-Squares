#include "impl.h"
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>

void linspace(float start, float end, int num, float *res)
{
  float delta = (end - start) / num;
  for(int i = 0; i < num; ++i) 
    res[i] = start + i * delta;
}

void mls_affine(float *image, int h, int w, float *p, float *q, 
                int ctrls, float alpha, float density, float *result)
{
  std::vector<float> gridX(int(w*density));
  std::vector<float> gridY(int(h*density));
  float *d_gridX = gridX.data();
  float *d_gridY = gridY.data();
  //0~w/h interplation according to density
  linspace(0, w, int(w * density), d_gridX);
  linspace(0, h, int(h * density), d_gridY);
  
  //make grids on the original image
  int grow = gridY.size(), gcol = gridX.size();
  float *d_vx = (float*) malloc(grow * gcol * sizeof(float));
  float *d_vy = (float*) malloc(grow * gcol * sizeof(float));
  for(int i = 0; i < gridX.size() * gridY.size(); ++i) {
    int u = i % gridX.size(), v = i / gridX.size();
	d_vx[i] = d_gridX[u];//col
	d_vy[i] = d_gridY[v];//row
  }
   
  //d_w.shape   grow*gcol*ctrls
  float *d_w = (float *) malloc(ctrls * grow * gcol * sizeof(float));
  for(int i = 0; i < ctrls; ++i) {
    for(int j = 0; j < grow * gcol; ++j) 
	  d_w[grow*gcol*i+j] = 1.0/(pow(p[2*i]-d_vx[j],2*alpha) + pow(p[2*i+1]-d_vy[j] , 2*alpha));
  }
  //d_pstar.shape grow*gcol*2
  float *d_pstar = (float *) malloc(grow * gcol * 2 * sizeof(float));
  float *d_qstar = (float *) malloc(grow * gcol * 2 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float w_px = 0, w_py = 0, w_tmp = 0;
    float w_qx = 0, w_qy = 0;
	for(int j = 0; j < ctrls; ++j) {
      w_tmp += d_w[j*grow*gcol+i];
	  w_px += (d_w[j*grow*gcol+i] * p[2*j]);
	  w_py += d_w[j*grow*gcol+i] * p[2*j+1];
	  w_qx += d_w[j*grow*gcol+i] * q[2*j];
	  w_qy += d_w[j*grow*gcol+i] * q[2*j+1];
    }
	d_pstar[i] = w_px / w_tmp;
	d_pstar[i + grow*gcol] = w_py / w_tmp;
	d_qstar[i] = w_qx / w_tmp;
	d_qstar[i + grow*gcol] = w_qy / w_tmp;
  }
  //d_phat.shape grow*gcol*2*ctrls
  float *d_phat = (float *) malloc(grow * gcol * 2 * ctrls * sizeof(float));
  float *d_qhat = (float *) malloc(grow * gcol * 2 * ctrls * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    for(int j = 0; j < ctrls; ++j) {
      d_phat[j*grow*gcol*2+i] = p[2*j] - d_pstar[i];
	  d_phat[j*grow*gcol*2+grow*gcol+i] = p[2*j+1] - d_pstar[i+grow*gcol];
      d_qhat[j*grow*gcol*2+i] = q[2*j] - d_qstar[i];
	  d_qhat[j*grow*gcol*2+grow*gcol+i] = q[2*j+1] - d_qstar[i+grow*gcol];
	}
  }
  //pTwp.shape 2*2*grow*gcol
  float *d_pTwp = (float*) malloc(grow * gcol * 4 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
	for(int j = 0; j < ctrls; ++j) {
      tmp1 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+i] * d_w[j*grow*gcol+i]);
      tmp2 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp3 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp4 += (d_phat[j*grow*gcol*2+grow*gcol+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
	}
	d_pTwp[4*i] = tmp1;
	d_pTwp[4*i+1] = tmp2;
	d_pTwp[4*i+2] = tmp3;
	d_pTwp[4*i+3] = tmp4;
  }
  //pTwq.shape 2*2*grow*gcol
  float *d_pTwq = (float*) malloc(grow * gcol * 4 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
	for(int j = 0; j < ctrls; ++j) {
      tmp1 += (d_phat[j*grow*gcol*2+i] * d_qhat[j*grow*gcol*2+i] * d_w[j*grow*gcol+i]);
      tmp2 += (d_phat[j*grow*gcol*2+i] * d_qhat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp3 += (d_qhat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp4 += (d_phat[j*grow*gcol*2+grow*gcol+i] * d_qhat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
	}
	d_pTwq[4*i] = tmp1;
	d_pTwq[4*i+1] = tmp2;
	d_pTwq[4*i+2] = tmp3;
	d_pTwq[4*i+3] = tmp4;
  }
  //d_m.shape 2*2*grow*gcol
  float *d_m = (float*) malloc(grow*gcol*4*sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    Eigen::Matrix2f pwp, pwp_inv, pwq, tmp_m;
    pwp<<d_pTwp[4*i], d_pTwp[4*i+1], d_pTwp[4*i+2], d_pTwp[4*i+3];
	pwp_inv = pwp.inverse();
    pwq<<d_pTwq[4*i], d_pTwq[4*i+1], d_pTwq[4*i+2], d_pTwq[4*i+3];
    tmp_m = pwp_inv * pwq;
    d_m[4*i] = tmp_m(0,0);
    d_m[4*i+1] = tmp_m(0,1);
    d_m[4*i+2] = tmp_m(1,0);
    d_m[4*i+3] = tmp_m(1,1);
  }
  //d_transform.shape 2*grow*gcol
  float *d_transform = (float*) malloc(grow*gcol*2*sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    d_transform[2*i] = (d_vx[i] - d_pstar[i]) * d_m[4*i] + 
	                   (d_vy[i] - d_pstar[i+grow*gcol]) * d_m[4*i+2] + 
					   d_qstar[i];
    d_transform[2*i+1] = (d_vx[i] - d_pstar[i]) * d_m[4*i+1] + 
	                     (d_vy[i] - d_pstar[i+grow*gcol]) * d_m[4*i+3] + 
						 d_qstar[i+grow*gcol];
  }
  for(int i = 0; i < grow*gcol*2; ++i) {
	if(!(d_transform[i] >= 0) || (d_transform[i] > h-1 && i%2==1) || 
	    (d_transform[i] > w-1 && i%2==0))
	  d_transform[i] = 0;
  }

  //new grid
  
  
  for(int i = 0; i < grow*gcol; ++i) {
    int u = int((i%gcol) / density);
    int v = int((i/gcol) / density);
    result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))] = image[3*(v*w+u)]; 
    result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+1] = image[3*(v*w+u)+1]; 
    result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+2] = image[3*(v*w+u)+2]; 
  }

}



void mls_affine_inv(float *image, int h, int w, float *p, float *q, 
                int ctrls, float alpha, float density, float *result)
{
  std::vector<float> gridX(int(w*density));
  std::vector<float> gridY(int(h*density));
  float *d_gridX = gridX.data();
  float *d_gridY = gridY.data();
  //0~w/h interplation according to density
  linspace(0, w, int(w * density), d_gridX);
  linspace(0, h, int(h * density), d_gridY);
  
  //make grids on the original image
  int grow = gridY.size(), gcol = gridX.size();
  float *d_vx = (float*) malloc(grow * gcol * sizeof(float));
  float *d_vy = (float*) malloc(grow * gcol * sizeof(float));
  for(int i = 0; i < gridX.size() * gridY.size(); ++i) {
    int u = i % gridX.size(), v = i / gridX.size();
	d_vx[i] = d_gridX[u];//col
	d_vy[i] = d_gridY[v];//row
  }
   
  //d_w.shape   grow*gcol*ctrls
  float *d_w = (float *) malloc(ctrls * grow * gcol * sizeof(float));
  for(int i = 0; i < ctrls; ++i) {
    for(int j = 0; j < grow * gcol; ++j) 
	  d_w[grow*gcol*i+j] = 1.0/(pow(p[2*i]-d_vx[j],2*alpha) + pow(p[2*i+1]-d_vy[j] , 2*alpha));
  }
  //d_pstar.shape grow*gcol*2
  float *d_pstar = (float *) malloc(grow * gcol * 2 * sizeof(float));
  float *d_qstar = (float *) malloc(grow * gcol * 2 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float w_px = 0, w_py = 0, w_tmp = 0;
    float w_qx = 0, w_qy = 0;
	for(int j = 0; j < ctrls; ++j) {
      w_tmp += d_w[j*grow*gcol+i];
	  w_px += (d_w[j*grow*gcol+i] * p[2*j]);
	  w_py += d_w[j*grow*gcol+i] * p[2*j+1];
	  w_qx += d_w[j*grow*gcol+i] * q[2*j];
	  w_qy += d_w[j*grow*gcol+i] * q[2*j+1];
    }
	d_pstar[i] = w_px / w_tmp;
	d_pstar[i + grow*gcol] = w_py / w_tmp;
	d_qstar[i] = w_qx / w_tmp;
	d_qstar[i + grow*gcol] = w_qy / w_tmp;
  }
  //d_phat.shape grow*gcol*2*ctrls
  float *d_phat = (float *) malloc(grow * gcol * 2 * ctrls * sizeof(float));
  float *d_qhat = (float *) malloc(grow * gcol * 2 * ctrls * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    for(int j = 0; j < ctrls; ++j) {
      d_phat[j*grow*gcol*2+i] = p[2*j] - d_pstar[i];
	  d_phat[j*grow*gcol*2+grow*gcol+i] = p[2*j+1] - d_pstar[i+grow*gcol];
      d_qhat[j*grow*gcol*2+i] = q[2*j] - d_qstar[i];
	  d_qhat[j*grow*gcol*2+grow*gcol+i] = q[2*j+1] - d_qstar[i+grow*gcol];
	}
  }
  //pTwp.shape 2*2*grow*gcol
  float *d_pTwp = (float*) malloc(grow * gcol * 4 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
	for(int j = 0; j < ctrls; ++j) {
      tmp1 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+i] * d_w[j*grow*gcol+i]);
      tmp2 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp3 += (d_phat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp4 += (d_phat[j*grow*gcol*2+grow*gcol+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
	}
	d_pTwp[4*i] = tmp1;
	d_pTwp[4*i+1] = tmp2;
	d_pTwp[4*i+2] = tmp3;
	d_pTwp[4*i+3] = tmp4;
  }
  //pTwq.shape 2*2*grow*gcol
  float *d_pTwq = (float*) malloc(grow * gcol * 4 * sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    float tmp1 = 0, tmp2 = 0, tmp3 = 0, tmp4 = 0;
	for(int j = 0; j < ctrls; ++j) {
      tmp1 += (d_phat[j*grow*gcol*2+i] * d_qhat[j*grow*gcol*2+i] * d_w[j*grow*gcol+i]);
      tmp2 += (d_phat[j*grow*gcol*2+i] * d_qhat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp3 += (d_qhat[j*grow*gcol*2+i] * d_phat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
      tmp4 += (d_phat[j*grow*gcol*2+grow*gcol+i] * d_qhat[j*grow*gcol*2+grow*gcol+i] * d_w[j*grow*gcol+i]);
	}
	d_pTwq[4*i] = tmp1;
	d_pTwq[4*i+1] = tmp2;
	d_pTwq[4*i+2] = tmp3;
	d_pTwq[4*i+3] = tmp4;
  }
  //d_m.shape 2*2*grow*gcol
  float *d_m = (float*) malloc(grow*gcol*4*sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    Eigen::Matrix2f pwq, pwq_inv, pwp, tmp_m;
    pwq<<d_pTwq[4*i], d_pTwq[4*i+1], d_pTwq[4*i+2], d_pTwq[4*i+3];
	pwq_inv = pwq.inverse();
    pwp<<d_pTwp[4*i], d_pTwp[4*i+1], d_pTwp[4*i+2], d_pTwp[4*i+3];
    tmp_m = pwq_inv * pwp;
    d_m[4*i] = tmp_m(0,0);
    d_m[4*i+1] = tmp_m(0,1);
    d_m[4*i+2] = tmp_m(1,0);
    d_m[4*i+3] = tmp_m(1,1);
  }
  //d_transform.shape 2*grow*gcol
  float *d_transform = (float*) malloc(grow*gcol*2*sizeof(float));
  for(int i = 0; i < grow * gcol; ++i) {
    d_transform[2*i] = (d_vx[i] - d_qstar[i]) * d_m[4*i] + 
	                   (d_vy[i] - d_qstar[i+grow*gcol]) * d_m[4*i+2] + 
					   d_pstar[i];
    d_transform[2*i+1] = (d_vx[i] - d_qstar[i]) * d_m[4*i+1] + 
	                     (d_vy[i] - d_qstar[i+grow*gcol]) * d_m[4*i+3] + 
						 d_pstar[i+grow*gcol];
  }
  for(int i = 0; i < grow*gcol*2; ++i) {
	if(!(d_transform[i] >= 0) || (d_transform[i] > h-1 && i%2==1) || 
	    (d_transform[i] > w-1 && i%2==0))
	  d_transform[i] = 0;
  }

  //new grid
  
  
  for(int i = 0; i < grow*gcol; ++i) {
    int u = int((i%gcol) / density);
    int v = int((i/gcol) / density);
    //result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))] = image[3*(v*w+u)]; 
    //result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+1] = image[3*(v*w+u)+1]; 
    //result[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+2] = image[3*(v*w+u)+2]; 
    result[3*(v*w+u)] = image[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))]; 
    result[3*(v*w+u)+1] = image[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+1]; 
    result[3*(v*w+u)+2] = image[3*(int(d_transform[2*i+1])*w+int(d_transform[2*i]))+2]; 
  }

}
