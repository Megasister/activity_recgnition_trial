#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "function.h"

//窗口长度 = 频率（52/s）*窗口长度（5s）
#define window_len 260
//六个轴
#define axis_no 6
//一个窗口内总特征个数
#define feature_no 66
//一个轴提取到的特征个数
#define axis_feature_no 11

//缓存区存储读入的数据，满则清零
static float buffer[axis_no][window_len];
//记录读入数据的个数，缓存满则清零
static int buffer_ord = 0;

void main() {
	init_buffer();
	for (int i = 0; i < window_len; i++) {
		printf("this is buffer_ord %d\n", buffer_ord);
		int single_res = putnum(0, 1, 2, 3, 4, 5);
		if (single_res == -1) continue;
		else printf("current result is : %d\n", single_res);
		/*if (i == (2 * window_len - 4)) {
			print_arr(buffer, axis_no);
		}*/
	}
}

//打印buffer
void print_arr(float arr[][window_len], int m) {
	for (int j = 0; j < window_len; j++) {
		for (int k = 0; k < m; k++) {
			printf("buffer at %d, %d: %f\n", k, j, arr[k][j]);
		}
	}
}

/*将一个数据点（6个数据）放入buffer内，
*如果buffer满了就计算分类并返回分类，
如果buffer没有满则返回-1*/
int putnum(float ax, float ay, float az, float wx, float wy, float wz) {
	buffer[0][buffer_ord] = ax;
	buffer[1][buffer_ord] = ay;
	buffer[2][buffer_ord] = az;
	buffer[3][buffer_ord] = wx;
	buffer[4][buffer_ord] = wy;
	buffer[5][buffer_ord] = wz;

	if (buffer_ord == window_len - 1){
		int single_res = classify(buffer);
		buffer_ord = 0;
		init_buffer();
		return single_res;
	}
	buffer_ord += 1;
	return -1;
}

struct featureWrap {
	float arr[feature_no];
};
struct axisWrap {
	float arr[window_len];
};

int classify() {
	/*return the single classification result for current buffer
	* numerized as 0-5
	* step1: get all features from 6 axis by calling get_feature_by_axis()
	* step2:convert struct to array
	* step3: predict the result using current features extracted
	* return:  number 0-5 representing jog, others, rope skipping, sit up and walk
	*/
	float ft_vec[feature_no];
	struct axisWrap a_wrap = get_feature_by_axis();
	for (int i = 0; i < feature_no; i++) {
		ft_vec[i] = a_wrap.arr[i];
		printf("feature_vector at index %d is %f\n", i, ft_vec[i]);
	}
	int res = predict(ft_vec);
}
struct axisWrap get_feature_by_axis() {
	/*
	* use struct to construct content of 2D array during func call
	* input: current loaded buffer
	* return a struct containing all features from 6 axis
	*/
	float container [feature_no];
	for (int i = 0; i < feature_no; i++) {
		container[i] = 0;
	}
	struct axisWrap a_wrap;
	int k = 0;
	while (k < feature_no) {
		for (int i = 0; i < axis_no; i++) {
			printf("this is axis: %d", i);
			// contain data vec in one axis
			float vec[window_len];
			for (int j = 0; j < window_len; j++) {
				vec[j] = buffer[i][j];
				//printf("vec [%d] is %f\n", j, vec[j]);
			}
			// get feature for each axis
			struct featureWrap fwrap = get_feature(vec);
			for (int m = 0; m < axis_feature_no; m++) {
				// concatenate all features
				a_wrap.arr[k] = fwrap.arr[m];
				k++;
			}
		}
	}
	/*for (int q = 0; q < feature_no; q++) {
		printf("awarp at %d: %f\n", q, a_wrap.arr[q]);
	}*/
	return a_wrap;
}
struct featureWrap get_feature(float vec[window_len]) {
	/*input: data input on each axis
	return: struct containing features from each axis*/
	struct featureWrap wrap;
	float std_val1 = std(vec);
	float std_val2 = std(vec);
	float std_val3 = std(vec);
	float std_val4 = std(vec);
	float var_val1 = var(vec);
	float var_val2 = var(vec);
	float var_val3 = var(vec);
	float var_val4 = var(vec);
	float mean_val1 = mean(vec);
	float mean_val2 = mean(vec);
	float mean_val3 = mean(vec);
	float temp_list [axis_feature_no] = { std_val1, std_val2, std_val3, std_val4, var_val1, var_val2, var_val3, var_val4, mean_val1, mean_val2, mean_val3 };
	for (int i = 0; i < axis_feature_no; i++) {
		wrap.arr[i] = temp_list[i];
	}
	return wrap;
}

//initialize the buffer with all zero
void init_buffer()
{
	for (int i = 0; i < axis_no; i++) {
		for (int j = 0; j < window_len; j++) {
			buffer[i][j] = 0;
		}
	}
}
//find mean of a number array
float mean(float data_stream []) {
	float sum = 0;
	int len = sizeof(data_stream);
	int i;
	for (i = 0; i < len; i++) {
		sum += data_stream[i];
	}
	return (sum/len);
}
//find standard diviation of a number array
float std(float data_stream[]) {
	float sum = 0;
	int len = sizeof(data_stream);
	float m = mean(data_stream);
	for (int i = 0; i < len; i++) {
		sum += pow((data_stream[i] - m), 2);
		//sum += pow((data_stream[i] - m), 2);
	}
	return sqrt(sum);
}
//find variance of a number array
float var(float data_stream[]) {
	float sum = 0;
	int len = sizeof(data_stream);
	float m = mean(data_stream);
	for (int i = 0; i < len; i++) {
		sum += pow(((double)data_stream[i] - m), 2);
		//sum += pow((data_stream[i] - m), 2);
	}
	return sum;
}
//find max of a number array
float find_max(float data_stream[]) {
	int len = sizeof(data_stream);
	float max = -100;
	for (int i = 0; i < len; i++) {
		if (data_stream[i] > max)
			max = data_stream[i];
	}
	return max;
}
//find min of a number array
float find_min(float data_stream[]) {
	int len = sizeof(data_stream);
	float min = +100;
	for (int i = 0; i < len; i++) {
		if (data_stream[i] < min)
			min = data_stream[i];
	}
	return min;
}
//dump useless test code here
void litter() {
	float input[5] = {2.3, 1.2, 4.6, 7.6, 9.0};
	float m = mean(input);
	float s = std(input);
	float v = var(input);
	float max = find_max(input);
	float min = find_min(input);
	printf("this is mean: %f\n", m);
	printf("this is std : %f\n", s);
	printf("this is var : %f\n", v);
	printf("this is max : %f\n", max);
	printf("this is min : %f\n", min);
}


