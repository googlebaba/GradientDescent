#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
using namespace std;

typedef vector<pair<vector<double>,double> > vec_pair_vec;
typedef vector<vector<double> > vec_vec;
typedef vector<vector<string> > vec_vec_str;
ifstream& open_file(ifstream& in,string filename)   //���ļ�
{
	in.close();
	in.clear();
	in.open(filename.c_str());
	return in;
}

double guiyihua(double x,double mean,double max,double min){      //�����ݹ�һ����-1��1֮��
	double result = (x-mean)/(max-min);
	return result;
}
//�Ѷ������������ݼ�
vec_vec example_to_vec(ifstream& in){   
	string str;
	vec_vec temp_vec;
	while(getline(in,str)){       //һ��һ�ж�������
		stringstream ss(str.c_str());
		string value;
		vector<double> d_vec;
		while(ss>>value){
			double d_value = atof(value.c_str());
			d_vec.push_back(d_value);
		}
		temp_vec.push_back(d_vec);
	}
	return temp_vec; 
}

//���ֵ�ĺ���
double Mean(vector<double> vec){
	vector<double>::iterator iter = vec.begin();
	vector<double>::size_type sz = vec.size();
	double sum = 0.0;
	for(;iter!=vec.end();++iter){
		sum +=*iter;
	}
	double result = sum/sz;
	return result;
}

//�����ֵ�ĺ���
double Max(vector<double> vec){
	double max = vec[0];
	vector<double>::iterator iter = vec.begin();
	for (;iter!=vec.end();++iter)
	{
		if(*iter>max)
			max = *iter;
	}
	return max;
}

//����Сֵ�ĺ���
double Min(vector<double> vec){
	double min = vec[0];
	vector<double>::iterator iter = vec.begin();
	for (;iter!=vec.end();++iter)
	{
		if(*iter<min)
			min = *iter;
	}
	return min;
}

//���ź���
void sgn(double& x){
	if(x>=0.0)
		x = 1;
	else
		x = -1;
}
//�������ݼ�,�����ݴ������Ҫ�ĸ�ʽ
vec_pair_vec training_example(vec_vec vec_example){
	vec_vec::iterator iter1 = vec_example.begin();
	const int  num_col = iter1->size();//����Ϊһ�������ж��ٸ�����
	const int  num_row = vec_example.size(); //����Ϊÿ��������ֵ
	size_t i = 0;
	vector<double> tt(num_row,0.0);
	vec_vec temp_vec(num_col,tt);          //���洦��������
	for(;iter1!=vec_example.end();++iter1,++i){
		vector<double>::iterator iter2 = iter1->begin();
		for(size_t j = 0;iter2!=iter1->end();++iter2,++j){
			temp_vec[j][i] = *iter2;                    //���������з�ת
		}
	}
	vec_vec mean_max_min_vec;    //���������Сֵƽ��ֵ
	vec_vec::iterator iter3 = temp_vec.begin();
	for(;iter3!=temp_vec.end();++iter3){
		vector<double> d_vec;
		double mean = Mean(*iter3);   //�õ���һ�е�ƽ����
		double max = Max(*iter3);    //�õ���һ�е����ֵ
		double min = Min(*iter3);  //�õ���һ�е���Сֵ
		d_vec.push_back(mean);
		d_vec.push_back(max);
		d_vec.push_back(min);
		mean_max_min_vec.push_back(d_vec);
	}
	vec_pair_vec res_vec;     //���ķ��ؽ��
	//�������ֵ�ù�һ������
	for(size_t i = 0;i!=vec_example.size();++i){
		for(size_t j = 0;j!=num_col;++j){
			vec_example[i][j] = guiyihua(vec_example[i][j],mean_max_min_vec[j][0],mean_max_min_vec[j][1],mean_max_min_vec[j][2]);
			if(j == num_col -1)
			{
				double temp = vec_example[i][j]; //�������һ��ֵ
				sgn(temp);
				vector<double> tem_vec(vec_example[i]);
				tem_vec.erase(tem_vec.end()-1);    //ɾ�����һ��Ԫ��
				res_vec.push_back(make_pair(tem_vec,temp));
			}
		}
	}
	//�����һ��������	
	return res_vec;
}

//�������ֵ
double out_put(vector<double> input,vector<double>w_vec)
{
	double result = 0.0;
	for(size_t ix = 0;ix!=input.size();++ix)
	{
		result += input[ix]*w_vec[ix];
	}
	sgn(result);
	return result;
}

//�ݶ��½��㷨��������
vector<double> Gradient_Descent(vec_pair_vec tra_exp,double n)
{
	srand( (unsigned)time( NULL ) );
	vector<double>w;   //���ڴ洢wi;
	size_t sz = tra_exp.begin()->first.size();
	for(size_t ix = 0;ix!=sz;++ix)           //��ʼ��wiΪĳ��С�����ֵ
	{
		double iy = rand()/double(RAND_MAX);
		w.push_back(iy);
	}
	vector<double>deta_w(sz,0.0);              //��ʼ��deta_wΪ0
	vec_pair_vec::iterator iter = tra_exp.begin();
	for(;iter!=tra_exp.end();++iter)      //����ÿ������
	{
		int out = out_put(iter->first,w);    //��ʵ��x���뵽�˵�Ԫ�������
		for(size_t ix = 0;ix!=sz;++ix){
			deta_w[ix] += n*(iter->second - out)*iter->first[ix];
			w[ix] += deta_w[ix];
		}
	}
	return w;
}
int main(){
	ifstream infile;
	string filename("trainingset2.txt");
	open_file(infile,filename);
	if(!infile)
	{
		cerr<<"wrong filename!"<<endl;
		return -1;
	}
	vec_vec tmp_vec = example_to_vec(infile);
	vec_pair_vec example = training_example(tmp_vec);
	double n = 0.05;
	vector<double> w = Gradient_Descent(example,n);
	return 0;
}