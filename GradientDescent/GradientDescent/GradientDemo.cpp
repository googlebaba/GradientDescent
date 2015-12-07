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
ifstream& open_file(ifstream& in,string filename)   //打开文件
{
	in.close();
	in.clear();
	in.open(filename.c_str());
	return in;
}

double guiyihua(double x,double mean,double max,double min){      //把数据归一化到-1到1之间
	double result = (x-mean)/(max-min);
	return result;
}
//把读入的流变成数据集
vec_vec example_to_vec(ifstream& in){   
	string str;
	vec_vec temp_vec;
	while(getline(in,str)){       //一行一行读入数据
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

//求均值的函数
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

//求最大值的函数
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

//求最小值的函数
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

//符号函数
void sgn(double& x){
	if(x>=0.0)
		x = 1;
	else
		x = -1;
}
//处理数据集,把数据处理成想要的格式
vec_pair_vec training_example(vec_vec vec_example){
	vec_vec::iterator iter1 = vec_example.begin();
	const int  num_col = iter1->size();//行数为一组数据有多少个输入
	const int  num_row = vec_example.size(); //列数为每组样例的值
	size_t i = 0;
	vector<double> tt(num_row,0.0);
	vec_vec temp_vec(num_col,tt);          //保存处理后的数据
	for(;iter1!=vec_example.end();++iter1,++i){
		vector<double>::iterator iter2 = iter1->begin();
		for(size_t j = 0;iter2!=iter1->end();++iter2,++j){
			temp_vec[j][i] = *iter2;                    //把数据行列翻转
		}
	}
	vec_vec mean_max_min_vec;    //储存最大最小值平均值
	vec_vec::iterator iter3 = temp_vec.begin();
	for(;iter3!=temp_vec.end();++iter3){
		vector<double> d_vec;
		double mean = Mean(*iter3);   //得到这一行的平均数
		double max = Max(*iter3);    //得到这一行的最大值
		double min = Min(*iter3);  //得到这一行的最小值
		d_vec.push_back(mean);
		d_vec.push_back(max);
		d_vec.push_back(min);
		mean_max_min_vec.push_back(d_vec);
	}
	vec_pair_vec res_vec;     //最后的返回结果
	//下面进行值得归一化操作
	for(size_t i = 0;i!=vec_example.size();++i){
		for(size_t j = 0;j!=num_col;++j){
			vec_example[i][j] = guiyihua(vec_example[i][j],mean_max_min_vec[j][0],mean_max_min_vec[j][1],mean_max_min_vec[j][2]);
			if(j == num_col -1)
			{
				double temp = vec_example[i][j]; //保存最后一个值
				sgn(temp);
				vector<double> tem_vec(vec_example[i]);
				tem_vec.erase(tem_vec.end()-1);    //删除最后一个元素
				res_vec.push_back(make_pair(tem_vec,temp));
			}
		}
	}
	//把最后一项分离出来	
	return res_vec;
}

//计算输出值
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

//梯度下降算法的主函数
vector<double> Gradient_Descent(vec_pair_vec tra_exp,double n)
{
	srand( (unsigned)time( NULL ) );
	vector<double>w;   //用于存储wi;
	size_t sz = tra_exp.begin()->first.size();
	for(size_t ix = 0;ix!=sz;++ix)           //初始化wi为某个小的随机值
	{
		double iy = rand()/double(RAND_MAX);
		w.push_back(iy);
	}
	vector<double>deta_w(sz,0.0);              //初始化deta_w为0
	vec_pair_vec::iterator iter = tra_exp.begin();
	for(;iter!=tra_exp.end();++iter)      //遍历每条输入
	{
		int out = out_put(iter->first,w);    //把实例x输入到此单元计算输出
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