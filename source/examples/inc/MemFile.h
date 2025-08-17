#ifndef MEM_FILE_H
#define MEM_FILE_H

#include <fstream>
#include <string>
#include <cassert>
// #include "DataType.h"

template<typename T>
class MemGuard {
	int m_size;
	T* data;

public:

	MemGuard() :data(nullptr), m_size(0) {}
	MemGuard(int sz) :m_size(sz) {
		data = (T*)malloc(sizeof(T)*sz);
		assert(data != nullptr);
		memset(data, 0, sizeof(T)*sz); // 初始化
	}
	MemGuard(MemGuard&& rhs) :data(rhs.data), m_size(rhs.m_size) {
		rhs.data = nullptr;
		rhs.m_size = 0;
	}
	MemGuard& operator=(MemGuard&& rhs) {
		free(data);
		data = rhs.data;
		m_size = rhs.m_size;
		rhs.data = nullptr;
		rhs.m_size = 0;
		return *this;
	}
	MemGuard(const MemGuard&) = delete; // TODO: 该不该拥有复制构造函数？
	MemGuard& operator=(const MemGuard&) = delete;
	~MemGuard() {
		free(data);
	}
	int size() {
		return m_size;
	}
	T* get() {
		return data;
	}
	T*& ref() {
		return data;
	}
	T& operator[](int index) {
		return data[index];
	}
	void resize(int sz) {
		free(data);
		data = (T*)malloc(sizeof(T)*sz);
		assert(data != nullptr);
		memset(data, 0, sizeof(T)*sz); // 初始化
	}
};

class MemFile {
public:
	template<typename T>
	struct Complex {
		T x, y;
	};
	template<typename T>
	static bool readBin(std::string fileName, T*& data, int* nx, int *ny) {
		std::ifstream ifs(fileName, std::ios::binary);
		if (!ifs) return false;
		ifs.read((char*)nx, 4);
		ifs.read((char*)ny, 4);
		data = new T[(*nx)*(*ny)];
		ifs.read((char*)data, sizeof(T)*(*nx)*(*ny));
		ifs.close();
		return true;
	}
#ifdef __VECTOR_TYPES_H__
	static bool readBin(const std::string& real_file_name, const std::string& imag_file_name, double2*& data, int* nx, int *ny) {
		assert(data == nullptr);
		MemGuard<double> real, imag;
		readBin(real_file_name, imag_file_name, real.ref(), imag.ref(), nx, ny);
		data = (double2*)malloc((*nx)*(*ny)*sizeof(double2));
		assert(data != nullptr);
		for (int i = 0; i < (*nx)*(*ny); i++) {
			data[i].x = real[i];
			data[i].y = imag[i];
		}
		return true;
	}
#endif

	// real和imag必须是nullptr
	template<typename T>
	static bool readBin(const std::string& fileName, T*& real, T*& imag, int* nx, int *ny) {
		assert(real == nullptr && imag == nullptr);
		MemGuard<Complex<T>> data;
		bool flag = readBin(fileName, data.ref(), nx, ny);
		if (flag == false) return false;
		real = new T[(*nx)*(*ny)];
		imag = new T[(*nx)*(*ny)];
		for (int i = 0; i < (*nx) * (*ny); i++) {
			real[i] = data[i].x;
			imag[i] = data[i].y;
		}
		return true;
	}
	static bool readBin(const std::string& real_file_name, const std::string& imag_file_name, double*& real, double*& imag, int* nx, int *ny) {
		assert(real == nullptr && imag == nullptr);
		readBin(real_file_name, real, nx, ny);
		int nx1, ny1;
		readBin(imag_file_name, imag, &nx1, &ny1);
		assert(*nx == nx1 && *ny == ny1);
		return true;
	}

	template<typename T>
	static bool writeBin(const std::string& file_name, T* data, int nx, int ny) {
		std::ofstream ofs(file_name, std::ios::binary);
		if (!ofs) return false;
		ofs.write((char*)&nx, sizeof(int));
		ofs.write((char*)&ny, sizeof(int));  // 前8个字节记录nx和ny
		ofs.write((char*)data, sizeof(T)*nx*ny);
		ofs.close();
		return true;
	}
	template<typename T>
	static bool writeBin(const std::string& file_name, T* real, T* imag, int nx, int ny) {
		MemGuard<Complex<T>> comp(nx*ny);
		for (int i = 0; i < nx*ny; i++)
			comp[i] = { real[i], imag[i] };
		std::ofstream ofs(file_name, std::ios::binary);
		if (!ofs) return false;
		ofs.write((char*)&nx, sizeof(int));
		ofs.write((char*)&ny, sizeof(int));  // 前8个字节记录nx和ny
		ofs.write((char*)comp.get(), sizeof(Complex<T>)*nx*ny);
		ofs.close();
		return true;
	}
};

#endif