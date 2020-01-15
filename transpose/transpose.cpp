/* transpose.cpp

Write a function to transpose a 16 pixels by 16 pixels image block,
where each pixel is represented by 1 byte. Write 2 versions of the
function: one in C and one in assembly that using x86 SIMD
instructions.

*/

#include "sse.h"

#include <iostream>
#include <iomanip>

#define MB_SIZE 16

using MB = unsigned char[][MB_SIZE];

// Transpose the values in a 16x16 macroblock using SSE SIMD instructions
inline void transpose_16x16(
	__m128i&  x0, __m128i&  x1, __m128i&  x2, __m128i&  x3,
	__m128i&  x4, __m128i&  x5, __m128i&  x6, __m128i&  x7,
	__m128i&  x8, __m128i&  x9, __m128i& x10, __m128i& x11,
	__m128i& x12, __m128i& x13, __m128i& x14, __m128i& x15)
{
	__m128i w00, w01, w02, w03;
	__m128i w10, w11, w12, w13;
	__m128i w20, w21, w22, w23;
	__m128i w30, w31, w32, w33;

	transpose_4x4_dwords(x0, x1, x2, x3, w00, w01, w02, w03);
	transpose_4x4_dwords(x4, x5, x6, x7, w10, w11, w12, w13);
	transpose_4x4_dwords(x8, x9, x10, x11, w20, w21, w22, w23);
	transpose_4x4_dwords(x12, x13, x14, x15, w30, w31, w32, w33);
	w00 = transpose_4x4(w00);
	w01 = transpose_4x4(w01);
	w02 = transpose_4x4(w02);
	w03 = transpose_4x4(w03);
	w10 = transpose_4x4(w10);
	w11 = transpose_4x4(w11);
	w12 = transpose_4x4(w12);
	w13 = transpose_4x4(w13);
	w20 = transpose_4x4(w20);
	w21 = transpose_4x4(w21);
	w22 = transpose_4x4(w22);
	w23 = transpose_4x4(w23);
	w30 = transpose_4x4(w30);
	w31 = transpose_4x4(w31);
	w32 = transpose_4x4(w32);
	w33 = transpose_4x4(w33);
	transpose_4x4_dwords(w00, w10, w20, w30, x0, x1, x2, x3);
	transpose_4x4_dwords(w01, w11, w21, w31, x4, x5, x6, x7);
	transpose_4x4_dwords(w02, w12, w22, w32, x8, x9, x10, x11);
	transpose_4x4_dwords(w03, w13, w23, w33, x12, x13, x14, x15);
}

// Transpose the values in a 16x16 macroblock using C++
void transpose(MB mb) {
	for (auto r = 1; r < MB_SIZE; r++) {
		for (auto c = 0; c < r; c++) {
			const unsigned char tmp = mb[r][c];
			mb[r][c] = mb[c][r];
			mb[c][r] = tmp;
		}
	}
}

// Print the byte values in macroblock
void print(MB mb) {
	for (auto i = 0; i < MB_SIZE; i++) {
		for (auto j = 0; j < MB_SIZE; j++) {
			std::cout << std::setw(3) << (int)mb[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

// Fill byte values in 16x16 macroblock with values that will make it easy to see if 'transpose()' worked
void fill(MB mb) {
	unsigned char val = 0;
	for (unsigned char i = 0; i < MB_SIZE; i++)
		for (unsigned char j = 0; j < MB_SIZE; j++)
			mb[i][j] = val++;
}

int main(void) {
	unsigned char mb[MB_SIZE][MB_SIZE];

	std::cout << "Using C++\n";

	fill(mb);
	print(mb);

	transpose(mb);
	print(mb);

	std::cout << "Using SSE instructions\n";
	transpose(mb);	// get back to original order
	print(mb);

	// pack each row
	__m128i  x0 = _128i_load(&mb[0][0]);
	__m128i  x1 = _128i_load(&mb[1][0]);
	__m128i  x2 = _128i_load(&mb[2][0]);
	__m128i  x3 = _128i_load(&mb[3][0]);
	__m128i  x4 = _128i_load(&mb[4][0]);
	__m128i  x5 = _128i_load(&mb[5][0]);
	__m128i  x6 = _128i_load(&mb[6][0]);
	__m128i  x7 = _128i_load(&mb[7][0]);
	__m128i  x8 = _128i_load(&mb[8][0]);
	__m128i  x9 = _128i_load(&mb[9][0]);
	__m128i  xA = _128i_load(&mb[10][0]);
	__m128i  xB = _128i_load(&mb[11][0]);
	__m128i  xC = _128i_load(&mb[12][0]);
	__m128i  xD = _128i_load(&mb[13][0]);
	__m128i  xE = _128i_load(&mb[14][0]);
	__m128i  xF = _128i_load(&mb[15][0]);

	transpose_16x16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, xA, xB, xC, xD, xE, xF);

	// unpack each row back into the MB
	_128i_store(&mb[0][0], x0);
	_128i_store(&mb[1][0], x1);
	_128i_store(&mb[2][0], x2);
	_128i_store(&mb[3][0], x3);
	_128i_store(&mb[4][0], x4);
	_128i_store(&mb[5][0], x5);
	_128i_store(&mb[6][0], x6);
	_128i_store(&mb[7][0], x7);
	_128i_store(&mb[8][0], x8);
	_128i_store(&mb[9][0], x9);
	_128i_store(&mb[10][0], xA);
	_128i_store(&mb[11][0], xB);
	_128i_store(&mb[12][0], xC);
	_128i_store(&mb[13][0], xD);
	_128i_store(&mb[14][0], xE);
	_128i_store(&mb[15][0], xF);

	print(mb);

	return 0;
}
