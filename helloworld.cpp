#include <iostream>
#include <cmath>
#include <xtensor/xio.hpp>
using namespace std;
using namespace xt;
#include <iomanip>  // For std::setprecision, std::setw, etc.
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xarray.hpp>

int main() {
  cout << "Outputs begin here\n";
  float x = 5;
  float y =x;
  cout << "One word" << " One \t another \n";
  // string sentence
  cout << "Type a sentence bro";
  // cin >>  sentence;
  // cout << sentence << "\t Here we go";
  
  bool choice;
  cout << "Enter 1 if u want to break the loop before the end";
  // cin >> choice;
  
  for (int i=0; i<10; i++) {
    string initia = "The nb is...";
    string ending = "... wait for it...";
    cout << initia.append(ending.append("ERROR HERE")) << i;
    cout << endl;
    if (choice == 1 and i == 5) {
      break;
    };
  };

  int counter = 7;
  while (counter < 10) {
    cout << counter +100;
    cout << endl;
    counter ++;
  }

  float modulopi[10];
  for (int k = 0; k < 10; k++) {
    modulopi[k] = k * M_PI;
  }
  cout << "\n Pi here \n";
  cout << "\nFormatted Pi values:\n";
  for (int k = 0; k < 10; k++) {
      cout << "Index: " << k << " | Value: " << modulopi[k] << endl;
  }

  struct OneStruct {
    int evil;
    string itsme;
    float near_zero;
  };

  OneStruct cursed_struct;

  cursed_struct.evil = 666;
  cursed_struct.itsme = "\nhey its me";
  cursed_struct.near_zero = 0.0001;

  cout << "\n" << cursed_struct.evil << endl;

  OneStruct clean_struct;
  clean_struct.evil = 555;

  cout << clean_struct.evil << endl;

  xarray<double> a = xarray<double>({1, 2});
  xarray<double> b = xarray<double>({7, 3});
  array d = {1, 2, 9};
  auto c = a * b;
  cout << c;

  std::tuple my_tuple = std::tuple<int, int, xarray<double>>{1, 2, a};
  cout << "The tuple elements " << get<0>(my_tuple) << endl << get<2>(my_tuple); 


  return 0;
}