    // #include <fstream>
    // #include <iostream>
    // #include <sstream>
    // #include <string>
    // #include <vector>
    // #include <typeinfo>


    // using namespace std;
    // int main()
    // {
    //     ifstream in("/home/ander/Documents/git/clustering/scripts/config.csv");
    //     vector<vector<double>> fields;
    //     double r_x, r_z, var_x1, var_x2;
        
    //     if (in) {
    //         string line;
    //         while (getline(in, line)) {
    //             stringstream sep(line);
    //             string field;
    //             string field_name;

    //             fields.push_back(vector<double>());
    //             unsigned int counter = 0;
    //             while (getline(sep, field, ',')) {
    //                 fields.back().push_back(stod(field));
    //             }
    //         }
    //     }
    //     for (auto row : fields) {
    //         for (auto field : row) {
    //             cout << field << ' ';
    //         }
    //         cout << '\n';
    //     }
    // }