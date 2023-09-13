use std::ops::{Add, Sub, Neg, Mul, Index, IndexMut};
use std::fmt;
use crate::vector::{V1,V2,V3,V4};
use crate::EPSILON;

pub trait MatrixTraits {
    type ColVector;
    type RowVector;
}
macro_rules! matrix_impl {
    ($matrix_name:ident, $col_vector_name:ident, $row_vector_name:ident) => {

        #[derive(Debug, PartialEq, Clone, Copy, Default)]
        pub struct $matrix_name {
            data: [$row_vector_name; $col_vector_name::SIZE],
        }

        impl MatrixTraits for $matrix_name {
            type ColVector = $col_vector_name;
            type RowVector = $row_vector_name;
        }

        impl $matrix_name {
            pub const COLS: usize = $row_vector_name::SIZE;
            pub const ROWS: usize = $col_vector_name::SIZE;

            pub fn new(data: [[f64;Self::COLS];Self::ROWS]) -> Self {
                //recive array of arrays, turn into array of vectors
                let mut matrix_data = [$row_vector_name::default();Self::ROWS];
                for i in 0..Self::ROWS {
                    matrix_data[i] = $row_vector_name::new(data[i]);
                }
                return $matrix_name { data : matrix_data };
            }

            pub fn create_default_col() -> $col_vector_name {
                $col_vector_name::default()
            }

            pub fn create_default_row() -> $row_vector_name {
                $row_vector_name::default()
            }

            pub fn rows(&self) -> usize {
                Self::ROWS
            }

            pub fn cols(&self) -> usize {
                Self::COLS
            }

            pub fn is_square(&self) -> bool {
                Self::ROWS == Self::COLS
            }

            pub fn is_lower_triangular(&self) -> bool {
                for i in 0..self.rows() {
                    for j in i+1..self.cols() {
                        if self[i][j] != 0.0 {
                            return false;
                        }
                    }
                }
                true
            }

            pub fn is_upper_triangular(&self) -> bool {
                for i in 0..self.rows() {
                    for j in 0..i {
                        if self[i][j] != 0.0 {
                            return false;
                        }
                    }
                }
                true
            }

            pub fn swap_rows(&mut self, i: usize, j: usize) {
                //used in gauss jordan elimination
                let temp = self[i];
                self[i] = self[j];
                self[j] = temp;
            }

            pub fn swap_cols(&mut self, i: usize, j: usize) {
                //because why not
                for k in 0..self.rows() {
                    let temp = self[k][i];
                    self[k][i] = self[k][j];
                    self[k][j] = temp;
                }
            }

            pub fn get_col(&self, i: usize) -> $col_vector_name {
                //row major order, so we provide a get column method
                let mut col = $col_vector_name::default();
                for j in 0..self.rows() {
                    col[j] = self[j][i];
                }
                col
            }

            pub fn is_close(&a : &$matrix_name, b: &$matrix_name, epsilon : Option<f64>) -> Result<bool,String> {
                if a.rows() != b.rows() || a.cols() != b.cols() {
                    return Err("Matrices are not the same size.".to_string());
                }
                let epsilon = epsilon.unwrap_or(EPSILON);

                for i in 0..a.rows() {
                    for j in 0..a.cols() {
                        if !((a[i][j] - b[i][j]).abs() < epsilon) {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            }
        }
        
        impl Index<usize> for $matrix_name {
            type Output = $row_vector_name;
            fn index(&self, index: usize) -> &$row_vector_name {
                &self.data[index]
            }
        }

        impl IndexMut<usize> for $matrix_name {
            fn index_mut(&mut self, index: usize) -> &mut $row_vector_name {
                &mut self.data[index]
            }
        }

        impl fmt::Display for $matrix_name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                if self.rows() == 0 {
                    return Ok(());
                } else {
                    for i in 0..self.rows() {
                            write!(f, "{}\n", self.data[i])?;
                    }
                }
                Ok(())
            }
        }

        impl Add for $matrix_name {
            type Output = $matrix_name;
            fn add(self, rhs: $matrix_name) -> $matrix_name {
                let mut res = $matrix_name::default();
                for i in 0..self.cols() {
                    res[i] = self.data[i] + rhs.data[i];
                }
                res
            }
        }

        impl Sub for $matrix_name {
            type Output = $matrix_name;
            fn sub(self, rhs: $matrix_name) -> $matrix_name {
                let mut res = $matrix_name::default();
                for i in 0..self.rows() {
                    res[i] = self.data[i] - rhs.data[i];
                }
                res
            }
        }

        impl Mul<f64> for $matrix_name {
            type Output = $matrix_name;
            fn mul(self, rhs: f64) -> $matrix_name {
                let mut res = $matrix_name::default();
                for i in 0..self.rows() {
                    res[i] = self.data[i] * rhs;
                }
                res
            }
        }

        impl Mul<$matrix_name> for f64 {
            type Output = $matrix_name;
            fn mul(self, rhs: $matrix_name) -> $matrix_name {
                let mut res = $matrix_name::default();
                for i in 0..$matrix_name::ROWS  {
                    res[i] = rhs.data[i] * self;
                }
                res
            }
        }

        impl Neg for $matrix_name {
            type Output = $matrix_name;
            fn neg(self) -> $matrix_name {
                let mut res = $matrix_name::default();
                for i in 0..$matrix_name::ROWS {
                    res[i] = -self.data[i];
                }
                res
            }
        }

        impl Mul<$row_vector_name> for $matrix_name {
            type Output = $col_vector_name;
            fn mul(self, rhs: $row_vector_name) -> $col_vector_name {
                let mut res = $col_vector_name::default();
                for i in 0..$matrix_name::ROWS {
                    for j in 0..$matrix_name::COLS {
                        res[i] += self.data[i][j] * rhs[j];
                    }
                }
                res
            }
        }

        impl Mul<$matrix_name> for $col_vector_name {
            type Output = $row_vector_name;
            fn mul(self, rhs: $matrix_name) -> $row_vector_name {
                let mut res = $row_vector_name::default();
                for j in 0..$matrix_name::COLS {
                    for i in 0..$matrix_name::ROWS {{
                            res[j] += self[i] * rhs.data[i][j];
                        }
                    }
                }
                res
            }
        }
    }
}

matrix_impl!(Matrix11, V1, V1);
matrix_impl!(Matrix12, V1, V2);
matrix_impl!(Matrix13, V1, V3);
matrix_impl!(Matrix14, V1, V4);

matrix_impl!(Matrix21, V2, V1);
matrix_impl!(Matrix22, V2, V2);
matrix_impl!(Matrix23, V2, V3);
matrix_impl!(Matrix24, V2, V4);

matrix_impl!(Matrix31, V3, V1);
matrix_impl!(Matrix32, V3, V2);
matrix_impl!(Matrix33, V3, V3);
matrix_impl!(Matrix34, V3, V4);

matrix_impl!(Matrix41, V4, V1);
matrix_impl!(Matrix42, V4, V2);
matrix_impl!(Matrix43, V4, V3);
matrix_impl!(Matrix44, V4, V4);

macro_rules! matrix_mul_impl {
    ($left_matrix:ident, $right_matrix:ident, $result_matrix:ident) => {
        impl Mul<$right_matrix> for $left_matrix {
            type Output = $result_matrix;

            fn mul(self, rhs: $right_matrix) -> $result_matrix {
                let mut result = $result_matrix::default();
                for i in 0..$left_matrix::ROWS {
                    for j in 0..$right_matrix::COLS {
                        for k in 0..$left_matrix::COLS {
                            result.data[i][j] += self.data[i][k] * rhs.data[k][j];
                        }
                    }
                }
                result
            }
        }
    }
}

matrix_mul_impl!(Matrix11, Matrix11, Matrix11);

matrix_mul_impl!(Matrix12, Matrix21, Matrix11);
matrix_mul_impl!(Matrix12, Matrix22, Matrix12);
matrix_mul_impl!(Matrix12, Matrix23, Matrix13);
matrix_mul_impl!(Matrix12, Matrix24, Matrix14);

matrix_mul_impl!(Matrix13, Matrix31, Matrix11);
matrix_mul_impl!(Matrix13, Matrix32, Matrix12);
matrix_mul_impl!(Matrix13, Matrix33, Matrix13);
matrix_mul_impl!(Matrix13, Matrix34, Matrix14);

matrix_mul_impl!(Matrix14, Matrix41, Matrix11);
matrix_mul_impl!(Matrix14, Matrix42, Matrix12);
matrix_mul_impl!(Matrix14, Matrix43, Matrix13);
matrix_mul_impl!(Matrix14, Matrix44, Matrix14);

matrix_mul_impl!(Matrix21, Matrix11, Matrix21);
matrix_mul_impl!(Matrix21, Matrix12, Matrix22);
matrix_mul_impl!(Matrix21, Matrix13, Matrix23);
matrix_mul_impl!(Matrix21, Matrix14, Matrix24);

matrix_mul_impl!(Matrix22, Matrix21, Matrix21);
matrix_mul_impl!(Matrix22, Matrix22, Matrix22);
matrix_mul_impl!(Matrix22, Matrix23, Matrix23);
matrix_mul_impl!(Matrix22, Matrix24, Matrix24);

matrix_mul_impl!(Matrix23, Matrix31, Matrix21);
matrix_mul_impl!(Matrix23, Matrix32, Matrix22);
matrix_mul_impl!(Matrix23, Matrix33, Matrix23);
matrix_mul_impl!(Matrix23, Matrix34, Matrix24);

matrix_mul_impl!(Matrix24, Matrix41, Matrix21);
matrix_mul_impl!(Matrix24, Matrix42, Matrix22);
matrix_mul_impl!(Matrix24, Matrix43, Matrix23);
matrix_mul_impl!(Matrix24, Matrix44, Matrix24);

matrix_mul_impl!(Matrix31, Matrix11, Matrix31);
matrix_mul_impl!(Matrix31, Matrix12, Matrix32);
matrix_mul_impl!(Matrix31, Matrix13, Matrix33);
matrix_mul_impl!(Matrix31, Matrix14, Matrix34);

matrix_mul_impl!(Matrix32, Matrix21, Matrix31);
matrix_mul_impl!(Matrix32, Matrix22, Matrix32);
matrix_mul_impl!(Matrix32, Matrix23, Matrix33);
matrix_mul_impl!(Matrix32, Matrix24, Matrix34);

matrix_mul_impl!(Matrix33, Matrix31, Matrix31);
matrix_mul_impl!(Matrix33, Matrix32, Matrix32);
matrix_mul_impl!(Matrix33, Matrix33, Matrix33);
matrix_mul_impl!(Matrix33, Matrix34, Matrix34);

matrix_mul_impl!(Matrix34, Matrix41, Matrix31);
matrix_mul_impl!(Matrix34, Matrix42, Matrix32);
matrix_mul_impl!(Matrix34, Matrix43, Matrix33);
matrix_mul_impl!(Matrix34, Matrix44, Matrix34);

matrix_mul_impl!(Matrix41, Matrix11, Matrix41);
matrix_mul_impl!(Matrix41, Matrix12, Matrix42);
matrix_mul_impl!(Matrix41, Matrix13, Matrix43);
matrix_mul_impl!(Matrix41, Matrix14, Matrix44);

matrix_mul_impl!(Matrix42, Matrix21, Matrix41);
matrix_mul_impl!(Matrix42, Matrix22, Matrix42);
matrix_mul_impl!(Matrix42, Matrix23, Matrix43);
matrix_mul_impl!(Matrix42, Matrix24, Matrix44);

matrix_mul_impl!(Matrix43, Matrix31, Matrix41);
matrix_mul_impl!(Matrix43, Matrix32, Matrix42);
matrix_mul_impl!(Matrix43, Matrix33, Matrix43);
matrix_mul_impl!(Matrix43, Matrix34, Matrix44);

matrix_mul_impl!(Matrix44, Matrix41, Matrix41);
matrix_mul_impl!(Matrix44, Matrix42, Matrix42);
matrix_mul_impl!(Matrix44, Matrix43, Matrix43);
matrix_mul_impl!(Matrix44, Matrix44, Matrix44);

macro_rules! matrix_transpose_impl {
    ($matrix_in:ident, $matrix_out:ident) => {
        impl $matrix_in {
            pub fn transpose(self) -> $matrix_out {
                let mut result = $matrix_out::default();
                for i in 0..$matrix_in::ROWS {
                    for j in 0..$matrix_in::COLS {
                        result.data[j][i] = self.data[i][j];
                    }
                }
                result
            }
        }
    }
}

matrix_transpose_impl!(Matrix11, Matrix11);
matrix_transpose_impl!(Matrix12, Matrix21);
matrix_transpose_impl!(Matrix13, Matrix31);
matrix_transpose_impl!(Matrix14, Matrix41);

matrix_transpose_impl!(Matrix21, Matrix12);
matrix_transpose_impl!(Matrix22, Matrix22);
matrix_transpose_impl!(Matrix23, Matrix32);
matrix_transpose_impl!(Matrix24, Matrix42);

matrix_transpose_impl!(Matrix31, Matrix13);
matrix_transpose_impl!(Matrix32, Matrix23);
matrix_transpose_impl!(Matrix33, Matrix33);
matrix_transpose_impl!(Matrix34, Matrix43);

matrix_transpose_impl!(Matrix41, Matrix14);
matrix_transpose_impl!(Matrix42, Matrix24);
matrix_transpose_impl!(Matrix43, Matrix34);
matrix_transpose_impl!(Matrix44, Matrix44);

//Square Matrices
macro_rules! matrix_square_functions_impl {
    ($matrix:ident) => {
        impl $matrix {
            pub fn identity() -> Self {
                let n = $matrix::ROWS;
                let mut result = $matrix::default();
                for i in 0..n {
                    result[i][i] = 1.0;
                }
                result
            }

            pub fn trace(&self) -> f64 {
                let mut sum = 0.0;
                for i in 0..self.rows() {
                    sum += self[i][i];
                }
                sum
            }

            pub fn is_invertible(self, epsilon : Option<f64>) -> bool {
                //if det returns a result and it is not zero
                //det might return error due to numeric issues
                (self.det()).abs() > epsilon.unwrap_or(EPSILON)
            }

            pub fn is_orthogonal(self, epsilon : Option<f64>) -> bool {
                //orthogonal matrix is a square matrix whose transpose is equal to its inverse
                //in other words, A * A^T = I
                Self::is_close(&(self.transpose() * self), &Self::identity(), epsilon).unwrap()
            }

            pub fn lu(self) -> Result<(Self, Self), String> {
                /*
                using Doolittle's method, l has 1's on the diagonal
            
                matrix = L * U
                L = lower triangular matrix
                U = upper triangular matrix

                should really be done with permutation matrix, but I am lazy
                 */            
                let n : usize = Self::ROWS;
                let mut l = Self::default();
                let mut u = Self::default();
            
                for i in 0..n {
                    for k in i..n {
                        let mut sum = 0.0;
                        for j in 0..i {
                            sum += l[i][j] * u[j][k];
                        }
                        u[i][k] = self[i][k] - sum;
                    }
                
                    for k in i..n {
                        if i == k {
                            l[i][i] = 1.0;
                        } else {
                            let mut sum = 0.0;
                            for j in 0..i {
                                sum += l[k][j] * u[j][i];
                            }
                            if u[i][i] == 0.0 {
                                return Err("Decomposition failed due to division by zero.".to_string());
                            }
                            l[k][i] = (self[k][i] - sum) / u[i][i];
                        }
                    }
                }
            
                Ok((l, u))
            }

            pub fn back_substitution(&self, b: <Self as MatrixTraits>::ColVector)
                                           -> Result<<Self as MatrixTraits>::ColVector, String> {
                if !self.is_upper_triangular() {
                    return Err("Matrix is not upper triangular.".to_string());
                }

                let n = Self::ROWS;
                let mut x = Self::create_default_col();
                for i in (0..n).rev() {
                    let mut sum = 0.0;
                    for j in i+1..n {
                        sum += self[i][j] * x[j];
                    }
                    if self[i][i] == 0.0 {
                        return Err("Decomposition failed due to division by zero. Matrix is not invertible".to_string());
                    }
                    x[i] = (b[i] - sum) / self[i][i];
                }
                Ok(x)
            }

            pub fn forward_substitution(&self, b: <Self as MatrixTraits>::ColVector)
                                           -> Result<<Self as MatrixTraits>::ColVector, String> {
                if !self.is_lower_triangular() {
                    return Err("Matrix is not lower triangular.".to_string());
                }

                let n = Self::ROWS;
                let mut x = Self::create_default_col();
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += self[i][j] * x[j];
                    }
                    if self[i][i] == 0.0 {
                        return Err("Decomposition failed due to division by zero. Matrix is not invertible".to_string());
                    }
                    x[i] = (b[i] - sum) / self[i][i];
                }
                Ok(x)
            }

            pub fn invert(&self) -> Result<Self, String> {
                //gauss jordan elimination
                //used here as default and only method, but in reality branching per matrix type should be done
                if !self.is_invertible(None) {
                    return Err("Matrix is not invertible.".to_string());
                }
                let n = Self::ROWS;
                let mut a = self.clone();
                let mut b = Self::identity();
                for i in 0..n {
                    let mut max_row = i;
                    for j in i+1..n {
                        if a[j][i].abs() > a[max_row][i].abs() {
                            max_row = j;
                        }
                    }
                    if max_row != i {
                        a.swap_rows(i, max_row);
                        b.swap_rows(i, max_row);
                    }
                    if a[i][i] == 0.0 {
                        //gptchat placed it here, and I dont mind, so I keep it
                        return Err("Matrix is not invertible.".to_string());
                    }
                    for j in i+1..n {
                        let coef = a[j][i] / a[i][i];
                        for k in 0..n {
                            a[j][k] -= a[i][k] * coef;
                            b[j][k] -= b[i][k] * coef;
                        }
                    }
                }
                for i in (0..n).rev() {
                    for j in 0..i {
                        let coef = a[j][i] / a[i][i];
                        for k in 0..n {
                            a[j][k] -= a[i][k] * coef;
                            b[j][k] -= b[i][k] * coef;
                        }
                    }
                    let coef = a[i][i];
                    for k in 0..n {
                        a[i][k] /= coef;
                        b[i][k] /= coef;
                    }
                }
                Ok(b)
            }

            fn _invert_l(&self) -> Result<Self, String> {
                let n = Self::ROWS;
                let mut l_inv_t = Self::default();

                let eye = Self::identity();
                for i in 0..n {
                    l_inv_t[i] = self.forward_substitution(eye[i])?;
                }
                Ok(l_inv_t.transpose())
            }

            fn _invert_u(&self) -> Result<Self, String> {
                let n = Self::ROWS;
                let mut u_inv_t = Self::default();
            
                let eye = Self::identity();
                for i in 0..n {
                    u_inv_t[i] = self.back_substitution(eye[i])?;
                }
                Ok(u_inv_t.transpose())
            }
        }   
    }
}

matrix_square_functions_impl!(Matrix11);
matrix_square_functions_impl!(Matrix22);
matrix_square_functions_impl!(Matrix33);
matrix_square_functions_impl!(Matrix44);

//implement det

impl Matrix11{
    pub fn det(&self) -> f64 {
        self[0][0]
    }
}

impl Matrix22{
    pub fn det(&self) -> f64 {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }
}

impl Matrix33{
    pub fn det(&self) -> f64 {
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[1][0];
        let e = self[1][1];
        let f = self[1][2];
        let g = self[2][0];
        let h = self[2][1];
        let i = self[2][2];
    
        a * e * i + b * f * g + c * d * h - c * e * g - a * f * h - b * d * i
    }
}

impl Matrix44 {
    pub fn det(&self) -> f64 {
        let a = self[0][0];
        let b = self[0][1];
        let c = self[0][2];
        let d = self[0][3];
        let e = self[1][0];
        let f = self[1][1];
        let g = self[1][2];
        let h = self[1][3];
        let i = self[2][0];
        let j = self[2][1];
        let k = self[2][2];
        let l = self[2][3];
        let m = self[3][0];
        let n = self[3][1];
        let o = self[3][2];
        let p = self[3][3];

        a * f * k * p + a * g * l * n + a * h * j * o - a * h * k * n - a * g * j * p - a * f * l * o
            - b * e * k * p - b * g * l * m - b * h * i * o + b * h * k * m + b * g * i * p + b * e * l * o
            + c * e * j * p + c * f * l * m + c * h * i * n - c * h * j * m - c * f * i * p - c * e * l * n
            - d * e * j * o - d * f * k * m - d * g * i * n + d * g * j * m + d * f * i * o + d * e * k * n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::abs_diff_eq;

    #[test]
    fn test_new(){
        let m = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m, Matrix22 { data: [V2::new([1.0, 2.0]), V2::new([3.0, 4.0])] });
    }

    #[test]
    fn test_vector_creation() {
        assert_eq!(Matrix32::create_default_row(), V2::new([0.0, 0.0]));
        assert_eq!(Matrix32::create_default_col(), V3::new([0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_swaps() {
        let mut m = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        m.swap_rows(0, 1);
        assert_eq!(m, Matrix22::new([[3.0, 4.0], [1.0, 2.0]]));

        let mut m = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        m.swap_cols(0, 1);
        assert_eq!(m, Matrix22::new([[2.0, 1.0], [4.0, 3.0]]));
    }

    #[test]
    fn test_get_col() {
        let m = Matrix32::new([[1.0, 2.0], 
                                              [3.0, 4.0],
                                              [5.0, 6.0]]);
        assert_eq!(m.get_col(0), V3::new([1.0, 3.0, 5.0]));
        assert_eq!(m.get_col(1), V3::new([2.0, 4.0, 6.0]));
    }

    #[test]
    fn test_matrix_traits() {
        assert_eq!(<Matrix32 as MatrixTraits>::RowVector::default(), V2::default());
        assert_eq!(<Matrix32 as MatrixTraits>::ColVector::default(), V3::default());
    }

    #[test]
    fn test_index(){
        let m = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(m[0], V2::new([1.0, 2.0]));
        assert_eq!(m[1], V2::new([3.0, 4.0]));
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[0][1], 2.0);
        assert_eq!(m[1][0], 3.0);
        assert_eq!(m[1][1], 4.0);
    }

    #[test]
    fn test_display() {
        let m = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        let output : String = m.to_string();
        let expect : String = "[1.00, 2.00]\n[3.00, 4.00]\n".to_string();
        assert_eq!(output, expect);
    }

    #[test]
    fn test_default() {
        let m = Matrix22::default();
        for i in 0..m.rows() {
            for j in 0..m.cols() {
                assert_eq!(m[i][j], 0.0);
            }
        }
    }

    #[test]
    fn test_add() {
        let m1 = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix22::new([[5.0, 6.0], [7.0, 8.0]]);
        let m3 = Matrix22::new([[6.0, 8.0], [10.0, 12.0]]);
        assert_eq!(m1 + m2, m3);
        assert_eq!(m1 + m2, m2 + m1);
    }

    #[test]
    fn test_mul_matrix22() {
        let m1 = Matrix22::new([
                                                [1.0, 2.0],
                                                [3.0, 4.0]
                                                ]);
        let m2 = Matrix22::new([
                                                [5.0, 6.0],
                                                [7.0, 8.0]
                                                ]);
        let m3 = Matrix22::new([[19.0, 22.0], [43.0, 50.0]]);
        assert_eq!(m1 * m2, m3);
    }

    #[test]
    fn test_mul_by_scalar() {
        let m1 = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix22::new([[2.0, 4.0], [6.0, 8.0]]);
        assert_eq!(m1 * 2.0, m2);
        assert_eq!(2.0 * m1, m2);
    }

    #[test]
    fn test_neg() {
        let m1 = Matrix22::new([[1.0, 2.0], [3.0, 4.0]]);
        let m2 = Matrix22::new([[-1.0, -2.0], [-3.0, -4.0]]);
        assert_eq!(-m1, m2);
    }

    #[test]
    fn test_mul_vector3() {
        let m = Matrix32::new([[1.0, 2.0], 
                                              [3.0, 4.0],
                                              [5.0, 6.0]]);
        let v = V2::new([1.0, 2.0]);
        let res = V3::new([5.0, 11.0, 17.0]);
        assert_eq!(m * v, res);

        let v = V3::new([1.0, 2.0, 3.0]);
        let res = V2::new([22.0, 28.0]);
        assert_eq!(v * m, res);
    }

    #[test]
    fn test_rows_cols() {
        let m = Matrix32::new([[1.0, 2.0], 
                                              [3.0, 4.0],
                                              [5.0, 6.0]]);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 2);
        assert_eq!(Matrix32::ROWS, 3);
        assert_eq!(Matrix32::COLS, 2);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix32::new([[1.0, 2.0], 
                                              [3.0, 4.0],
                                              [5.0, 6.0]]);
        let m_transpose = Matrix23::new([[1.0, 3.0, 5.0],
                                                        [2.0, 4.0, 6.0]]);
        assert_eq!(m.transpose(), m_transpose);
    }

    #[test]
    fn test_lu() {
        let m = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 5.0, 3.0],
                                              [1.0, 0.0, 8.0]]);

        let (l,u)  = m.lu().unwrap();
        assert!(l.is_lower_triangular());
        assert!(u.is_upper_triangular());
        assert_eq!(l * u,m);
    }

    #[test]
    fn test_det() {
        let m11 = Matrix11::new([[1.0]]);
        abs_diff_eq!(m11.det(), 1.0, epsilon = f64::EPSILON);

        let m22 = Matrix22::new([[1.0, 2.0],
                                                [3.0, 4.0]]);
        abs_diff_eq!(m22.det(), -2.0, epsilon = f64::EPSILON);

        let m33 = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 5.0, 3.0],
                                              [1.0, 0.0, 8.0]]);
        abs_diff_eq!(m33.det(), -1.0, epsilon = f64::EPSILON);

        let m44 = Matrix44::new([[1.0, 2.0, 3.0, 4.0],
                                              [2.0, 5.0, 3.0, 4.0],
                                              [1.0, 0.0, 8.0, 4.0],
                                              [1.0, 3.0, 8.0, 4.0]]);
        abs_diff_eq!(m44.det(), 60.0, epsilon = f64::EPSILON);
    }

    #[test]
    fn test_is_invertible() {
        let m = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 5.0, 3.0],
                                              [1.0, 0.0, 8.0]]);
        assert!(m.is_invertible(None));

        let m = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 4.0, 6.0],
                                              [0.0, 1.0, 0.0]]);
        assert!(!m.is_invertible(None));
    }

    #[test]
    fn test_back_forward_subtitution() {
        let m: Matrix33 = Matrix33::new([[1.0, 2.0, 3.0],
                                                [0.0, 1.0, 4.0],
                                                [0.0, 0.0, 1.0]]);
        let b : V3 = V3::new([1.0, 2.0, 3.0]);
        let x = m.back_substitution(b).unwrap();
        assert_eq!(m * x, b);

        let mt = m.transpose();
        let x = mt.forward_substitution(b).unwrap();
        assert_eq!(mt * x, b);
    }

    #[test]
    fn test_invert_u_l() {
        let m = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 5.0, 3.0],
                                              [1.0, 0.0, 8.0]]);
        let (l,u) = m.lu().unwrap();

        let l_inv = l._invert_l().unwrap();
        let u_inv = u._invert_u().unwrap();

        let eye = Matrix33::identity();
        assert!(Matrix33::is_close(&(l_inv * l), &eye, None).unwrap());
        assert!(Matrix33::is_close(&(u_inv * u), &eye, None).unwrap());

        let m_inv = u_inv * l_inv;
        assert!(Matrix33::is_close(&(m_inv * m), &eye, None).unwrap());
    }

    #[test]
    fn test_invert() {
        let m = Matrix33::new([[1.0, 2.0, 3.0],
                                              [2.0, 5.0, 3.0],
                                              [1.0, 0.0, 8.0]]);
        let m_expect = Matrix33::new([[-40.0, 16.0, 9.0],
                                                   [13.0, -5.0, -3.0],
                                                   [5.0, -2.0, -1.0]]);

        let m_output = m.invert().unwrap();

        assert!(Matrix33::is_close(&m_expect, &m_output, Some(EPSILON)).unwrap())
    }

    #[test]
    fn test_orthogonal() {
        let mut m: Matrix33 = Matrix33::new([[0.0, 1.0, 0.0],
                                                [0.0, 0.0, 1.0],
                                                [1.0, 0.0, 0.0]]);
        assert!(m.is_orthogonal(None));

        m[0][0] = 2.0;
        assert!(!m.is_orthogonal(None));
    }
}
