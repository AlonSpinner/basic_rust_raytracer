use std::ops::{Add, Sub, Neg, Mul, Index, IndexMut};
use std::fmt;
use crate::EPSILON;

macro_rules! vector_impl {
    ($name:ident, $size:expr) => {
        #[derive(Debug, PartialEq, Clone, Copy, Default)]
        pub struct $name {
            data: [f64; $size],
        }

        impl $name {
            pub const SIZE : usize = $size;

            pub fn new(data: [f64; $size]) -> Self {
                $name { data }
            }
        }

        impl Add for $name {
            type Output = $name;
            fn add(self, rhs: $name) -> $name {
                let mut res = [0.0; $size];
                for i in 0..$size {
                    res[i] = self.data[i] + rhs.data[i];
                }
                $name::new(res)
            }
        }

        impl Sub for $name {
            type Output = $name;
            fn sub(self, rhs: $name) -> $name {
                let mut res = [0.0; $size];
                for i in 0..$size {
                    res[i] = self.data[i] - rhs.data[i];
                }
                $name::new(res)
            }
        }

        impl Neg for $name {
            type Output = $name;
            fn neg(self) -> $name {
                let mut res = [0.0; $size];
                for i in 0..$size {
                    res[i] = -self.data[i];
                }
                $name::new(res)
            }
        }

        impl Mul<f64> for $name {
            type Output = $name;
            fn mul(self, rhs: f64) -> $name {
                let mut res = [0.0; $size];
                for i in 0..$size {
                    res[i] = self.data[i] * rhs;
                }
                $name::new(res)
            }
        }

        impl Mul<$name> for f64 {
            type Output = $name;
            fn mul(self, rhs: $name) -> $name {
                let mut res = [0.0; $size];
                for i in 0..$size {
                    res[i] = rhs.data[i] * self;
                }
                $name::new(res)
            }
        }

        impl Index<usize> for $name {
            type Output = f64;
            fn index(&self, index: usize) -> &f64 {
                &self.data[index]
            }
        }

        impl IndexMut<usize> for $name {
            fn index_mut(&mut self, index: usize) -> &mut f64 {
                &mut self.data[index]
            }
        }

        //implement dot product
        impl Mul for $name {
            type Output = f64;
            fn mul(self, rhs: $name) -> f64 {
                let mut res = 0.0;
                for i in 0..$size {
                    res += self.data[i] * rhs.data[i];
                }
                res
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "[")?;
                for i in 0..$size {
                    if i == $size - 1 {
                        write!(f, "{:.2}", self.data[i])?;
                    } else {
                        write!(f, "{:.2}, ", self.data[i])?;
                    }
                }
                write!(f, "]")
            }
        }

        //implement normalize
        impl $name {
            pub fn norm2(&self) -> f64 {
                let mut res = 0.0;
                for i in 0..$size {
                    res += self.data[i] * self.data[i];
                }
                res
            }

            pub fn norm(&self) -> f64 {
                self.norm2().sqrt()
            }

            pub fn is_unit_length(&self) -> bool {
                (self.norm() - 1.0).abs() < EPSILON
            }

            pub fn normalize(&self) -> $name {
                let mut res = [0.0; $size];
                let norm = self.norm();
                for i in 0..$size {
                    res[i] = self.data[i] / norm;
                }
                $name::new(res)
            }
            pub fn dot(a: $name, b: $name) -> f64 {
                a * b
            }

            pub fn zeros() -> $name {
                $name::new([0.0; $size])
            }

            pub fn ones() -> $name {
                $name::new([1.0; $size])
            }

            pub fn is_close(a : &$name, b : &$name, epsilon : Option<f64>) -> bool {
                let eps = epsilon.unwrap_or(EPSILON);
                for i in 0..$size {
                    if (a.data[i] - b.data[i]).abs() > eps {
                        return false;
                    }
                }
                true
            }
        }
    };
}

vector_impl!(V1, 1);
vector_impl!(V2, 2);
vector_impl!(V3, 3);
vector_impl!(V4, 4);
vector_impl!(V5,5);
vector_impl!(V6,6);

//implement cross for V3
impl V3{
    pub fn cross(self, rhs: V3) -> V3 {
        let mut res = [0.0; 3];
        res[0] = self[1] * rhs[2] - self[2] * rhs[1];
        res[1] = self[2] * rhs[0] - self[0] * rhs[2];
        res[2] = self[0] * rhs[1] - self[1] * rhs[0];
        V3::new(res)
    }
}

#[cfg(test)]
mod tests{
    use approx::abs_diff_eq;
    use super::*;

    //row1
    #[test]
    pub fn test_add() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([2.0, 2.0, 2.0]);
        let c = V3::new([3.0, 3.0, 3.0]);
        assert_eq!(a + b, c);
        assert_eq!(a + b, b + a);
    }

    #[test]
    pub fn test_sub() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([2.0, 2.0, 2.0]);
        let c = V3::new([-1.0, -1.0, -1.0]);
        assert_eq!(a - b, c);
        assert_eq!(a - b, -(b - a));
    }

    #[test]
    pub fn test_neg() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([-1.0, -1.0, -1.0]);
        assert_eq!(-a, b);
        assert_eq!(-(-a), a);
    }

    #[test]
    pub fn test_mul_scalar() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([2.0, 2.0, 2.0]);
        let c = V3::new([4.0, 4.0, 4.0]);
        assert_eq!(a * 2.0, b);
        assert_eq!(2.0 * b, c);
    }

    #[test]
    pub fn test_mul() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([2.0, 2.0, 2.0]);
        assert_eq!(a * b, 6.0);
        assert_eq!(b * a, 6.0);
    }

    #[test]
    pub fn test_dot() {
        let a = V3::new([1.0, 1.0, 1.0]);
        let b = V3::new([2.0, 2.0, 2.0]);
        assert_eq!(V3::dot(a, b), 6.0);
        assert_eq!(V3::dot(b, a), 6.0);
    }

    #[test]
    pub fn test_norm() {
        let a = V3::new([1.0, 1.0, 1.0]);
        assert_eq!(a.norm2(), 3.0);
        abs_diff_eq!(a.normalize().norm2(), 1.0, epsilon = f64::EPSILON);
    }

    #[test]
    pub fn test_cross() {
        let a = V3::new([1.0, 0.0, 0.0]);
        let b = V3::new([0.0, 1.0, 0.0]);
        let c = V3::new([0.0, 0.0, 1.0]);
        assert_eq!(a.cross(b), c);
        assert_eq!(b.cross(a), -c);
    }
    #[test]
    pub fn test_zeros() {
        let a = V3::zeros();
        let b = V3::new([0.0, 0.0, 0.0]);
        assert_eq!(a, b);
    }

    #[test]
    pub fn test_ones() {
        let a = V3::ones();
        let b = V3::new([1.0, 1.0, 1.0]);
        assert_eq!(a, b);
    }

    #[test]
    pub fn test_display() {
        let a = V3::ones();
        let output : String = a.to_string();
        assert_eq!(output, "[1.00, 1.00, 1.00]");
    }

    #[test]
    pub fn test_index() {
        let mut a = V3::ones();
        a[0] = 2.0;
        a[1] = 4.0;
        a[2] = 6.0;
        assert_eq!(a[0], 2.0);
        assert_eq!(a[1], 4.0);
        assert_eq!(a[2], 6.0);
    }
    
    #[test]
    pub fn test_default() {
        let a = V3::default();
        let b = V3::new([0.0, 0.0, 0.0]);
        assert_eq!(a, b);
    }

    #[test]
    pub fn test_norm_close() {
        let a = V3::new([1.0, 1.0 - EPSILON/2.0, 1.0]);
        let b = V3::new([1.0, 1.0, 1.0]);
        assert!(V3::is_close(&a, &b, None));
    }
}