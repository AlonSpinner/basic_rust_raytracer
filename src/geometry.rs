use crate::vector::{V3,V6};
use crate::matrix::{Matrix33, Matrix44};
use crate::EPSILON;
use std::ops::Mul;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: V3,
    pub direction: V3,
}

impl Ray {
    pub fn new(origin: V3, direction: V3) -> Ray {
        assert!((direction.norm2() - 1.0).abs() < EPSILON);
        Ray {
            origin: origin,
            direction: direction,
        }
    }

    pub fn at(&self, t: f64) -> V3 {
        self.origin + self.direction * t
    }

    pub fn reflect(&self, intersection_point: V3, normal: V3) -> Ray {
        /*
        new_direction = old_direction + change_due_to_intersection
        change_due_to_intersection is only in the normal direction
        we need to remove the component of old_direction in the normal direction, and then 
        add it in the opposite direction. 
        That is:
            change_due_to_intersection = size * normal
        
        Because the incident ray is in the opposite direction of the reflected ray 
        That is:
            dot(new_direction,normal) = -dot(old_direction,normal)
        the size is then:
            size = -2 * dot(old_direction,normal)
        
        so change_due_to_intersection = -2 * dot(old_direction,normal) * normal 
         */
        Ray {
            origin: intersection_point,
            direction: self.direction - normal * 2.0 * V3::dot(self.direction,normal),
        }
    }

    pub fn transmit(&self, intersection_point: V3, normal: V3, n1: f64, n2: f64) -> Option<Ray> {
        //need to understand this better, later
        assert!(normal.is_unit_length());

        let eta = n1 / n2;
        let cos_theta1 = -V3::dot(self.direction, normal);  // Assuming both vectors are normalized
    
        let k = 1.0 - eta * eta * (1.0 - cos_theta1 * cos_theta1);
    
        if k < 0.0 {
            // Total internal reflection
            return None;
        }
    
        let cos_theta2 = k.sqrt();
        let new_direction = eta * self.direction + (eta * cos_theta1 - cos_theta2) * normal;
        
        Some(Ray {
            origin: intersection_point,
            direction: new_direction,
        })
    }

}

impl fmt::Display for Ray {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ray {{ origin: {}, direction: {} }}", self.origin, self.direction)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub center: V3,
    pub normal: V3,
}

impl Plane {
    pub fn new(center: V3, normal: V3) -> Plane {
        assert!(normal.is_unit_length());
        Plane {
            center: center,
            normal: normal,
        }
    }

    pub fn distance_to_plane(&self, point: V3) -> f64 {
        V3::dot(point - self.center, self.normal).abs()
    }

    pub fn on_plane(&self, point: V3) -> bool {
        self.distance_to_plane(point) < EPSILON
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sphere {
    pub center: V3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: V3, radius: f64) -> Sphere {
        Sphere {
            center: center,
            radius: radius,
        }
    }

    pub fn distance_to_sphere(&self, point: V3) -> f64 {
        (point - self.center).norm() - self.radius
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(non_snake_case)]
pub struct SO3 {
    pub R: Matrix33,
}

impl SO3 {
    pub fn new(matrix: Matrix33) -> SO3 {
        if (matrix.det() < 0.0) || !matrix.is_orthogonal(None){
            //orthogonal matrices have determinant 1 or -1
            panic!("Matrix is not orthogonal");}
        SO3 {
            R: matrix,
        }
    }

    pub fn identity() -> SO3 {
        SO3 {
            R: Matrix33::identity(),
        }
    }

    pub fn default() -> SO3 {
        Self::identity()
    }

    pub fn inverse(&self) -> SO3 {
        SO3 {
            R: self.R.transpose(),
        }
    }

    #[allow(non_snake_case)]
    pub fn Exp(tau : V3) -> SO3 {
        let th = tau.norm();
        let theta = tau * th.recip();
        if th < EPSILON {
            return SO3::new(Matrix33::identity());
        } else {
            let hat = SO3::hat(theta);
            let hat2 = hat * hat;
            let R = Matrix33::identity() + th.sin() * hat  + (1.0 - th.cos()) * hat2;
            return SO3::new(R);
        }
    }

    #[allow(non_snake_case)]
    pub fn Log(g : SO3) -> V3 {
        let theta = ((g.R.trace() - 1.0) / 2.0).acos();
        if theta < EPSILON {
            return V3::new([0.0, 0.0, 0.0])
        } else {
            return SO3::vee(g.R - g.R.transpose()) * (theta / (2.0 * theta.sin()));
        }
    }

    fn vee(m33 : Matrix33) -> V3 {
        V3::new([m33[2][1], m33[0][2], m33[1][0]])
    }

    fn hat(v3 : V3) -> Matrix33 {
        Matrix33::new([[0.0, -v3[2], v3[1]],
                       [v3[2], 0.0, -v3[0]],
                       [-v3[1], v3[0], 0.0]])
    }
}

impl Mul for SO3 {
    type Output = SO3;
    fn mul(self, rhs: SO3) -> Self::Output {
        SO3 {
            R: self.R * rhs.R,
        }
    }
}

impl Mul<V3> for SO3 {
    type Output = V3;
    fn mul(self, rhs: V3) -> Self::Output {
        self.R * rhs
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[allow(non_snake_case)]
pub struct SE3 {
    pub r: SO3,
    pub t: V3,
}

impl SE3 {
    pub fn new(r : SO3, t : V3) -> SE3 {
        SE3 {
            r: r,
            t: t,
        }
    }

    #[allow(non_snake_case)]
    pub fn from_eye_target_up(eye : V3, target : V3, up : V3) -> SE3 {
        /*
        follows camera convention, where z is forward, x is right, y is down
        in theory, up is close to negative y, but has some flexability as we are using cross products
         */
        let z = (target - eye).normalize();
        let x = V3::cross(z,up).normalize();
        let y = V3::cross(z,x).normalize();
        let R = Matrix33::new([[x[0], y[0], z[0]],
                               [x[1], y[1], z[1]],
                               [x[2], y[2], z[2]]]);
        SE3 {
            r: SO3::new(R),
            t: eye,
        }
    }

    pub fn identity() -> SE3 {
        SE3 {
            r: SO3::identity(),
            t: V3::new([0.0, 0.0, 0.0]),
        }
    }

    pub fn default() -> SE3 {
        Self::identity()
    }

    #[allow(non_snake_case)]
    pub fn from_matrix(m44 : Matrix44) -> SE3 {
        let R = Matrix33::new([[m44[0][0], m44[0][1], m44[0][2]],
                              [m44[1][0], m44[1][1], m44[1][2]],
                              [m44[2][0], m44[2][1], m44[2][2]]]);
        SE3 {
            r: SO3::new(R),
            t: V3::new([m44[0][3], m44[1][3], m44[2][3]]),
        }
    }

    pub fn inverse(&self) -> SE3 {
        let r_inv = self.r.inverse();
        SE3 {
            r: r_inv,
            t: -(r_inv.R * self.t),
        }
    }

    #[allow(non_snake_case)]
    pub fn Log(g : SE3) -> V6 {
        let theta = SO3::Log(SO3::new(g.r.R));
        let th = theta.norm();
        let hat = SO3::hat(theta);
        let hat2 = hat * hat;
        let V = Matrix33::identity() + 
                        (1.0 -th.cos())/(th*th) * hat +
                        (th - th.sin())/(th*th*th) * hat2;

        let rho = V.invert().unwrap() * g.t;
        return V6::new([theta[0], theta[1], theta[2], rho[0], rho[1], rho[2]]);
    }

    #[allow(non_snake_case)]
    pub fn Exp(tau : V6) -> SE3 {
        let theta = V3::new([tau[0],tau[1],tau[2]]);
        let rho = V3::new([tau[3],tau[4],tau[5]]);

        let th = theta.norm();
        let hat = SO3::hat(theta);
        let hat2 = hat * hat;
        let V = Matrix33::identity() + 
                        (1.0 -th.cos())/(th*th) * hat +
                        (th - th.sin())/(th*th*th) * hat2;
        
        let t = V * rho;
        let r = SO3::Exp(theta);
        return SE3::new(r,t);
    }

    fn _hat(tau : V6) -> Matrix44 {
        Matrix44::new([[0.0, -tau[2], tau[1], tau[3]],
                            [tau[2], 0.0, -tau[0], tau[4]],
                            [-tau[1], tau[0], 0.0, tau[5]],
                            [0.0, 0.0, 0.0, 0.0]])
        }
    fn _vee(m44 : Matrix44) -> V6 {
        V6::new([m44[2][1], m44[0][2], m44[1][0], m44[0][3], m44[1][3], m44[2][3]])
    }
}

impl Mul<V3> for SE3 {
    type Output = V3;
    fn mul(self, rhs: V3) -> Self::Output {
        self.r.R * rhs + self.t
    }
}

impl Mul<SE3> for SE3 {
    type Output = SE3;
    fn mul(self, rhs: SE3) -> Self::Output {
        SE3 {
            r: SO3::new(self.r.R * rhs.r.R),
            t: self.r.R * rhs.t + self.t,
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    const PI : f64 = std::f64::consts::PI;

    #[test]
    pub fn test_so3_new() {
        let theta = V3::new([PI, PI/2.0, PI/4.0]);
        let a = SO3::Exp(theta);
        let b = SO3::new(a.R);
        assert!(Matrix33::is_close(&a.R, &b.R, None).unwrap());

    }

    #[test]
    pub fn test_so3_log_exp() {
        let theta1 = V3::new([PI/5.0, PI/3.0, PI/4.0]);
        let r1 = SO3::Exp(theta1);
        let theta2 = SO3::Log(r1);
        let r2 = SO3::Exp(theta2);
        assert!(V3::is_close(&theta1, &theta2, Some(1e-5f64)));
        assert!(Matrix33::is_close(&r1.R, &r2.R, Some(1e-5f64)).unwrap());
    }

    #[test]
    pub fn test_so3_mul_and_invert() {
        let x1: SO3 = SO3::Exp(V3::new([PI/5.0, PI/3.0, PI/4.0]));
        let u = SO3::Exp(V3::new([PI/2.0, PI/4.0, PI/6.0]));
        let x2 = x1 * u;
        let x3 = x2 * u.inverse();
        assert!(Matrix33::is_close(&x1.R, &x3.R, Some(1e-5f64)).unwrap());

        let u_test = x1.inverse() * x2;
        assert!(Matrix33::is_close(&u.R, &u_test.R, Some(1e-5f64)).unwrap());
    }

    #[test]
    pub fn test_se3_new() {
        let theta = V3::new([PI/8.0, PI/2.0, PI/4.0]);
        let rho = V3::new([1.0, 2.0, 3.0]);
        let a = SE3::Exp(V6::new([theta[0], theta[1], theta[2], rho[0], rho[1], rho[2]]));
        let b = SE3::new(a.r, a.t);
        assert!(Matrix33::is_close(&a.r.R, &b.r.R, None).unwrap());
        assert!(V3::is_close(&a.t, &b.t, None));
    }

    #[test]
    pub fn test_se3_log_exp() {
        let theta = V3::new([PI/8.0, PI/2.0, PI/4.0]);
        let rho = V3::new([1.0, 2.0, 3.0]);
        let g1 = SE3::Exp(V6::new([theta[0], theta[1], theta[2], rho[0], rho[1], rho[2]]));
        let tau = SE3::Log(g1);
        let g2 = SE3::Exp(tau);
        assert!(V6::is_close(&tau, &V6::new([theta[0], theta[1], theta[2], rho[0], rho[1], rho[2]]), Some(1e-5f64)));
        assert!(Matrix33::is_close(&g1.r.R, &g2.r.R, Some(1e-5f64)).unwrap());
        assert!(V3::is_close(&g1.t, &g2.t, Some(1e-5f64)));
    }

    #[test]
    pub fn test_se3_mul_and_invert() {
        let theta = V3::new([PI/8.0, PI/2.0, PI/4.0]);
        let rho = V3::new([1.0, 2.0, 3.0]);
        let x1 = SE3::Exp(V6::new([theta[0], theta[1], theta[2], rho[0], rho[1], rho[2]]));
        let u = SE3::Exp(V6::new([PI/2.0, PI/4.0, PI/6.0, 1.0, 2.0, 3.0]));
        let x2 = x1 * u;
        let x3 = x2 * u.inverse();
        assert!(Matrix33::is_close(&x1.r.R, &x3.r.R, Some(1e-5f64)).unwrap());
        assert!(V3::is_close(&x1.t, &x3.t, Some(1e-5f64)));
    }

    #[test]
    pub fn test_defaults_and_identities() {
        assert!(SO3::default() == SO3::identity());
        assert!(SO3::identity().R == Matrix33::identity());

        assert!(SE3::default() == SE3::identity());
        assert!(SE3::identity().r == SO3::identity());
        assert!(SE3::identity().t == V3::new([0.0, 0.0, 0.0]));
    }
}